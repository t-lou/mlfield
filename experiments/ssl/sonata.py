"""Sonata draft (Wu et al., CVPR 2025, "Sonata: Self-Supervised Learning of Reliable Point
Representations") on top of the pure-PyTorch PTv3 encoder in ``point_transformer_v3.py``.

Method in one paragraph
-----------------------
Self-distillation (DINO / DINOv2 style) with a PTv3 *encoder only* (no decoder).
  * teacher = EMA of the student, sees the unmasked global views;
  * student sees (a) the global views with masked "mask units" (features replaced by a
    learned mask token right after the stem) and (b) unmasked local crops;
  * features of the last encoder stage are *up-cast* (parameter-free unpooling + concat with
    the skip features) to a coarser-than-finest level (``up_cast_level``) and the loss is
    computed there, which removes the local-geometry shortcut of point-level distillation;
  * per-point prototype heads (``mask_head`` / ``unmask_head``), Sinkhorn-Knopp centering
    on the teacher side, three losses:
        mask loss       student masked view v  <-> teacher view v      (masked points only)
        roll-mask loss  student masked view v  <-> teacher view v+1    (via correspondences)
        unmask loss     student local views    <-> teacher global views (via correspondences)

Data contract
-------------
Any iterable (e.g. ``torch.utils.data.DataLoader`` with your own collate_fn) yielding dicts::

    {
      "global_views": [view, view, ...],   # >= 2, same scenes in the same order in every view
      "local_views":  [view, ...],         # optional, may be empty
    }

where each ``view`` is a batched point cloud dict::

    coord        (N, 3) float   augmented coordinates of this view (world units, e.g. metres)
    feat         (N, C) float   C == backbone.in_channels
    grid_coord   (N, 3) long    integer voxel indices   (or give ``grid_size`` instead)
    batch (N,) long   or   offset (B,) long             scene index of every point
    origin_coord (N, 3) float   coordinates *before* view-specific augmentation (rotation,
                                scale, flip, jitter, ...), i.e. in a frame shared by all
                                views of one scene. Used for cross-view correspondences.
                                Falls back to ``coord`` (only valid if views are not
                                geometrically augmented relative to each other).

Every (view, scene) pair must contain at least one point.

The CLI training adapter treats the configured XYZ fields as coordinates and uses
the other fields from the first point cloud as feature channels, excluding common
label fields by default. Later point clouds are aligned by field name; missing
fields are zero-filled and additional fields are ignored.

Usage
-----
    model = Sonata(SonataConfig())
    trainer = SonataTrainer(model, loader, SonataTrainConfig(max_steps=100_000))
    trainer.fit()
    trainer.export_backbone("sonata_teacher_backbone.pt")   # teacher weights for downstream use

Geometry configuration
----------------------
    grid = PointGridSpec(
        min_xyz=(-50.0, -50.0, -5.0),
        max_xyz=(50.0, 50.0, 5.0),
        voxel_size_xyz=(0.1, 0.1, 0.1),
    )
    model = Sonata(SonataConfig(
        input_grid=grid,
        output_voxel_size_xyz=(0.8, 0.8, 0.8),
    ))

``input_grid`` adapts the PTv3 serialization depth to the physical XYZ bounds.
``output_voxel_size_xyz`` optionally adapts integer stage strides. PTv3 currently
supports one scalar stride per stage shared by X, Y, and Z; therefore output scaling
must be isotropic. An anisotropic output grid requires per-axis GridPooling support.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from components.dataset.pcd_only_dataset import PcdOnlyDataset
from components.mmperc.encoder.point_transformer_v3 import (
    Point,
    PointGridSpec,
    PointTransformerV3Config,
    PointTransformerV3Encoder,
    adapt_ptv3_config_for_input,
    adapt_ptv3_config_for_output,
    build_neighbors,
    serialize,
)
from components.utils.device import get_device
from components.utils.logger import configure_logger, logger
from components.utils.onnx_export import export_onnx
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger("sonata")

DEFAULT_IGNORED_FIELDS = (
    "label",
    "labels",
    "semantic",
    "semantics",
    "semantic_label",
    "semantic_id",
    "class",
    "class_id",
    "instance",
    "instance_id",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _default_backbone() -> PointTransformerV3Config:
    # Sonata-sized PTv3 encoder (xyz + colour + normal = 9 input channels).
    return PointTransformerV3Config(
        in_channels=9,
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024,) * 5,
        drop_path=0.3,
        enc_mode=True,
    )


@dataclass
class SonataConfig:
    backbone: PointTransformerV3Config = field(default_factory=_default_backbone)
    # Optional physical geometry. Both geometry fields are applied to a copy of backbone
    # during construction: input_grid adapts serialization depth, and
    # output_voxel_size_xyz adapts stage strides for isotropic output scaling.
    input_grid: PointGridSpec | dict | None = None
    output_voxel_size_xyz: tuple[float, float, float] | float | None = None
    up_cast_level: int = 2  # loss is computed this many stages below the last one
    num_prototypes: int = 4096
    head_hidden: int = 2048
    head_bottleneck: int = 256
    # masking (in world units of ``coord``)
    mask_ratio: float = 0.4
    mask_size: float = 0.1
    mask_jitter: float = 0.01
    # distillation
    student_temp: float = 0.1
    sinkhorn_iters: int = 3
    # cross-view correspondences (on origin_coord of up-cast points)
    match_k: int = 8
    match_radius: float = 0.32
    # memory cap: at most this many up-cast points per loss term go through the heads / losses
    max_loss_points: int = 20_000
    # loss weights
    mask_loss_weight: float = 2.0
    roll_mask_loss_weight: float = 2.0
    unmask_loss_weight: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.input_grid, dict):
            self.input_grid = PointGridSpec(**self.input_grid)
        if self.output_voxel_size_xyz is not None and self.input_grid is None:
            raise ValueError("Sonata output_voxel_size_xyz requires input_grid")

        backbone = self.backbone
        if self.input_grid is not None:
            backbone = adapt_ptv3_config_for_input(backbone, self.input_grid)
        if self.output_voxel_size_xyz is not None:
            backbone = adapt_ptv3_config_for_output(
                backbone,
                self.input_grid,
                self.output_voxel_size_xyz,
            )
        self.backbone = backbone


# --------------------------------------------------------------------------- #
# Backbone with mask-token support
# --------------------------------------------------------------------------- #
class SonataBackbone(PointTransformerV3Encoder):
    """PTv3 encoder (enc_mode) + learned mask token injected after the stem."""

    def __init__(self, config: PointTransformerV3Config):
        super().__init__(replace(config, enc_mode=True))
        self.mask_token = nn.Parameter(torch.zeros(config.enc_channels[0]))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, data: dict, mask: torch.Tensor | None = None) -> Point:
        """``mask`` (N,) bool in input point order; True -> feature replaced by mask token."""
        point = self._prepare(data)
        grid_bound = 1 << self.serial_depth
        with torch.no_grad():
            nbr_stem = build_neighbors(point.grid_coord, point.batch, self.stem_kernel, grid_bound)
        point.feat = F.gelu(self.embed_norm(self.embed_conv(point.feat, nbr_stem)))
        if mask is not None:
            point.feat = torch.where(mask[:, None], self.mask_token.to(point.feat.dtype), point.feat)

        for s in range(self.num_stages):
            if s > 0:
                point = self.pools[s - 1](point)
            serialize(point, self.order, self.serial_depth, self.shuffle_orders and self.training)
            point.nbr = build_neighbors(point.grid_coord, point.batch, 3, grid_bound)
            for blk in self.enc[s]:
                point = blk(point)
        return point


class SonataBackboneExport(nn.Module):
    """Tensor-only adapter for exporting a Sonata backbone to ONNX."""

    def __init__(self, backbone: SonataBackbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.order = ["z"]

    def forward(
        self,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        point = self.backbone(
            {
                "coord": grid_coord.to(dtype=feat.dtype),
                "feat": feat,
                "grid_coord": grid_coord,
                "batch": batch,
            }
        )
        return point.feat


class SonataFullExport(nn.Module):
    """Tensor-only adapter exporting the full SonataNet teacher backbone and heads."""

    def __init__(self, net: SonataNet, up_cast_level: int):
        super().__init__()
        self.net = net
        self.up_cast_level = up_cast_level
        self.net.backbone.order = ["z"]

    def forward(
        self,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top = self.net.backbone(
            {
                "coord": grid_coord.to(dtype=feat.dtype),
                "feat": feat,
                "grid_coord": grid_coord,
                "batch": batch,
            }
        )
        num_input = feat.shape[0]
        encoded = upcast(
            top,
            self.up_cast_level,
            grid_coord.to(dtype=feat.dtype),
            torch.zeros(num_input, dtype=torch.long, device=feat.device),
            torch.zeros(num_input, dtype=torch.long, device=feat.device),
            None,
        )
        return (
            self.net.mask_head(encoded.feat),
            self.net.unmask_head(encoded.feat),
        )


class PrototypeHead(nn.Module):
    """MLP -> L2-norm -> cosine logits against L2-normalised prototypes (DINO-style)."""

    def __init__(self, in_dim: int, hidden: int, bottleneck: int, num_prototypes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, bottleneck) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.mlp(x), dim=-1)
        w = F.normalize(self.prototypes, dim=-1)
        return z @ w.t()  # (M, K) in [-1, 1]; temperatures are applied in the loss


class SonataNet(nn.Module):
    """Backbone + the two prototype heads. Student and teacher are both instances of this."""

    def __init__(self, cfg: SonataConfig, drop_path: float | None = None):
        super().__init__()
        bb = replace(cfg.backbone, enc_mode=True)
        if drop_path is not None:
            bb = replace(bb, drop_path=drop_path)
        n = len(bb.enc_channels)
        if not 0 <= cfg.up_cast_level <= n - 1:
            raise ValueError(f"up_cast_level must be in [0, {n - 1}]")
        self.backbone = SonataBackbone(bb)
        in_dim = sum(bb.enc_channels[n - 1 - cfg.up_cast_level :])  # skip feats are concatenated
        make = lambda: PrototypeHead(in_dim, cfg.head_hidden, cfg.head_bottleneck, cfg.num_prototypes)  # noqa: E731
        self.mask_head, self.unmask_head = make(), make()


# --------------------------------------------------------------------------- #
# Helpers: views, masking, up-casting, matching, Sinkhorn
# --------------------------------------------------------------------------- #
def _view_batch(view: dict) -> torch.Tensor:
    if "batch" in view:
        return view["batch"].long()
    off = view["offset"].long()
    counts = torch.diff(off, prepend=off.new_zeros(1))
    return torch.repeat_interleave(torch.arange(off.numel(), device=off.device), counts)


def concat_views(views: list[dict]):
    """Merge views into one batch (each (view, scene) pair becomes its own cloud)."""
    batches = [_view_batch(v) for v in views]
    n_scene = max(int(b.max()) + 1 for b in batches)
    grid = [
        PointTransformerV3Encoder._prepare_grid_coord(v, v["coord"], b) for v, b in zip(views, batches, strict=True)
    ]
    view_id = torch.cat([torch.full_like(b, i) for i, b in enumerate(batches)])
    scene = torch.cat(batches)
    data = dict(
        coord=torch.cat([v["coord"] for v in views]),
        feat=torch.cat([v["feat"] for v in views]),
        grid_coord=torch.cat(grid),
        batch=view_id * n_scene + scene,
    )
    origin = torch.cat([v.get("origin_coord", v["coord"]) for v in views])
    return data, origin, view_id, scene


@torch.no_grad()
def generate_mask(coord: torch.Tensor, batch: torch.Tensor, ratio: float, size: float, jitter: float) -> torch.Tensor:
    """Mask whole cubic units of side ``size`` (random grid origin per cloud, fuzzy borders)."""
    noisy = coord + torch.randn_like(coord) * jitter
    nb = int(batch.max()) + 1
    mn = torch.full((nb, 3), float("inf"), device=coord.device, dtype=coord.dtype)
    mn = mn.scatter_reduce(0, batch[:, None].expand(-1, 3), noisy, "amin")
    shift = torch.rand(nb, 3, device=coord.device, dtype=coord.dtype) * size
    ijk = torch.floor((noisy - mn[batch] + shift[batch]) / size).long()
    S = int(ijk.max()) + 1
    key = ((batch * S + ijk[:, 0]) * S + ijk[:, 1]) * S + ijk[:, 2]
    uniq, inv = torch.unique(key, return_inverse=True)
    unit_masked = torch.rand(uniq.numel(), device=coord.device) < ratio
    return unit_masked[inv]


@dataclass
class Encoded:
    feat: torch.Tensor  # (M, C_up) up-cast features at the coarse level
    origin: torch.Tensor  # (M, 3)   mean origin_coord of the descendants
    view: torch.Tensor  # (M,)
    scene: torch.Tensor  # (M,)
    mask: torch.Tensor | None  # (M,) bool, coarse point counts as masked if > 50% of its points are


def upcast(top: Point, up: int, origin, view_id, scene, mask) -> Encoded:
    """Unpool ``up`` stages (concat with skip features) and map per-point extras to that level."""
    chain = [top]
    while "pooling_parent" in chain[-1]:
        chain.append(chain[-1].pooling_parent)
    chain = chain[::-1]  # finest ... coarsest
    n = len(chain)
    t = n - 1 - up

    feat = chain[-1].feat
    for s in range(n - 1, t, -1):
        feat = torch.cat([chain[s - 1].feat, feat[chain[s].pooling_inverse]], dim=-1)

    m = torch.arange(chain[0].feat.shape[0], device=feat.device)  # finest -> level t
    for s in range(1, t + 1):
        m = chain[s].pooling_inverse[m]
    M = chain[t].feat.shape[0]
    cnt = torch.bincount(m, minlength=M).clamp_min(1).to(origin.dtype)
    origin_c = torch.zeros(M, 3, device=origin.device, dtype=origin.dtype).index_add_(0, m, origin) / cnt[:, None]
    view_c = torch.zeros(M, dtype=torch.long, device=feat.device)
    view_c[m] = view_id
    scene_c = torch.zeros_like(view_c)
    scene_c[m] = scene
    mask_c = None
    if mask is not None:
        frac = torch.zeros(M, device=feat.device).index_add_(0, m, mask.float()) / cnt.float()
        mask_c = frac > 0.5
    return Encoded(feat, origin_c, view_c, scene_c, mask_c)


def _dist_sum(x: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(x)
    return x


@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor, temp: float, iters: int = 3) -> torch.Tensor:
    """Teacher targets (M, K): rows sum to 1, prototype usage balanced (DINOv2 centering)."""
    logits = logits.float()
    Q = torch.exp((logits - logits.max()) / temp).t()  # (K, M)
    K = Q.shape[0]
    world = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    B = Q.shape[1] * world
    Q /= _dist_sum(Q.sum())
    for _ in range(iters):
        Q /= _dist_sum(Q.sum(dim=1, keepdim=True))
        Q /= K
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()


@torch.no_grad()
def match_points(oa, sa, ob, sb, k: int, radius: float, chunk: int = 4096):
    """k nearest neighbours of every a-point among b-points of the same scene (within radius)."""
    ia_out, ib_out = [], []
    for s in torch.unique(sa):
        ia = (sa == s).nonzero().squeeze(1)
        ib = (sb == s).nonzero().squeeze(1)
        if ib.numel() == 0:
            continue
        kk = min(k, ib.numel())
        for i in range(0, ia.numel(), chunk):
            idx = ia[i : i + chunk]
            d = torch.cdist(oa[idx].float(), ob[ib].float())
            dist, nn_idx = d.topk(kk, dim=1, largest=False)
            keep = dist <= radius
            ia_out.append(idx[:, None].expand_as(nn_idx)[keep])
            ib_out.append(ib[nn_idx][keep])
    if not ia_out:
        e = torch.empty(0, dtype=torch.long, device=oa.device)
        return e, e
    return torch.cat(ia_out), torch.cat(ib_out)


@torch.no_grad()
def aggregate_targets(q: torch.Tensor, ia, ib, n_a: int, chunk: int = 16384):
    """Mean teacher distribution over the matched neighbours of every student point."""
    tgt = q.new_zeros(n_a, q.shape[1])
    cnt = q.new_zeros(n_a)
    for i in range(0, ia.numel(), chunk):
        a, b = ia[i : i + chunk], ib[i : i + chunk]
        tgt.index_add_(0, a, q[b])
        cnt.index_add_(0, a, torch.ones_like(a, dtype=q.dtype))
    return tgt, cnt


def soft_ce(logits: torch.Tensor, target: torch.Tensor, temp: float) -> torch.Tensor:
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    return -(target * F.log_softmax(logits.float() / temp, dim=-1)).sum(-1).mean()


# --------------------------------------------------------------------------- #
# Sonata
# --------------------------------------------------------------------------- #
class Sonata(nn.Module):
    def __init__(self, cfg: SonataConfig | None = None):
        super().__init__()
        self.cfg = cfg or SonataConfig()
        self.student = SonataNet(self.cfg)
        self.teacher = SonataNet(self.cfg, drop_path=0.0)
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher.requires_grad_(False)

    # -- encoding ----------------------------------------------------------- #
    def _encode(self, net: SonataNet, views: list[dict], masked: bool) -> Encoded:
        data, origin, view_id, scene = concat_views(views)
        mask = None
        if masked:
            c = self.cfg
            mask = generate_mask(data["coord"], data["batch"], c.mask_ratio, c.mask_size, c.mask_jitter)
        top = net.backbone(data, mask)
        return upcast(top, self.cfg.up_cast_level, origin, view_id, scene, mask)

    def _matched_loss(self, s_logits, q_t, o_s, sc_s, o_t, sc_t) -> torch.Tensor | None:
        c = self.cfg
        ia, ib = match_points(o_s, sc_s, o_t, sc_t, c.match_k, c.match_radius)
        tgt, cnt = aggregate_targets(q_t, ia, ib, s_logits.shape[0])
        valid = cnt > 0
        if not valid.any():
            return None
        return soft_ce(s_logits[valid], tgt[valid] / cnt[valid, None], c.student_temp)

    @staticmethod
    def _mean(losses: list[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        return torch.stack(losses).mean() if losses else ref.sum() * 0.0

    # -- forward ------------------------------------------------------------ #
    def forward(self, batch: dict, teacher_temp: float) -> dict[str, torch.Tensor]:
        c = self.cfg
        gviews, lviews = batch["global_views"], batch.get("local_views") or []
        Vg = len(gviews)
        if Vg < 2:
            raise ValueError("Sonata needs at least two global views")

        with torch.no_grad():
            t = self._encode(self.teacher, gviews, masked=False)
            q_mask = sinkhorn_knopp(self.teacher.mask_head(t.feat), teacher_temp, c.sinkhorn_iters)
            q_unmask = sinkhorn_knopp(self.teacher.unmask_head(t.feat), teacher_temp, c.sinkhorn_iters)

        # student, masked global views (same point/pooling order as the teacher's)
        sg = self._encode(self.student, gviews, masked=True)

        # masked points (optionally sub-sampled) -> mask head; shared by mask and roll-mask loss
        sel = self._subsample(sg.mask.nonzero().squeeze(1))
        s_mask_logits = self.student.mask_head(sg.feat[sel])
        sel_view, sel_scene, sel_origin = sg.view[sel], sg.scene[sel], sg.origin[sel]

        # 1) mask loss: same view, masked points only
        l_mask = soft_ce(s_mask_logits, q_mask[sel], c.student_temp)

        # 2) roll-mask loss: masked points of view v vs teacher view v+1
        roll = []
        for v in range(Vg):
            w = (v + 1) % Vg
            ids = (sel_view == v).nonzero().squeeze(1)
            idt = (t.view == w).nonzero().squeeze(1)
            if ids.numel() == 0:
                continue
            loss = self._matched_loss(
                s_mask_logits[ids], q_mask[idt], sel_origin[ids], sel_scene[ids], t.origin[idt], t.scene[idt]
            )
            if loss is not None:
                roll.append(loss)
        l_roll = self._mean(roll, s_mask_logits)

        # 3) unmask loss: local views vs every teacher global view
        # (the unmask head always participates so the parameter set is identical on every step)
        l_unmask = sum(p.sum() for p in self.student.unmask_head.parameters()) * 0.0
        if lviews:
            sl = self._encode(self.student, lviews, masked=False)
            keep = self._subsample(torch.arange(sl.feat.shape[0], device=sl.feat.device))
            s_un_logits = self.student.unmask_head(sl.feat[keep])
            l_view, l_scene, l_origin = sl.view[keep], sl.scene[keep], sl.origin[keep]
            un = []
            for l_ in range(len(lviews)):
                ids = (l_view == l_).nonzero().squeeze(1)
                if ids.numel() == 0:
                    continue
                for w in range(Vg):
                    idt = (t.view == w).nonzero().squeeze(1)
                    loss = self._matched_loss(
                        s_un_logits[ids], q_unmask[idt], l_origin[ids], l_scene[ids], t.origin[idt], t.scene[idt]
                    )
                    if loss is not None:
                        un.append(loss)
            if un:
                l_unmask = l_unmask + torch.stack(un).mean()

        total = c.mask_loss_weight * l_mask + c.roll_mask_loss_weight * l_roll + c.unmask_loss_weight * l_unmask
        # collapse monitors: sharp targets (low entropy) and balanced prototype usage (-> 1.0)
        tgt_entropy = -(q_mask * q_mask.clamp_min(1e-12).log()).sum(-1).mean()
        usage = q_mask.mean(0)
        usage_ppl = torch.exp(-(usage * usage.clamp_min(1e-12).log()).sum()) / usage.numel()
        return dict(
            loss=total,
            mask=l_mask.detach(),
            roll_mask=l_roll.detach(),
            unmask=l_unmask.detach(),
            target_entropy=tgt_entropy,
            proto_usage=usage_ppl,
        )

    @torch.no_grad()
    def extract_features(self, view: dict, up_levels: int | None = None, use_teacher: bool = True) -> Encoded:
        """Downstream features for one batched view (same dict format as in training).

        Runs the (teacher by default) backbone in eval mode and unpools ``up_levels`` stages
        (default: all the way to the input resolution). With the default, ``result.feat`` is
        (N, sum(enc_channels)) in the *input point order*. Training mode is restored afterwards.
        """
        net = (self.teacher if use_teacher else self.student).backbone
        was_training = net.training
        net.eval()
        try:
            data, origin, view_id, scene = concat_views([view])
            top = net(data)
            up = len(net.config.enc_channels) - 1 if up_levels is None else up_levels
            return upcast(top, up, origin, view_id, scene, None)
        finally:
            net.train(was_training)

    def _subsample(self, idx: torch.Tensor) -> torch.Tensor:
        cap = self.cfg.max_loss_points
        if cap <= 0 or idx.numel() <= cap:
            return idx
        return idx[torch.randperm(idx.numel(), device=idx.device)[:cap]]

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        t_params = list(self.teacher.parameters())
        s_params = [p.detach() for p in self.student.parameters()]
        torch._foreach_mul_(t_params, momentum)
        torch._foreach_add_(t_params, s_params, alpha=1.0 - momentum)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class SonataTrainConfig:
    max_steps: int = 100_000
    lr: float = 2e-3
    block_lr_scale: float = 0.1  # transformer blocks (backbone.enc.*) train with lr * this
    min_lr: float = 1e-6
    warmup_steps: int = 2_000
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.2
    grad_clip: float = 3.0
    grad_accum_steps: int = 1
    momentum_start: float = 0.994  # teacher EMA, cosine to momentum_end
    momentum_end: float = 1.0
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_steps: int = 5_000
    amp: bool = True  # mixed-precision autocast
    log_every: int = 20
    eval_every: int = 0  # 0 = never; calls ``eval_fn(model, step)`` every this many steps
    save_every: int = 2_000
    out_dir: str = "checkpoints/sonata"
    resume: str | None = None


def _cosine(start: float, end: float, t: float) -> float:
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


class SonataTrainer:
    """Single-process trainer. ``loader`` is any iterable yielding the dicts described above."""

    def __init__(
        self,
        model: Sonata,
        loader: Iterable[dict],
        cfg: SonataTrainConfig,
        device: str | torch.device | None = None,
        eval_fn: Callable[[Sonata, int], dict | None] | None = None,
    ):
        """``eval_fn(model, step) -> dict | None`` is e.g. a linear probe built on
        ``model.extract_features``; it runs every ``cfg.eval_every`` steps and its result is logged."""
        if device is None:
            device = get_device()
        self.model = model.to(device)
        self.eval_fn = eval_fn
        self.loader, self.cfg, self.device = loader, cfg, torch.device(device)
        self.step = 0
        self.epoch = 0
        self.opt = torch.optim.AdamW(self._param_groups(), lr=cfg.lr, betas=cfg.betas)
        if cfg.resume:
            self.load(cfg.resume)

    # -- setup -------------------------------------------------------------- #
    def _param_groups(self) -> list[dict]:
        groups: dict[tuple[bool, bool], list] = {}
        for name, p in self.model.student.named_parameters():
            groups.setdefault((name.startswith("backbone.enc."), p.ndim > 1), []).append(p)
        return [
            dict(
                params=ps,
                lr_scale=self.cfg.block_lr_scale if is_block else 1.0,
                apply_wd=decay,
                weight_decay=0.0,
            )
            for (is_block, decay), ps in groups.items()
        ]

    def _schedules(self, step: int) -> tuple[float, float, float]:
        c = self.cfg
        if step < c.warmup_steps:
            lr = c.lr * (step + 1) / c.warmup_steps
        else:
            lr = _cosine(c.lr, c.min_lr, (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps))
        frac = step / max(1, c.max_steps)
        wd = _cosine(c.weight_decay_start, c.weight_decay_end, frac)
        mom = _cosine(c.momentum_start, c.momentum_end, frac)
        for g in self.opt.param_groups:
            g["lr"] = lr * g["lr_scale"]
            g["weight_decay"] = wd if g["apply_wd"] else 0.0
        tt = c.teacher_temp_start + (c.teacher_temp_end - c.teacher_temp_start) * min(
            1.0, step / max(1, c.teacher_temp_warmup_steps)
        )
        return lr, mom, tt

    # -- loop --------------------------------------------------------------- #
    def _next_batch(self, it):
        try:
            return next(it), it
        except StopIteration:
            self.epoch += 1
            sampler = getattr(self.loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            it = iter(self.loader)
            return next(it), it

    def fit(self) -> None:
        c = self.cfg
        self.model.train()
        use_amp = c.amp
        amp_dtype = torch.float16 if self.device.type == "mps" else torch.bfloat16
        it = iter(self.loader)
        student_params = list(self.model.student.parameters())

        while self.step < c.max_steps:
            lr, mom, tt = self._schedules(self.step)
            self.opt.zero_grad(set_to_none=True)
            logs: dict[str, float] = {}
            for _ in range(c.grad_accum_steps):
                batch, it = self._next_batch(it)
                batch = to_device(batch, self.device)
                with torch.autocast(self.device.type, dtype=amp_dtype, enabled=use_amp):
                    out = self.model(batch, teacher_temp=tt)
                (out["loss"] / c.grad_accum_steps).backward()
                for k, v in out.items():
                    logs[k] = logs.get(k, 0.0) + float(v.detach()) / c.grad_accum_steps

            grad_norm = torch.nn.utils.clip_grad_norm_(student_params, c.grad_clip)
            if torch.isfinite(grad_norm):
                self.opt.step()
                self.model.ema_update(mom)
            else:
                logger.warning("step %d: non-finite grad norm, update skipped", self.step)
            self.step += 1

            if self.step % c.log_every == 0:
                logger.info(
                    "step %d | loss %.4f (mask %.4f roll %.4f unmask %.4f) | H_tgt %.2f usage %.2f | gn %.2f | lr %.2e | ema %.5f | T_t %.3f",
                    self.step, logs["loss"], logs["mask"], logs["roll_mask"], logs["unmask"],
                    logs["target_entropy"], logs["proto_usage"], float(grad_norm), lr, mom, tt,
                )  # fmt: skip
            if self.eval_fn is not None and c.eval_every and self.step % c.eval_every == 0:
                self.model.eval()
                try:
                    with torch.no_grad():
                        res = self.eval_fn(self.model, self.step)
                finally:
                    self.model.train()
                if res:
                    logger.info("eval @ step %d: %s", self.step, res)
            if self.step % c.save_every == 0 or self.step == c.max_steps:
                self.save(os.path.join(c.out_dir, f"step_{self.step:07d}.pt"))

    # -- io ----------------------------------------------------------------- #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save(
            dict(model=self.model.state_dict(), optimizer=self.opt.state_dict(), step=self.step, epoch=self.epoch), tmp
        )
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.step, self.epoch = ckpt["step"], ckpt.get("epoch", 0)

    def export_backbone(self, path: str) -> None:
        """Teacher backbone weights (what Sonata releases for downstream fine-tuning / probing)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.teacher.backbone.state_dict(), path)

    def export_backbone_onnx(self, path: str, in_channels: int, num_points: int = 4096) -> None:
        """Export only the teacher PTv3 backbone with a fixed number of input points."""
        if num_points < 1:
            raise ValueError("num_points must be positive")
        feat, grid_coord, batch = self._onnx_sample(in_channels, num_points)
        exported = SonataBackboneExport(self.model.teacher.backbone).to(self.device)
        export_onnx(
            exported,
            (feat, grid_coord, batch),
            path,
            input_names=("features", "grid_coord", "batch"),
            output_names=("encoded_features",),
            dynamic_axes={},
        )

    def export_full_onnx(self, path: str, in_channels: int, num_points: int = 4096) -> None:
        """Export the teacher backbone and both Sonata prototype heads."""
        if num_points < 1:
            raise ValueError("num_points must be positive")
        feat, grid_coord, batch = self._onnx_sample(in_channels, num_points)
        exported = SonataFullExport(
            self.model.teacher,
            up_cast_level=self.model.cfg.up_cast_level,
        ).to(self.device)
        export_onnx(
            exported,
            (feat, grid_coord, batch),
            path,
            input_names=("features", "grid_coord", "batch"),
            output_names=("mask_logits", "unmask_logits"),
            dynamic_axes={},
        )

    def _onnx_sample(self, in_channels: int, num_points: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        side = math.ceil(num_points ** (1 / 3))
        serial_depth = self.model.teacher.backbone.serial_depth
        if side >= 1 << serial_depth:
            raise ValueError(
                f"num_points={num_points} requires a cubic grid side of {side}, "
                f"which exceeds serial_depth={serial_depth}"
            )
        point_ids = torch.arange(num_points, device=self.device)
        grid = torch.stack(
            (point_ids // (side * side), (point_ids // side) % side, point_ids % side),
            dim=1,
        )
        feat = torch.zeros((num_points, in_channels), dtype=torch.float32, device=self.device)
        batch = torch.zeros(num_points, dtype=torch.long, device=self.device)
        return feat, grid.to(dtype=torch.long), batch


class SonataPCDDataset(Dataset):
    """Convert unstructured PCD/PLY arrays into two global views and one local view."""

    def __init__(
        self,
        root_dir: str,
        points_per_cloud: int = 4096,
        grid_size: float = 0.02,
        coord_fields: tuple[str, str, str] = ("x", "y", "z"),
        ignore_fields: tuple[str, ...] = DEFAULT_IGNORED_FIELDS,
    ):
        if points_per_cloud < 1:
            raise ValueError("points_per_cloud must be positive")
        if not math.isfinite(grid_size) or grid_size <= 0:
            raise ValueError("grid_size must be positive and finite")
        if len(coord_fields) != 3:
            raise ValueError("coord_fields must contain exactly three field names")

        self.dataset = PcdOnlyDataset(root_dirs=[root_dir], include_fields=True)
        if len(self.dataset) == 0:
            raise ValueError(f"No PCD or PLY files found in test data directory: {root_dir}")

        first_cloud, self.fields = self.dataset[0]
        self.fields = tuple(self.fields)
        self.paths = self.dataset.datasets[0].pcd_paths
        self.points_per_cloud = points_per_cloud
        self.grid_size = grid_size

        field_to_idx = {field.lower(): idx for idx, field in enumerate(self.fields)}
        coord_names = tuple(name.strip().lower() for name in coord_fields)
        missing = [name for name in coord_names if name not in field_to_idx]
        if missing:
            raise ValueError(
                f"Point-cloud fields {self.fields} do not include coordinate fields {coord_fields}; "
                "use --coord-fields to specify the XYZ field names"
            )
        self.coord_names = coord_names
        ignored_names = {name.strip().lower() for name in ignore_fields}
        self.feature_fields = tuple(
            field for field in self.fields if field.lower() not in coord_names and field.lower() not in ignored_names
        )
        self.in_channels = len(self.feature_fields) or 3
        self._warned_schemas: set[tuple[str, ...]] = set()
        if first_cloud.ndim != 2 or first_cloud.shape[1] != len(self.fields):
            raise ValueError(
                f"Point-cloud field metadata does not match data shape: "
                f"fields={len(self.fields)}, shape={first_cloud.shape}"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        array, fields = self.dataset[idx]
        field_to_idx = {field.lower(): i for i, field in enumerate(fields)}
        schema = tuple(field.lower() for field in fields)
        expected_fields = {field.lower() for field in self.fields}
        if set(schema) != expected_fields and schema not in self._warned_schemas:
            missing = sorted(expected_fields - set(schema))
            extra = sorted(set(schema) - expected_fields)
            logger.warning(
                "Point cloud %s has a different field schema; missing fields will be zero-filled and extra fields ignored "
                "(missing=%s, extra=%s)",
                self.paths[idx],
                missing,
                extra,
            )
            self._warned_schemas.add(schema)
        if array.ndim != 2 or array.shape[1] != len(fields):
            raise ValueError(f"Point-cloud data shape {array.shape} does not match its {len(fields)} fields")

        row_count = array.shape[0]
        coordinates = torch.zeros((row_count, 3), dtype=torch.float32)
        for axis, coord_field in enumerate(self.coord_names):
            source_idx = field_to_idx.get(coord_field)
            if source_idx is None:
                raise ValueError(f"Point cloud at {self.paths[idx]} is missing coordinate field {coord_field!r}")
            coordinates[:, axis] = torch.as_tensor(array[:, source_idx], dtype=torch.float32)
        valid = torch.isfinite(coordinates).all(dim=1)
        coordinates = coordinates[valid]
        if coordinates.shape[0] == 0:
            raise ValueError(f"Point cloud at {self.paths[idx]} contains no finite XYZ coordinates")

        if self.feature_fields:
            features = torch.zeros((row_count, len(self.feature_fields)), dtype=torch.float32)
            for column, field in enumerate(self.feature_fields):
                source_idx = field_to_idx.get(field.lower())
                if source_idx is not None:
                    features[:, column] = torch.as_tensor(array[:, source_idx], dtype=torch.float32)
            features = torch.nan_to_num(features[valid])
            features = (features - features.mean(dim=0, keepdim=True)) / features.std(
                dim=0, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            features = features.clamp(-10, 10)
        else:
            features = coordinates

        origin = coordinates - coordinates.mean(dim=0, keepdim=True)
        count = origin.shape[0]
        if count > self.points_per_cloud:
            indices = torch.randperm(count)[: self.points_per_cloud]
            origin, features = origin[indices], features[indices]

        def make_view(indices: torch.Tensor) -> dict:
            base_coord = origin[indices]
            coord = base_coord + torch.randn_like(base_coord) * (self.grid_size * 0.1)
            return {
                "coord": coord,
                "origin_coord": base_coord,
                "feat": features[indices],
                "grid_size": self.grid_size,
                "batch": torch.zeros(indices.numel(), dtype=torch.long),
            }

        all_indices = torch.arange(origin.shape[0])
        local_count = max(1, all_indices.numel() // 2)
        local_indices = all_indices[torch.randperm(all_indices.numel())[:local_count]]
        return {
            "global_views": [make_view(all_indices), make_view(all_indices)],
            "local_views": [make_view(local_indices)],
        }


def _single_item_collate(items: list[dict]) -> dict:
    if len(items) != 1:
        raise ValueError(f"Expected one point cloud per batch, got {len(items)}")
    return items[0]


def build_train_loader(
    data_dir: str,
    points_per_cloud: int = 4096,
    grid_size: float = 0.02,
    coord_fields: tuple[str, str, str] = ("x", "y", "z"),
    ignore_fields: tuple[str, ...] = DEFAULT_IGNORED_FIELDS,
) -> tuple[SonataPCDDataset, DataLoader]:
    """Build a shuffled training loader from recursively collected PCD/PLY files."""
    dataset = SonataPCDDataset(data_dir, points_per_cloud, grid_size, coord_fields, ignore_fields)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_single_item_collate)
    return dataset, loader


def _smoke_test() -> None:
    """Run a tiny Sonata training step on the best available device."""
    backbone = PointTransformerV3Config(
        in_channels=3,
        order=("z",),
        enc_depths=(1, 1),
        enc_channels=(4, 8),
        enc_num_head=(1, 2),
        enc_patch_size=(8, 8),
        stride=(2,),
        drop_path=0.0,
        serial_depth=8,
        stem_kernel=3,
        enc_mode=True,
    )
    cfg = SonataConfig(
        backbone=backbone,
        up_cast_level=1,
        num_prototypes=4,
        head_hidden=8,
        head_bottleneck=4,
        mask_ratio=1.0,
        mask_jitter=0.0,
        match_radius=1.0,
        max_loss_points=16,
    )

    device = get_device()
    torch.manual_seed(0)
    grid = torch.cartesian_prod(*(torch.arange(3) for _ in range(3))).to(device)
    coord = grid.float() * 0.1
    feat = torch.rand(grid.shape[0], backbone.in_channels, device=device)
    view = dict(
        coord=coord,
        feat=feat,
        grid_coord=grid,
        batch=torch.zeros(grid.shape[0], dtype=torch.long, device=device),
    )
    model = Sonata(cfg).to(device)
    losses = model({"global_views": [view, view], "local_views": [view]}, teacher_temp=0.04)
    losses["loss"].backward()

    assert all(torch.isfinite(value).all() for value in losses.values()), "Sonata smoke test produced non-finite losses"
    assert any(p.grad is not None for p in model.student.parameters()), (
        "Sonata smoke test produced no student gradients"
    )
    print(f"Sonata smoke test passed ({device}).")


def _small_gpu_profile(cfg: SonataConfig) -> SonataConfig:
    """Reduce model and loss memory for short test runs on small GPUs."""
    backbone = replace(
        cfg.backbone,
        order=("z",),
        stride=(2, 2, 2),
        enc_depths=(1, 1, 1, 1),
        enc_channels=(16, 32, 64, 128),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(256,) * 4,
        stem_kernel=3,
    )
    if cfg.input_grid is not None:
        backbone = adapt_ptv3_config_for_input(backbone, cfg.input_grid)
    if cfg.output_voxel_size_xyz is not None:
        backbone = adapt_ptv3_config_for_output(
            backbone,
            cfg.input_grid,
            cfg.output_voxel_size_xyz,
        )
    return replace(
        cfg,
        backbone=backbone,
        up_cast_level=min(cfg.up_cast_level, 3),
        num_prototypes=min(cfg.num_prototypes, 256),
        head_hidden=min(cfg.head_hidden, 256),
        head_bottleneck=min(cfg.head_bottleneck, 64),
        max_loss_points=2_048,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run a tiny forward/backward smoke test")
    parser.add_argument(
        "--small-gpu",
        action="store_true",
        help="use a compact model and at most 1024 points per cloud for small-GPU test runs",
    )
    parser.add_argument("--cfg", type=str, default=None, help="path to a SonataConfig JSON file")
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default="./data/gen",
        help="folder containing PCD/PLY point clouds (searched recursively)",
    )
    parser.add_argument("--steps", type=int, default=1000, help="number of optimizer steps")
    parser.add_argument(
        "--points-per-cloud",
        type=int,
        default=4096,
        help="maximum points sampled per point-cloud view and fixed ONNX input point count",
    )
    parser.add_argument("--grid-size", type=float, default=0.02, help="voxel size in coordinate units")
    parser.add_argument(
        "--coord-fields",
        type=str,
        default="x,y,z",
        help="comma-separated names of the point-cloud coordinate fields",
    )
    parser.add_argument(
        "--ignore-fields",
        type=str,
        default=",".join(DEFAULT_IGNORED_FIELDS),
        help="comma-separated point-cloud fields excluded from features (default: common labels)",
    )
    parser.add_argument("--input-grid-min", type=str, default=None, help="input grid XYZ minimum: x,y,z")
    parser.add_argument("--input-grid-max", type=str, default=None, help="input grid XYZ maximum: x,y,z")
    parser.add_argument(
        "--input-grid-voxel",
        type=str,
        default=None,
        help="input grid voxel size: scalar or vx,vy,vz",
    )
    parser.add_argument(
        "--output-grid-voxel",
        type=str,
        default=None,
        help="isotropic output voxel size: scalar or vx,vy,vz",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/sonata")
    parser.add_argument("--resume", type=str, default=None, help="resume from a Sonata trainer checkpoint")
    parser.add_argument(
        "--out", type=str, default="sonata_teacher_backbone.pt", help="output path for the teacher backbone weights"
    )
    parser.add_argument(
        "--onnx-out",
        type=str,
        default="sonata_teacher_backbone.onnx",
        help="output path for PTv3-only teacher ONNX model (set empty to disable)",
    )
    parser.add_argument(
        "--full-onnx-out",
        type=str,
        default="sonata_full.onnx",
        help="output path for full Sonata ONNX model (backbone + prototype heads; set empty to disable)",
    )
    args = parser.parse_args()

    if args.smoke:
        _smoke_test()
        raise SystemExit(0)
    if args.steps < 1:
        parser.error("--steps must be positive")

    if args.cfg is None:
        cfg = SonataConfig()
    else:
        with open(args.cfg, encoding="utf-8") as config_file:
            cfg_dict = json.load(config_file)
        backbone = cfg_dict.get("backbone")
        if isinstance(backbone, dict):
            cfg_dict["backbone"] = PointTransformerV3Config(**backbone)
        cfg = SonataConfig(**cfg_dict)
    if (args.input_grid_min is None) != (args.input_grid_max is None):
        parser.error("--input-grid-min and --input-grid-max must be provided together")
    if args.input_grid_min is not None and args.input_grid_voxel is None:
        parser.error("--input-grid-min, --input-grid-max, and --input-grid-voxel must be provided together")
    if args.output_grid_voxel is not None and (args.input_grid_min is None or args.input_grid_voxel is None):
        parser.error("--output-grid-voxel requires --input-grid-min, --input-grid-max, and --input-grid-voxel")
    if args.input_grid_min is not None:

        def parse_xyz(value: str, name: str) -> tuple[float, float, float]:
            parts = tuple(float(part.strip()) for part in value.split(","))
            if len(parts) != 3:
                parser.error(f"--{name} must contain exactly three comma-separated numbers")
            return parts

        def parse_voxel(value: str, name: str) -> tuple[float, float, float] | float:
            parts = tuple(float(part.strip()) for part in value.split(","))
            if len(parts) == 1:
                return parts[0]
            if len(parts) == 3:
                return parts
            parser.error(f"--{name} must contain one scalar or three comma-separated numbers")

        input_grid = PointGridSpec(
            min_xyz=parse_xyz(args.input_grid_min, "input-grid-min"),
            max_xyz=parse_xyz(args.input_grid_max, "input-grid-max"),
            voxel_size_xyz=parse_voxel(args.input_grid_voxel, "input-grid-voxel"),
        )
        output_voxel = (
            None if args.output_grid_voxel is None else parse_voxel(args.output_grid_voxel, "output-grid-voxel")
        )
        cfg = replace(
            cfg,
            input_grid=input_grid,
            output_voxel_size_xyz=output_voxel,
        )
    points_per_cloud = args.points_per_cloud
    if args.small_gpu:
        cfg = _small_gpu_profile(cfg)
        points_per_cloud = min(points_per_cloud, 1024)
    configure_logger("sonata")

    device = get_device()

    coord_fields = tuple(field.strip() for field in args.coord_fields.split(","))
    dataset, train_loader = build_train_loader(
        args.test_data_dir,
        points_per_cloud=points_per_cloud,
        grid_size=args.grid_size,
        coord_fields=coord_fields,
        ignore_fields=tuple(field.strip() for field in args.ignore_fields.split(",") if field.strip()),
    )
    cfg = replace(cfg, backbone=replace(cfg.backbone, in_channels=dataset.in_channels))
    trainer_cfg = SonataTrainConfig(
        max_steps=args.steps,
        out_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    trainer = SonataTrainer(Sonata(cfg), train_loader, trainer_cfg, device=device)
    logger.info(
        "Training on %d point-cloud files from %s (shuffled); feature fields=%s, input channels=%d",
        len(dataset),
        args.test_data_dir,
        dataset.feature_fields,
        dataset.in_channels,
    )
    trainer.fit()
    trainer.export_backbone(args.out)
    logger.info("Exported teacher backbone to %s", args.out)
    if args.onnx_out:
        trainer.export_backbone_onnx(
            args.onnx_out,
            in_channels=dataset.in_channels,
            num_points=points_per_cloud,
        )
        logger.info("Exported PTv3-only teacher ONNX to %s", args.onnx_out)
    if args.full_onnx_out:
        trainer.export_full_onnx(
            args.full_onnx_out,
            in_channels=dataset.in_channels,
            num_points=points_per_cloud,
        )
        logger.info("Exported full Sonata ONNX to %s", args.full_onnx_out)
