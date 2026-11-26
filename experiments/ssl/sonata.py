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
Sonata defaults to the shared ``BevParams`` ROI and voxel size. The training dataset
filters points to that ROI, aggregates XYZ/features per voxel, and passes explicit
``grid_coord`` values to PTv3.

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

import argparse
import logging
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from components.dataset.pcd_only_dataset import PcdOnlyDataset
from components.definitions.bev_params import BevParams
from components.mmperc.encoder.point_transformer_v3 import (
    Point,
    PointGridSpec,
    PointTransformerV3Config,
    PointTransformerV3Encoder,
    adapt_ptv3_config_for_input,
    adapt_ptv3_config_for_output,
    build_neighbors,
    points_to_voxels,
    serialize,
)
from components.utils.config import load_yaml
from components.utils.device import get_device
from components.utils.logger import configure_logger, logger
from components.utils.onnx_export import export_onnx
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger("sonata")

DEFAULT_FEATURE_FIELDS = ("intensity",)


def _default_input_grid() -> PointGridSpec:
    return PointGridSpec.from_bev_params(BevParams())


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _default_backbone() -> PointTransformerV3Config:
    # Sonata-sized PTv3 encoder (xyz + colour + normal = 9 input channels).
    return PointTransformerV3Config(
        in_channels=1,
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 128, 64),
        drop_path=0.3,
        enc_mode=True,
    )


@dataclass
class SonataConfig:
    backbone: PointTransformerV3Config = field(default_factory=_default_backbone)
    input_grid: PointGridSpec | dict | None = field(
        default_factory=_default_input_grid,
    )  # Physical BEV bounds and voxel size; applied to a copy of backbone.
    output_voxel_size_xyz: tuple[float, float, float] | float | None = None  # Optional output-grid voxel size.
    up_cast_level: int = 2  # Loss level: stages below the last encoder stage.
    num_prototypes: int = 4096  # Number of prototype vectors in each prediction head.
    head_hidden: int = 2048  # Hidden width of the per-point prototype heads.
    head_bottleneck: int = 256  # Bottleneck width before prototype prediction.
    mask_ratio: float = 0.4  # Fraction of mask units selected for masking.
    mask_size: float = 0.1  # Mask-unit edge size in world units of ``coord``.
    mask_jitter: float = 0.01  # Random mask-unit position jitter in world units.
    student_temp: float = 0.1  # Student softmax temperature for distillation.
    sinkhorn_iters: int = 3  # Sinkhorn-Knopp iterations used to center teacher logits.
    match_k: int = 8  # Nearest correspondences considered per cross-view point.
    match_radius: float = 0.32  # Maximum correspondence radius on ``origin_coord``.
    max_loss_points: int = 20_000  # Memory cap for up-cast points per loss term.
    mask_loss_weight: float = 2.0  # Weight of the masked-view distillation loss.
    roll_mask_loss_weight: float = 2.0  # Weight of the shifted masked-view loss.
    unmask_loss_weight: float = 2.0  # Weight of the local-to-global distillation loss.

    def __post_init__(self) -> None:
        if isinstance(self.input_grid, dict):
            grid = dict(self.input_grid)
            grid["min_xyz"] = tuple(grid["min_xyz"])
            grid["max_xyz"] = tuple(grid["max_xyz"])
            if isinstance(grid["voxel_size_xyz"], list):
                grid["voxel_size_xyz"] = tuple(grid["voxel_size_xyz"])
            self.input_grid = PointGridSpec(**grid)
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


def _set_export_order(backbone: SonataBackbone) -> None:
    backbone.order = ["z"]
    for stage in backbone.enc:
        for block in stage:
            block.attn.order_index = 0


class SonataBackboneExport(nn.Module):
    """Tensor-only adapter for exporting a Sonata backbone to ONNX.

    Returns the same tensor as the PyTorch visualization path: the encoder features of all
    stages up-cast to the input resolution and concatenated, ``(N, sum(enc_channels))`` in
    input point order.
    """

    def __init__(self, backbone: SonataBackbone):
        super().__init__()
        self.backbone = backbone
        _set_export_order(self.backbone)

    def forward(
        self,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        top = self.backbone(
            {
                "coord": grid_coord.to(dtype=feat.dtype),
                "feat": feat,
                "grid_coord": grid_coord,
                "batch": batch,
            }
        )
        return _full_resolution_features(top, self.backbone.num_stages - 1, feat, grid_coord).feat


def _full_resolution_features(top: Point, up: int, feat: torch.Tensor, grid_coord: torch.Tensor) -> Encoded:
    """``upcast`` for single-cloud tensor-only exports (``up = num_stages - 1`` -> input resolution)."""
    num_input = feat.shape[0]
    zeros = torch.zeros(num_input, dtype=torch.long, device=feat.device)
    return upcast(top, up, grid_coord.to(dtype=feat.dtype), zeros, zeros, None)


class SonataFullExport(nn.Module):
    """Tensor-only adapter exporting the full SonataNet teacher backbone and heads."""

    def __init__(self, net: SonataNet, up_cast_level: int):
        super().__init__()
        self.net = net
        self.up_cast_level = up_cast_level
        _set_export_order(self.net.backbone)

    def forward(
        self,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(mask_logits, unmask_logits, encoded_features)``.

        The logits live at the ``up_cast_level`` resolution; ``encoded_features`` is the
        full-resolution multi-scale tensor, identical to ``SonataBackboneExport`` and to the
        PyTorch visualization path.
        """
        top = self.net.backbone(
            {
                "coord": grid_coord.to(dtype=feat.dtype),
                "feat": feat,
                "grid_coord": grid_coord,
                "batch": batch,
            }
        )
        encoded = _full_resolution_features(top, self.up_cast_level, feat, grid_coord)
        full = _full_resolution_features(top, self.net.backbone.num_stages - 1, feat, grid_coord)
        return (
            self.net.mask_head(encoded.feat),
            self.net.unmask_head(encoded.feat),
            full.feat,
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
    max_steps: int = 100_000  # Total optimizer steps.
    lr: float = 2e-3  # Base learning rate.
    block_lr_scale: float = 0.1  # Transformer-block learning-rate multiplier.
    min_lr: float = 1e-6  # Minimum cosine-schedule learning rate.
    warmup_steps: int = 2_000  # Linear learning-rate warmup steps.
    betas: tuple[float, float] = (0.9, 0.999)  # Adam optimizer beta values.
    weight_decay_start: float = 0.04  # Initial weight decay.
    weight_decay_end: float = 0.2  # Final cosine-scheduled weight decay.
    grad_clip: float = 3.0  # Maximum gradient norm.
    grad_accum_steps: int = 1  # Optimizer steps accumulated across backward passes.
    momentum_start: float = 0.994  # Initial teacher EMA momentum.
    momentum_end: float = 1.0  # Final teacher EMA momentum.
    teacher_temp_start: float = 0.04  # Initial teacher softmax temperature.
    teacher_temp_end: float = 0.07  # Final teacher softmax temperature.
    teacher_temp_warmup_steps: int = 5_000  # Steps used to warm teacher temperature.
    amp: bool = True  # Enable mixed-precision autocast.
    log_every: int = 20  # Training log interval in optimizer steps.
    eval_every: int = 0  # Evaluation interval; zero disables evaluation.
    save_every: int = 2_000  # Checkpoint save interval in optimizer steps.
    out_dir: str = "checkpoints/sonata"  # Checkpoint output directory.
    resume: str | None = None  # Optional trainer checkpoint to resume.


@dataclass
class SonataDataConfig:
    data_dir: str = "./data/gen"  # Folder containing PCD/PLY point clouds.
    points_per_cloud: int = 4096  # Maximum points sampled per view; also sets ONNX input count.
    batch_size: int = 1  # Point clouds per training batch.
    num_workers: int = 4  # CPU data-loading worker processes; zero disables prefetched workers.
    prefetch_factor: int = 2  # Batches prefetched per data-loading worker.
    coord_fields: tuple[str, str, str] = ("x", "y", "z")  # Point-cloud XYZ field names.
    feature_fields: tuple[str, ...] = DEFAULT_FEATURE_FIELDS  # Whitelist of feature channels to load.

    def __post_init__(self) -> None:
        self.coord_fields = tuple(self.coord_fields)
        self.feature_fields = tuple(self.feature_fields)


@dataclass
class SonataExportConfig:
    backbone_path: str = "sonata_teacher_backbone.pt"  # Teacher backbone PyTorch weights output.
    backbone_onnx_path: str = "sonata_teacher_backbone.onnx"  # PTv3-only teacher ONNX output.
    full_path: str = "sonata_full.pt"  # Full teacher SonataNet PyTorch weights output.
    full_onnx_path: str = "sonata_full.onnx"  # Full Sonata ONNX output.


@dataclass
class SonataExperimentConfig:
    model: SonataConfig = field(default_factory=SonataConfig)
    train: SonataTrainConfig = field(default_factory=SonataTrainConfig)
    data: SonataDataConfig = field(default_factory=SonataDataConfig)
    export: SonataExportConfig = field(default_factory=SonataExportConfig)


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


def export_sonata_encoder_pt(model: Sonata, path: str | os.PathLike[str]) -> None:
    """Save the EMA teacher encoder state dict for downstream use."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.teacher.backbone.state_dict(), output_path)


def export_sonata_full_pt(model: Sonata, path: str | os.PathLike[str]) -> None:
    """Save the EMA teacher encoder and both prototype heads."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.teacher.state_dict(), output_path)


def _onnx_sample(
    model: Sonata,
    device: torch.device,
    in_channels: int,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if in_channels < 1:
        raise ValueError("in_channels must be positive")
    if num_points < 1:
        raise ValueError("num_points must be positive")
    side = math.ceil(num_points ** (1 / 3))
    serial_depth = model.teacher.backbone.serial_depth
    if side >= 1 << serial_depth:
        raise ValueError(
            f"num_points={num_points} requires a cubic grid side of {side}, which exceeds serial_depth={serial_depth}"
        )
    point_ids = torch.arange(num_points, device=device)
    grid = torch.stack(
        (point_ids // (side * side), (point_ids // side) % side, point_ids % side),
        dim=1,
    )
    feat = torch.zeros((num_points, in_channels), dtype=torch.float32, device=device)
    batch = torch.zeros(num_points, dtype=torch.long, device=device)
    return feat, grid.to(dtype=torch.long), batch


def export_sonata_encoder_onnx(
    model: Sonata,
    path: str | os.PathLike[str],
    in_channels: int,
    num_points: int = 4096,
) -> None:
    """Export the EMA teacher encoder with a fixed number of input points."""
    device = next(model.parameters()).device
    feat, grid_coord, batch = _onnx_sample(model, device, in_channels, num_points)
    exported = SonataBackboneExport(model.teacher.backbone).to(device)
    export_onnx(
        exported,
        (feat, grid_coord, batch),
        path,
        input_names=("features", "grid_coord", "batch"),
        output_names=("encoded_features",),
        dynamic_axes={},
    )


def export_sonata_full_onnx(
    model: Sonata,
    path: str | os.PathLike[str],
    in_channels: int,
    num_points: int = 4096,
) -> None:
    """Export the EMA teacher encoder and both prototype heads."""
    device = next(model.parameters()).device
    feat, grid_coord, batch = _onnx_sample(model, device, in_channels, num_points)
    exported = SonataFullExport(
        model.teacher,
        up_cast_level=model.cfg.up_cast_level,
    ).to(device)
    export_onnx(
        exported,
        (feat, grid_coord, batch),
        path,
        input_names=("features", "grid_coord", "batch"),
        output_names=("mask_logits", "unmask_logits", "encoded_features"),
        dynamic_axes={},
    )


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
            dict(
                model=self.model.state_dict(),
                model_config=asdict(self.model.cfg),
                optimizer=self.opt.state_dict(),
                step=self.step,
                epoch=self.epoch,
            ),
            tmp,
        )
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.step, self.epoch = ckpt["step"], ckpt.get("epoch", 0)

    def export_backbone(self, path: str) -> None:
        """Teacher backbone weights (what Sonata releases for downstream fine-tuning / probing)."""
        export_sonata_encoder_pt(self.model, path)

    def export_full(self, path: str) -> None:
        """Teacher backbone and prototype-head weights."""
        export_sonata_full_pt(self.model, path)

    def export_backbone_onnx(self, path: str, in_channels: int, num_points: int = 4096) -> None:
        """Export only the teacher PTv3 backbone with a fixed number of input points."""
        export_sonata_encoder_onnx(self.model, path, in_channels, num_points)

    def export_full_onnx(self, path: str, in_channels: int, num_points: int = 4096) -> None:
        """Export the teacher backbone and both Sonata prototype heads."""
        export_sonata_full_onnx(self.model, path, in_channels, num_points)

    def _onnx_sample(self, in_channels: int, num_points: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _onnx_sample(self.model, self.device, in_channels, num_points)


class SonataPCDDataset(Dataset):
    """Convert unstructured PCD/PLY arrays into two global views and one local view."""

    def __init__(
        self,
        root_dir: str,
        points_per_cloud: int = 4096,
        grid_size: float | None = None,
        coord_fields: tuple[str, str, str] = ("x", "y", "z"),
        feature_fields: tuple[str, ...] = DEFAULT_FEATURE_FIELDS,
        input_grid: PointGridSpec | None = None,
    ):
        if points_per_cloud < 1:
            raise ValueError("points_per_cloud must be positive")
        if grid_size is not None and (not math.isfinite(grid_size) or grid_size <= 0):
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
        self.input_grid = input_grid or _default_input_grid()
        if grid_size is not None:
            self.input_grid = replace(self.input_grid, voxel_size_xyz=grid_size)
        self.jitter_size = min(self.input_grid.voxel_size) * 0.1

        field_to_idx = {field.lower(): idx for idx, field in enumerate(self.fields)}
        coord_names = tuple(name.strip().lower() for name in coord_fields)
        missing = [name for name in coord_names if name not in field_to_idx]
        if missing:
            raise ValueError(
                f"Point-cloud fields {self.fields} do not include coordinate fields {coord_fields}; "
                "use --coord-fields to specify the XYZ field names"
            )
        self.coord_names = coord_names

        # Validate that all requested feature fields are available in the point cloud
        feature_names = tuple(name.strip().lower() for name in feature_fields)
        available_fields = set(field.lower() for field in self.fields)
        missing_features = [name for name in feature_names if name not in available_fields]
        if missing_features:
            raise ValueError(
                f"Point-cloud fields {self.fields} do not include requested feature fields {missing_features}; "
                f"available fields are {sorted(available_fields)}"
            )

        self.feature_fields = tuple(field for field in self.fields if field.lower() in feature_names)
        self.in_channels = len(self.feature_fields) or 3
        self._warned_schemas: set[tuple[str, ...]] = set()
        if first_cloud.ndim != 2 or first_cloud.shape[1] != len(self.fields):
            raise ValueError(
                f"Point-cloud field metadata does not match data shape: "
                f"fields={len(self.fields)}, shape={first_cloud.shape}"
            )

    def __len__(self):
        return len(self.dataset)

    def _field_indexes(self, array, fields: tuple[str, ...], idx: int) -> dict[str, int]:
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
        return field_to_idx

    def _voxelize(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        path: os.PathLike[str] | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        voxel_input = torch.cat((coordinates, features), dim=1)
        voxel_data, is_real = points_to_voxels(
            voxel_input,
            torch.zeros(len(coordinates), dtype=torch.long),
            self.input_grid,
            batch_size=1,
            feature_dim=voxel_input.shape[1],
        )
        if not is_real.any():
            raise ValueError(f"Point cloud at {path} contains no points inside the configured input grid")
        return (
            voxel_data["coord"][is_real],
            voxel_data["feat"][is_real, 3:],
            voxel_data["grid_coord"][is_real],
        )

    def __getitem__(self, idx: int) -> dict:
        """Return the sample at ``idx``; unusable files are logged and replaced by the next usable one."""
        error: Exception | None = None
        for offset in range(len(self)):
            current = (idx + offset) % len(self)
            try:
                return self._load_sample(current)
            except (OSError, ValueError) as exc:
                error = exc
                logger.warning("Skipping point cloud %s: %s", self.paths[current], exc)
        raise RuntimeError("No usable point cloud in the dataset") from error

    def _load_sample(self, idx: int) -> dict:
        array, fields = self.dataset[idx]
        field_to_idx = self._field_indexes(array, fields, idx)

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
            features = torch.nan_to_num(features[valid], nan=0.0, posinf=0.0, neginf=0.0)
            features = (features - features.mean(dim=0, keepdim=True)) / features.std(
                dim=0, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            features = features.clamp(-10, 10)
        else:
            features = coordinates

        coordinates, features, grid_coord = self._voxelize(coordinates, features, self.paths[idx])
        origin = coordinates - coordinates.mean(dim=0, keepdim=True)
        count = origin.shape[0]
        if count > self.points_per_cloud:
            indices = torch.randperm(count)[: self.points_per_cloud]
            origin, features, grid_coord = origin[indices], features[indices], grid_coord[indices]

        def make_view(indices: torch.Tensor) -> dict:
            base_coord = origin[indices]
            coord = base_coord + torch.randn_like(base_coord) * self.jitter_size
            return {
                "coord": coord,
                "origin_coord": base_coord,
                "feat": features[indices],
                "grid_coord": grid_coord[indices],
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
    """Collate point clouds without padding; per-point batch IDs distinguish scenes."""
    if len(items) == 0:
        raise ValueError("Expected at least one point cloud per batch")

    if len(items) == 1:
        return items[0]

    # Batch multiple clouds into one model input.
    batched = {"global_views": [], "local_views": []}

    # Process global views
    global_views = [item["global_views"] for item in items]
    max_num_global = max(len(gv) for gv in global_views)
    for view_idx in range(max_num_global):
        view_batch = [gv[view_idx] if view_idx < len(gv) else None for gv in global_views]
        batched["global_views"].append(_collate_views(view_batch))

    # Process local views
    local_views = [item["local_views"] for item in items]
    max_num_local = max(len(lv) for lv in local_views) if local_views else 0
    for view_idx in range(max_num_local):
        view_batch = [lv[view_idx] if view_idx < len(lv) else None for lv in local_views]
        batched["local_views"].append(_collate_views(view_batch))

    return batched


def _collate_views(views: list[dict | None]) -> dict:
    """Concatenate present scene views and identify each scene by its item index."""
    present_views = [(batch_idx, view) for batch_idx, view in enumerate(views) if view is not None]
    if not present_views:
        raise ValueError("Expected at least one point cloud view")

    result = {}
    all_keys = set().union(*(view.keys() for _, view in present_views))
    for key in all_keys:
        reference = next(view[key] for _, view in present_views if key in view)
        tensors = []
        for batch_idx, view in present_views:
            point_count = view["coord"].shape[0]
            if key in view:
                tensor = view[key]
            elif key == "origin_coord":
                tensor = view["coord"]
            else:
                tensor = reference.new_zeros((point_count,) + reference.shape[1:])

            if key == "batch":
                tensor = torch.full((point_count,), batch_idx, dtype=torch.long, device=tensor.device)
            else:
                if tensor.shape[0] != point_count:
                    raise ValueError(f"View field {key!r} must contain one entry per coordinate")

            tensors.append(tensor)

        result[key] = torch.cat(tensors, dim=0)
    return result


def build_train_loader(
    data_dir: str,
    points_per_cloud: int = 4096,
    batch_size: int = 1,
    grid_size: float | None = None,
    coord_fields: tuple[str, str, str] = ("x", "y", "z"),
    feature_fields: tuple[str, ...] = DEFAULT_FEATURE_FIELDS,
    input_grid: PointGridSpec | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> tuple[SonataPCDDataset, DataLoader]:
    """Build a shuffled training loader from recursively collected PCD/PLY files.

    Args:
        data_dir: Directory containing PCD/PLY point clouds
        points_per_cloud: Maximum points sampled per cloud
        batch_size: Number of point clouds per batch (supports variable batch sizes per GPU)
        num_workers: CPU worker processes used to load and prepare point clouds
        prefetch_factor: Batches prepared in advance per worker
        pin_memory: Pin host tensors for asynchronous CUDA transfers
        grid_size: Optional override for voxel size
        coord_fields: Names of coordinate fields (x, y, z)
        feature_fields: Whitelist of feature field names to load
        input_grid: Optional grid specification
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")

    dataset = SonataPCDDataset(
        data_dir,
        points_per_cloud,
        grid_size,
        coord_fields,
        feature_fields,
        input_grid,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": _single_item_collate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**loader_kwargs)
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


def _validate_run_config(config: SonataExperimentConfig) -> None:
    if config.train.max_steps < 1:
        raise ValueError("configured train.max_steps must be positive")
    if config.data.points_per_cloud < 1:
        raise ValueError("configured data.points_per_cloud must be positive")
    if config.data.batch_size < 1:
        raise ValueError("configured data.batch_size must be positive")
    if config.data.num_workers < 0:
        raise ValueError("configured data.num_workers must be non-negative")
    if config.data.prefetch_factor < 1:
        raise ValueError("configured data.prefetch_factor must be positive")


def _prepare_model_config(
    config: SonataExperimentConfig,
    args: argparse.Namespace,
) -> tuple[SonataConfig, int]:
    model_config = config.model
    points_per_cloud = config.data.points_per_cloud
    if args.small_gpu:
        model_config = _small_gpu_profile(model_config)
        points_per_cloud = min(points_per_cloud, 1024)
    config.model = model_config
    return model_config, points_per_cloud


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--small-gpu",
        action="store_true",
        help="Use a compact model and at most 1024 points per cloud for small-GPU test runs.",
    )
    parser.add_argument(
        "--path-config",
        dest="path_config",
        type=str,
        default="./experiments/ssl/sonata_config.yml",
        help="Path to the Sonata experiment YAML config.",
    )
    args = parser.parse_args()
    cfg = load_yaml(Path(args.path_config), SonataExperimentConfig)
    _validate_run_config(cfg)
    model_cfg, points_per_cloud = _prepare_model_config(cfg, args)
    configure_logger("sonata")

    device = get_device()

    dataset, train_loader = build_train_loader(
        cfg.data.data_dir,
        points_per_cloud=points_per_cloud,
        batch_size=cfg.data.batch_size,
        coord_fields=cfg.data.coord_fields,
        feature_fields=cfg.data.feature_fields,
        input_grid=model_cfg.input_grid,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        pin_memory=device.type == "cuda",
    )
    cfg.model = replace(
        model_cfg,
        backbone=replace(model_cfg.backbone, in_channels=dataset.in_channels),
    )
    trainer = SonataTrainer(Sonata(cfg.model), train_loader, cfg.train, device=device)
    logger.info(
        "Training on %d point-cloud files from %s (shuffled); feature fields=%s, input channels=%d, batch_size=%d",
        len(dataset),
        cfg.data.data_dir,
        dataset.feature_fields,
        dataset.in_channels,
        cfg.data.batch_size,
    )
    trainer.fit()
    trainer.export_backbone(cfg.export.backbone_path)
    logger.info("Exported teacher backbone to %s", cfg.export.backbone_path)
    if cfg.export.full_path:
        trainer.export_full(cfg.export.full_path)
        logger.info("Exported full teacher weights to %s", cfg.export.full_path)
    if cfg.export.backbone_onnx_path:
        trainer.export_backbone_onnx(
            cfg.export.backbone_onnx_path,
            in_channels=dataset.in_channels,
            num_points=points_per_cloud,
        )
        logger.info("Exported PTv3-only teacher ONNX to %s", cfg.export.backbone_onnx_path)
    if cfg.export.full_onnx_path:
        trainer.export_full_onnx(
            cfg.export.full_onnx_path,
            in_channels=dataset.in_channels,
            num_points=points_per_cloud,
        )
        logger.info("Exported full Sonata ONNX to %s", cfg.export.full_onnx_path)
