"""Point Transformer V3 (PTv3) in pure PyTorch (encoder + decoder).

No spconv / flash-attn / torch_scatter required. Follows the official PTv3 design
(Wu et al., CVPR 2024, "Point Transformer V3: Simpler, Faster, Stronger"):

  1. Point serialization (Z-order, Hilbert and their "-trans" variants), orders shuffled
     across blocks during training.
  2. Patch attention on the serialized order (patch padding re-uses the previous patch).
  3. xCPE: sparse 3x3x3 submanifold conv -> Linear -> LayerNorm, added residually.
  4. Stem: sparse 5x5x5 submanifold conv + BN + GELU.
  5. Serialization-based grid pooling (max-pool of projected features) and a U-Net
     decoder with grid unpooling (skip projection + upsampled projection).

Submanifold sparse convs are implemented by neighbour lookup (sorted hash keys +
searchsorted) followed by gather + matmul, i.e. they are mathematically equal to
spconv.SubMConv3d (stride 1).

Optional, off by default (so the default model matches the canonical one):
  * qk_norm: LayerNorm on per-head q and k (attention-logit stabilisation, as used in
    later ViT work). Not part of the original PTv3.

Input dict:
  coord      (N, 3) float
  feat       (N, C_in) float
  grid_coord (N, 3) long   integer voxel indices (preferred; one point per voxel), or
  grid_size  float         voxel size; grid_coord = floor((coord - min_per_cloud) / grid_size)
  batch      (N,) long     or  offset (B,) long (cumulative point counts)

Output: Point (dict with attribute access). `point.feat` is (N, dec_channels[0]) in the
input point order (or the last encoder stage if enc_mode=True).
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Point container
# --------------------------------------------------------------------------- #
class Point(dict):
    """dict with attribute access: point.feat, point.coord, point.batch, ..."""

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _spread3(x: torch.Tensor) -> torch.Tensor:
    """Spread the lower 21 bits of x so that there are two zero bits between each."""
    x = x & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def _interleave(x0, x1, x2):
    return (_spread3(x0) << 2) | (_spread3(x1) << 1) | _spread3(x2)


def z_order_encode(gc: torch.Tensor, depth: int) -> torch.Tensor:
    gc = gc.long()
    return _interleave(gc[:, 0], gc[:, 1], gc[:, 2])


def hilbert_encode(gc: torch.Tensor, depth: int) -> torch.Tensor:
    """3D Hilbert index via Skilling's transpose algorithm (vectorized, out-of-place)."""
    X = [gc[:, i].long() for i in range(3)]
    M = 1 << (depth - 1)
    # inverse undo excess work
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(3):
            has = (X[i] & Q) != 0
            if i == 0:
                X[0] = torch.where(has, X[0] ^ P, X[0])  # invert
            else:
                x0_inv = torch.where(has, X[0] ^ P, X[0])  # invert
                t = torch.where(has, torch.zeros_like(X[0]), (X[0] ^ X[i]) & P)  # exchange
                X[0] = x0_inv ^ t
                X[i] = X[i] ^ t
        Q >>= 1
    # Gray encode
    X[1] = X[1] ^ X[0]
    X[2] = X[2] ^ X[1]
    t = torch.zeros_like(X[0])
    Q = M
    while Q > 1:
        t = torch.where((X[2] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    X = [x ^ t for x in X]
    return _interleave(*X)


@torch.no_grad()
def serialize(point: Point, order: Sequence[str], depth: int, shuffle_orders: bool) -> Point:
    """Compute serialization orders + per-stage patch bookkeeping (cloud sizes, caches)."""
    assert depth <= 21, "depth > 21 overflows int64 Morton codes"
    gc, batch = point.grid_coord.long(), point.batch.long()
    assert int(gc.min()) >= 0 and int(gc.max()) < (1 << depth), "grid_coord out of range for depth"
    assert int(batch.max()).bit_length() + 3 * depth <= 62, "batch index + code does not fit in int64"
    codes = []
    for o in order:
        g = gc[:, [2, 1, 0]] if o.endswith("-trans") else gc
        base = o.replace("-trans", "")
        if base == "z":
            c = z_order_encode(g, depth)
        elif base == "hilbert":
            c = hilbert_encode(g, depth)
        else:
            raise ValueError(f"unknown order {o}")
        codes.append((batch << (3 * depth)) | c)  # batch in the high bits keeps clouds contiguous
    codes = torch.stack(codes)
    if shuffle_orders:
        codes = codes[torch.randperm(codes.shape[0], device=codes.device)]
    perm = torch.argsort(codes, dim=1)
    inverse = torch.empty_like(perm).scatter_(1, perm, torch.arange(perm.shape[1], device=perm.device).expand_as(perm))
    point.serial_order, point.serial_inverse = perm, inverse
    # per-stage patch bookkeeping; clouds appear in ascending batch id in the sorted order
    sizes = torch.bincount(batch)
    sizes = sizes[sizes > 0]
    point.cloud_sizes = sizes
    point.min_cloud = int(sizes.min())
    point.layout_cache = {}
    return point


# --------------------------------------------------------------------------- #
# Submanifold sparse convolution via neighbour lookup
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_neighbors(grid_coord: torch.Tensor, batch: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """(N, k^3) indices of the neighbours among occupied voxels; N marks 'missing'."""
    assert kernel_size % 2 == 1
    N = grid_coord.shape[0]
    p = kernel_size // 2
    gc = grid_coord.long() + p
    b = batch.long()
    S = int(gc.max()) + p + 1

    def _key(bb, c):
        return ((bb * S + c[:, 0]) * S + c[:, 1]) * S + c[:, 2]

    sk, perm = _key(b, gc).sort()
    r = torch.arange(-p, p + 1, device=gc.device)
    offs = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    out = []
    for o in offs:
        q = _key(b, gc + o)
        pos = torch.searchsorted(sk, q).clamp_(max=N - 1)
        found = sk[pos] == q
        out.append(torch.where(found, perm[pos], torch.full_like(pos, N)))
    return torch.stack(out, 1).to(grid_coord.device)


class SubMConv3d(nn.Module):
    """Submanifold sparse conv (stride 1) == spconv.SubMConv3d, via gather + matmul."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.in_dim, self.out_dim, self.k3 = in_dim, out_dim, kernel_size**3
        self.weight = nn.Parameter(torch.empty(self.k3 * in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.normal_(self.weight, std=(2.0 / (self.k3 * in_dim)) ** 0.5)

    def forward(self, feat: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        padded = torch.cat([feat, feat.new_zeros(1, feat.shape[1])], 0).to(feat.device)
        x = padded[nbr].reshape(feat.shape[0], -1)  # (N, k^3 * C)
        out = x @ self.weight.to(feat.device)
        return out if self.bias is None else out + self.bias


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    """Stochastic depth, per point (as in the official code, which applies timm's DropPath to (N, C))."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = x.new_empty(x.shape[0], 1).bernoulli_(keep) / keep
        return x * mask


class MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


def _pad_unpad(bc: torch.Tensor, ps: int):
    """bc: (B,) cloud sizes, all >= ps. Pad every cloud to a multiple of ps by re-using the previous patch."""
    dev, B = bc.device, bc.numel()
    bc_pad = (bc + ps - 1) // ps * ps
    off = F.pad(bc.cumsum(0), (1, 0))
    off_pad = F.pad(bc_pad.cumsum(0), (1, 0))
    ids = torch.arange(B, device=dev)
    cid_pad = torch.repeat_interleave(ids, bc_pad)
    local = torch.arange(int(off_pad[-1]), device=dev) - off_pad[cid_pad]
    local = torch.where(local >= bc[cid_pad], local - ps, local)  # reuse previous patch
    pad = off[cid_pad] + local
    cid = torch.repeat_interleave(ids, bc)
    unpad = torch.arange(int(off[-1]), device=dev) - off[cid] + off_pad[cid]
    return pad, unpad


@torch.no_grad()
def patch_layout(point: Point, max_patch: int, order_index: int):
    """Cached (patch size, gather order into padded patches, scatter-back index)."""
    ps = min(max_patch, point.min_cloud)  # a patch can't exceed the smallest cloud
    cache = point.layout_cache
    key = (order_index, ps)
    if key not in cache:
        if ("pad", ps) not in cache:
            cache[("pad", ps)] = _pad_unpad(point.cloud_sizes, ps)
        pad, unpad = cache[("pad", ps)]
        cache[key] = (
            point.serial_order[order_index][pad],
            unpad[point.serial_inverse[order_index]],
        )
    order, inverse = cache[key]
    return ps, order, inverse


class SerializedAttention(nn.Module):
    def __init__(self, dim, heads, patch_size, order_index, qk_norm=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.patch_size, self.order_index = dim, heads, patch_size, order_index
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        hd = dim // heads
        self.q_norm = nn.LayerNorm(hd) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(hd) if qk_norm else nn.Identity()
        self.attn_drop, self.proj_drop = attn_drop, nn.Dropout(proj_drop)

    def forward(self, feat: torch.Tensor, point: Point) -> torch.Tensor:
        C = feat.shape[1]
        ps, order, inverse = patch_layout(point, self.patch_size, self.order_index)
        qkv = self.qkv(feat)[order]  # (N_pad, 3C)
        qkv = qkv.reshape(-1, ps, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (P, H, ps, d)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(-1, C)[inverse]
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        patch_size,
        order_index,
        mlp_ratio=4.0,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.cpe_conv = SubMConv3d(dim, dim, 3, bias=True)
        self.cpe_lin = nn.Linear(dim, dim)
        self.cpe_norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SerializedAttention(dim, heads, patch_size, order_index, qk_norm, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), proj_drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, point: Point) -> Point:
        point.feat = point.feat + self.cpe_norm(self.cpe_lin(self.cpe_conv(point.feat, point.nbr)))
        point.feat = point.feat + self.drop_path(self.attn(self.norm1(point.feat), point))
        point.feat = point.feat + self.drop_path(self.mlp(self.norm2(point.feat)))
        return point


class GridPooling(nn.Module):
    """Voxel pooling: Linear -> max over cells -> BN -> GELU; coords are averaged."""

    def __init__(self, in_dim, out_dim, stride=2):
        super().__init__()
        assert stride >= 1
        self.stride = stride
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, point: Point) -> Point:
        gc = point.grid_coord.long() // self.stride
        b = point.batch.long()
        S = int(gc.max()) + 1
        key = ((b * S + gc[:, 0]) * S + gc[:, 1]) * S + gc[:, 2]
        uniq, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
        M = uniq.shape[0]

        feat = self.proj(point.feat)
        pooled = torch.zeros(M, feat.shape[1], device=feat.device, dtype=feat.dtype).scatter_reduce(
            0, inv[:, None].expand_as(feat), feat, reduce="amax", include_self=False
        )
        coord = torch.zeros(M, 3, device=feat.device, dtype=point.coord.dtype).index_add_(0, inv, point.coord)
        coord = coord / counts[:, None].to(coord.dtype)
        new_gc = torch.zeros(M, 3, dtype=torch.long, device=feat.device)
        new_gc[inv] = gc
        new_b = torch.zeros(M, dtype=torch.long, device=feat.device)
        new_b[inv] = b

        return Point(
            feat=F.gelu(self.norm(pooled)),
            coord=coord,
            grid_coord=new_gc,
            batch=new_b,
            pooling_parent=point,
            pooling_inverse=inv,
        )


class GridUnpooling(nn.Module):
    """Decoder upsampling: parent.feat = proj_skip(parent.feat) + proj(child.feat)[inverse]."""

    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU())
        self.proj_skip = nn.Sequential(nn.Linear(skip_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU())

    def forward(self, point: Point) -> Point:
        parent, inverse = point.pooling_parent, point.pooling_inverse
        parent.feat = self.proj_skip(parent.feat) + self.proj(point.feat)[inverse]
        return parent


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class PointTransformerV3Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        order: Sequence[str] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: Sequence[int] = (2, 2, 2, 2),
        enc_depths: Sequence[int] = (2, 2, 2, 6, 2),
        enc_channels: Sequence[int] = (32, 64, 128, 256, 512),
        enc_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        enc_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: Sequence[int] = (2, 2, 2, 2),
        dec_channels: Sequence[int] = (64, 64, 128, 256),
        dec_num_head: Sequence[int] = (4, 4, 8, 16),
        dec_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        shuffle_orders: bool = True,
        serial_depth: int = 16,
        stem_kernel: int = 5,
        enc_mode: bool = False,
    ):
        super().__init__()
        n = len(enc_depths)
        assert len(stride) == n - 1 and len(enc_channels) == n and len(enc_num_head) == n and len(enc_patch_size) == n
        self.order, self.shuffle_orders, self.serial_depth = list(order), shuffle_orders, serial_depth
        self.num_stages, self.enc_mode, self.stem_kernel = n, enc_mode, stem_kernel
        n_ord = len(self.order)

        # stem: sparse 5x5x5 conv + BN + GELU
        self.embed_conv = SubMConv3d(in_channels, enc_channels[0], stem_kernel, bias=False)
        self.embed_norm = nn.BatchNorm1d(enc_channels[0])

        # encoder
        dpr = torch.linspace(0, drop_path, sum(enc_depths)).tolist()
        self.pools = nn.ModuleList()
        self.enc = nn.ModuleList()
        c = 0
        for s in range(n):
            if s > 0:
                self.pools.append(GridPooling(enc_channels[s - 1], enc_channels[s], stride[s - 1]))
            self.enc.append(
                nn.ModuleList(
                    Block(
                        enc_channels[s],
                        enc_num_head[s],
                        enc_patch_size[s],
                        i % n_ord,
                        mlp_ratio,
                        qk_norm,
                        attn_drop,
                        proj_drop,
                        dpr[c + i],
                    )
                    for i in range(enc_depths[s])
                )
            )
            c += enc_depths[s]
        self.out_channels = enc_channels[-1]

        # decoder
        if not enc_mode:
            assert len(dec_depths) == len(dec_channels) == len(dec_num_head) == len(dec_patch_size) == n - 1
            dec_ch = list(dec_channels) + [enc_channels[-1]]
            ddpr = torch.linspace(0, drop_path, sum(dec_depths)).tolist()
            self.dec_up = nn.ModuleList()  # built from the deepest stage to the shallowest
            self.dec = nn.ModuleList()
            for s in reversed(range(n - 1)):
                self.dec_up.append(GridUnpooling(dec_ch[s + 1], enc_channels[s], dec_ch[s]))
                dp = ddpr[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])][::-1]
                self.dec.append(
                    nn.ModuleList(
                        Block(
                            dec_ch[s],
                            dec_num_head[s],
                            dec_patch_size[s],
                            i % n_ord,
                            mlp_ratio,
                            qk_norm,
                            attn_drop,
                            proj_drop,
                            dp[i],
                        )
                        for i in range(dec_depths[s])
                    )
                )
            self.out_channels = dec_channels[0]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _prepare(self, data: dict) -> Point:
        coord, feat = data["coord"], data["feat"]
        if "batch" in data:
            batch = data["batch"].long()
        else:
            off = data["offset"].long()
            batch = torch.repeat_interleave(
                torch.arange(len(off), device=off.device), torch.diff(off, prepend=off.new_zeros(1))
            )
        batch = batch.to(feat.device)
        if "grid_coord" in data:
            gc = data["grid_coord"].long()
        else:
            B = int(batch.max()) + 1
            mn = torch.full((B, 3), float("inf"), device=coord.device, dtype=coord.dtype)
            mn = mn.scatter_reduce(0, batch[:, None].expand(-1, 3), coord, "amin")
            # small epsilon guards against float rounding for points exactly on voxel boundaries
            gc = torch.floor((coord - mn[batch]) / data["grid_size"] + 1e-4).long()
        return Point(coord=coord, feat=feat, grid_coord=gc, batch=batch)

    def forward(self, data: dict) -> Point:
        point = self._prepare(data)
        with torch.no_grad():
            nbr_stem = build_neighbors(point.grid_coord, point.batch, self.stem_kernel)

        point.feat = F.gelu(self.embed_norm(self.embed_conv(point.feat, nbr_stem)))

        for s in range(self.num_stages):
            if s > 0:
                point = self.pools[s - 1](point)
            serialize(point, self.order, self.serial_depth, self.shuffle_orders and self.training)
            point.nbr = build_neighbors(point.grid_coord, point.batch, 3)
            for blk in self.enc[s]:
                point = blk(point)

        if not self.enc_mode:
            for up, blocks in zip(self.dec_up, self.dec):
                point = up(point)
                for blk in blocks:
                    point = blk(point)
        return point


if __name__ == "__main__":
    from components.utils.device import get_device

    torch.manual_seed(0)
    gs = 0.02

    device = get_device()

    def make_cloud(n):
        g = torch.unique(torch.randint(0, 50, (n, 3)), dim=0).to(device)
        return g.float() * gs, g

    clouds = [make_cloud(6000), make_cloud(3500)]
    coord = torch.cat([c for c, _ in clouds])
    batch = torch.cat([torch.full((len(c),), i) for i, (c, _) in enumerate(clouds)])
    grid_coord = torch.cat([g for _, g in clouds])
    data = dict(coord=coord, feat=torch.cat([coord, torch.rand_like(coord)], 1), batch=batch, grid_coord=grid_coord)

    kw = dict(
        in_channels=6,
        enc_depths=(2, 2, 2, 2),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(128,) * 4,
        stride=(2, 2, 2),
        dec_depths=(1, 1, 1),
        dec_channels=(32, 64, 128),
        dec_num_head=(2, 4, 8),
        dec_patch_size=(128,) * 3,
    )
    enc = PointTransformerV3Encoder(enc_mode=True, **kw).to(device)
    out = enc(data)
    print("encoder:", tuple(out.feat.shape), "points per cloud ->", out.batch.bincount().tolist())

    model = PointTransformerV3Encoder(**kw).to(device)
    out = model(data)
    print("encoder+decoder:", tuple(out.feat.cpu().shape))
    out.feat.mean().backward()
    print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
