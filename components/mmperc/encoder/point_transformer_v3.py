"""Vanilla Point Transformer V3 (PTv3) encoder in pure PyTorch.

No spconv / flash-attn / torch_scatter required. Follows the PTv3 design:
  1. Point serialization (Z-order, Hilbert and their "trans" variants)
  2. Patch-based attention on serialized points (shuffled orders across blocks)
  3. xCPE: conditional positional encoding via a sparse 3x3x3 conv on the voxel grid
  4. Serialization-based grid pooling between stages

Deviations from the official implementation (to stay dependency free):
  * xCPE uses a *depthwise* submanifold 3x3x3 conv (neighbour lookup via hashing)
    followed by Linear + LayerNorm, instead of a full spconv SubMConv3d.
  * The stem is Linear + BN + GELU instead of a 5x5x5 sparse conv.
  * Attention uses F.scaled_dot_product_attention on equal-size padded patches.

Input dict:
  coord      (N, 3) float   point positions
  feat       (N, C_in) float
  grid_coord (N, 3) long    integer voxel indices (preferred), or
  grid_size  float          voxel size; grid_coord = floor((coord - min) / grid_size) per cloud
  batch      (N,) long      or  offset (B,) long (cumulative point counts)
Points are assumed to be voxelized already (one point per voxel).
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
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
# def z_order_encode(gc: torch.Tensor, depth: int) -> torch.Tensor:
#     gc = gc.long()
#     code = torch.zeros_like(gc[:, 0])
#     for i in range(depth):
#         for d in range(3):
#             code |= ((gc[:, d] >> i) & 1) << (3 * i + (2 - d))
#     return code
def _spread3(x):
    x = x & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def z_order_encode(gc: torch.Tensor, depth: int) -> torch.Tensor:
    gc = gc.long()
    return (_spread3(gc[:, 0]) << 2) | (_spread3(gc[:, 1]) << 1) | _spread3(gc[:, 2])


def hilbert_encode(gc: torch.Tensor, depth: int) -> torch.Tensor:
    """3D Hilbert index via Skilling's transpose algorithm (vectorized)."""
    x = gc.long().clone()
    M = 1 << (depth - 1)
    # inverse undo excess work
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(3):
            mask = (x[:, i] & Q) != 0
            x[:, 0] = torch.where(mask, x[:, 0] ^ P, x[:, 0])
            t = (x[:, 0] ^ x[:, i]) & P
            t = torch.where(mask, torch.zeros_like(t), t)
            x[:, 0] = x[:, 0] ^ t
            x[:, i] = x[:, i] ^ t
        Q >>= 1
    # Gray encode
    for i in range(1, 3):
        x[:, i] = x[:, i] ^ x[:, i - 1]
    t = torch.zeros_like(x[:, 0])
    Q = M
    while Q > 1:
        mask = (x[:, 2] & Q) != 0
        t = torch.where(mask, t ^ (Q - 1), t)
        Q >>= 1
    x = x ^ t[:, None]
    # interleave transposed bits into a single key
    code = torch.zeros_like(x[:, 0])
    for i in range(depth):
        for d in range(3):
            code |= ((x[:, d] >> i) & 1) << (3 * i + (2 - d))
    return code


def serialize(point: Point, order: Sequence[str], depth: int, shuffle_orders: bool):
    gc, batch = point.grid_coord, point.batch.long()
    assert int(gc.min()) >= 0 and int(gc.max()) < (1 << depth), "grid_coord out of range for depth"
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
        codes.append((batch << (3 * depth)) | c)  # batch in high bits keeps clouds contiguous
    codes = torch.stack(codes)
    if shuffle_orders:
        codes = codes[torch.randperm(codes.shape[0], device=codes.device)]
    perm = torch.argsort(codes, dim=1)
    inverse = torch.empty_like(perm).scatter_(1, perm, torch.arange(perm.shape[1], device=perm.device).expand_as(perm))
    point.serial_order, point.serial_inverse = perm, inverse
    return point


# --------------------------------------------------------------------------- #
# Neighbour lookup for the submanifold conv (xCPE)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_neighbors(grid_coord: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """(N, 27) indices of 3x3x3 neighbours among occupied voxels; N marks 'missing'."""
    N = grid_coord.shape[0]
    gc = grid_coord.long() + 1
    b = batch.long()
    S = int(gc.max()) + 3

    def key(bb, c):
        return ((bb * S + c[:, 0]) * S + c[:, 1]) * S + c[:, 2]

    sk, perm = key(b, gc).sort()
    r = torch.arange(-1, 2, device=gc.device)
    offs = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    out = []
    for o in offs:
        q = key(b, gc + o)
        pos = torch.searchsorted(sk, q).clamp_(max=N - 1)
        found = sk[pos] == q
        out.append(torch.where(found, perm[pos], torch.full_like(pos, N)))
    return torch.stack(out, 1)


class DepthwiseSubMConv3d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(27, dim) * (1.0 / 27) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, feat: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        padded = torch.cat([feat, feat.new_zeros(1, feat.shape[1])], 0)
        out = feat.new_zeros(feat.shape)
        for k in range(27):
            out = out + padded[nbr[:, k]] * self.weight[k]
        return out + self.bias


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        # per-sample drop is awkward with flat points; drop per point
        mask = x.new_empty(x.shape[0], 1).bernoulli_(keep) / keep
        return x * mask


class MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


# def _pad_unpad(bincount: torch.Tensor, ps: int, device):
#     """Indices that pad each cloud to a multiple of `ps` (reusing the tail of the previous patch)."""
#     bc = bincount.tolist()
#     bc_pad = [-(-c // ps) * ps for c in bc]
#     off, off_pad = [0], [0]
#     for c, cp in zip(bc, bc_pad):
#         off.append(off[-1] + c)
#         off_pad.append(off_pad[-1] + cp)
#     pad = torch.arange(off_pad[-1], device=device)
#     unpad = torch.arange(off[-1], device=device)
#     for i in range(len(bc)):
#         unpad[off[i] : off[i + 1]] += off_pad[i] - off[i]
#         if bc[i] != bc_pad[i]:
#             r = bc[i] % ps
#             pad[off_pad[i + 1] - ps + r : off_pad[i + 1]] = pad[off_pad[i + 1] - 2 * ps + r : off_pad[i + 1] - ps]
#         pad[off_pad[i] : off_pad[i + 1]] -= off_pad[i] - off[i]
#     return pad, unpad
def _pad_unpad(bc: torch.Tensor, ps: int):
    """bc: (B,) cloud sizes, all > 0 and >= ps."""
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


class SerializedAttention(nn.Module):
    def __init__(self, dim, heads, patch_size, order_index, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.patch_size, self.order_index = dim, heads, patch_size, order_index
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop, self.proj_drop = attn_drop, nn.Dropout(proj_drop)

    def forward(self, point: Point) -> torch.Tensor:
        feat = point.feat
        N, C = feat.shape
        bincount = torch.bincount(point.batch.long())
        ps = int(min(self.patch_size, bincount.min().item()))  # patch can't exceed smallest cloud
        pad, unpad = _pad_unpad(bincount, ps)
        order = point.serial_order[self.order_index][pad]
        inverse = unpad[point.serial_inverse[self.order_index]]

        qkv = self.qkv(feat)[order]  # (N_pad, 3C)
        qkv = qkv.reshape(-1, ps, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (P, H, ps, d)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(-1, C)[inverse]
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, dim, heads, patch_size, order_index, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, drop_path=0.0):
        super().__init__()
        self.cpe_conv = DepthwiseSubMConv3d(dim)
        self.cpe_lin = nn.Linear(dim, dim)
        self.cpe_norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SerializedAttention(dim, heads, patch_size, order_index, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), proj_drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, point: Point) -> Point:
        point.feat = point.feat + self.cpe_norm(self.cpe_lin(self.cpe_conv(point.feat, point.nbr)))
        point.feat = point.feat + self.drop_path(self.attn(Point(point, feat=self.norm1(point.feat))))
        point.feat = point.feat + self.drop_path(self.mlp(self.norm2(point.feat)))
        return point


class GridPooling(nn.Module):
    def __init__(self, in_dim, out_dim, stride=2):
        super().__init__()
        assert stride & (stride - 1) == 0, "stride must be a power of two"
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

        out = Point(feat=F.gelu(self.norm(pooled)), coord=coord, grid_coord=new_gc, batch=new_b)
        return out


# --------------------------------------------------------------------------- #
# Encoder
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
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        shuffle_orders: bool = True,
        serial_depth: int = 16,
    ):
        super().__init__()
        n = len(enc_depths)
        assert len(stride) == n - 1 and len(enc_channels) == n and len(enc_num_head) == n
        self.order, self.shuffle_orders, self.serial_depth = list(order), shuffle_orders, serial_depth

        self.embed = nn.Sequential(nn.Linear(in_channels, enc_channels[0]), nn.BatchNorm1d(enc_channels[0]), nn.GELU())
        dpr = torch.linspace(0, drop_path, sum(enc_depths)).tolist()
        self.pools = nn.ModuleList()
        self.stages = nn.ModuleList()
        c = 0
        for s in range(n):
            self.pools.append(
                GridPooling(enc_channels[s - 1], enc_channels[s], stride[s - 1]) if s > 0 else nn.Identity()
            )
            self.stages.append(
                nn.ModuleList(
                    [
                        Block(
                            enc_channels[s],
                            enc_num_head[s],
                            enc_patch_size[s],
                            i % len(self.order),
                            mlp_ratio,
                            attn_drop,
                            proj_drop,
                            dpr[c + i],
                        )
                        for i in range(enc_depths[s])
                    ]
                )
            )
            c += enc_depths[s]
        self.out_channels = enc_channels[-1]

    def _prepare(self, data: dict) -> Point:
        coord, feat = data["coord"], data["feat"]
        if "batch" in data:
            batch = data["batch"].long()
        else:
            off = data["offset"].long()
            batch = torch.repeat_interleave(
                torch.arange(len(off), device=off.device), torch.diff(off, prepend=off.new_zeros(1))
            )
        if "grid_coord" in data:
            gc = data["grid_coord"].long()
        else:
            gc = torch.zeros(coord.shape[0], 3, dtype=torch.long, device=coord.device)
            for _bi in batch.unique():
                # m = batch == _bi
                # gc[m] = torch.floor((coord[m] - coord[m].min(0).values) / data["grid_size"]).long()
                B = int(batch.max()) + 1
                mn = torch.full((B, 3), float("inf"), device=coord.device, dtype=coord.dtype)
                mn = mn.scatter_reduce(0, batch[:, None].expand(-1, 3), coord, "amin")
                gc = torch.floor((coord - mn[batch]) / data["grid_size"] + 1e-6).long()
        return Point(coord=coord, feat=feat, grid_coord=gc, batch=batch)

    def forward(self, data: dict, return_all_stages: bool = False):
        point = self._prepare(data)
        point.feat = self.embed(point.feat)
        feats: list[Point] = []
        for pool, blocks in zip(self.pools, self.stages):
            if not isinstance(pool, nn.Identity):
                point = pool(point)
            serialize(point, self.order, self.serial_depth, self.shuffle_orders and self.training)
            point.nbr = build_neighbors(point.grid_coord, point.batch)
            for blk in blocks:
                point = blk(point)
            if return_all_stages:
                feats.append(Point(point))
        return feats if return_all_stages else point


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    gs = 0.02

    def make_cloud(n):
        g = torch.unique(torch.randint(0, 50, (n, 3)), dim=0)
        return g.float() * gs, g

    clouds = [make_cloud(6000), make_cloud(3500)]
    coord = torch.cat([c for c, _ in clouds])
    batch = torch.cat([torch.full((len(c),), i) for i, (c, _) in enumerate(clouds)])
    grid_coord = torch.cat([g for _, g in clouds])  # pass exact voxel indices (avoids float rounding in floor)
    data = dict(coord=coord, feat=torch.cat([coord, torch.rand_like(coord)], 1), batch=batch, grid_coord=grid_coord)

    model = PointTransformerV3Encoder(
        in_channels=6,
        enc_depths=(2, 2, 2, 2),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(128,) * 4,
        stride=(2, 2, 2),
    )
    out = model(data)
    print("final:", tuple(out.feat.shape), "points per stage ->", out.batch.bincount().tolist())
    out.feat.mean().backward()
    print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
