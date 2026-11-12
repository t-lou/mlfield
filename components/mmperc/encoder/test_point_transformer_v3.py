import itertools

import torch

from components.mmperc.encoder.point_transformer_v3 import (
    PointTransformerV3Encoder,
    _pad_unpad,
    build_neighbors,
    hilbert_encode,
    z_order_encode,
)

torch.manual_seed(0)


def test_z_order():
    gc = torch.randint(0, 1 << 10, (500, 3))
    ref = torch.zeros(500, dtype=torch.long)
    for i in range(10):
        for d in range(3):
            ref |= ((gc[:, d] >> i) & 1) << (3 * i + (2 - d))
    assert torch.equal(z_order_encode(gc, 10), ref)


def test_hilbert():
    for depth in (1, 2, 3, 4):
        n = 1 << depth
        gc = torch.tensor(list(itertools.product(range(n), repeat=3)))
        code = hilbert_encode(gc, depth)
        assert torch.equal(code.sort().values, torch.arange(n**3)), f"not a bijection, depth {depth}"
        path = gc[code.argsort()]
        step = (path[1:] - path[:-1]).abs().sum(1)
        assert bool((step == 1).all()), f"curve not continuous, depth {depth}"


def pad_unpad_ref(bincount, ps):  # the original loop implementation
    bc = bincount.tolist()
    bc_pad = [-(-c // ps) * ps for c in bc]
    off, off_pad = [0], [0]
    for c, cp in zip(bc, bc_pad):
        off.append(off[-1] + c)
        off_pad.append(off_pad[-1] + cp)
    pad = torch.arange(off_pad[-1])
    unpad = torch.arange(off[-1])
    for i in range(len(bc)):
        unpad[off[i] : off[i + 1]] += off_pad[i] - off[i]
        if bc[i] != bc_pad[i]:
            r = bc[i] % ps
            pad[off_pad[i + 1] - ps + r : off_pad[i + 1]] = pad[off_pad[i + 1] - 2 * ps + r : off_pad[i + 1] - ps]
        pad[off_pad[i] : off_pad[i + 1]] -= off_pad[i] - off[i]
    return pad, unpad


def test_pad_unpad():
    for _ in range(20):
        ps = int(torch.randint(1, 20, ()))
        bc = torch.randint(ps, 100, (int(torch.randint(1, 6, ())),))
        p1, u1 = _pad_unpad(bc, ps)
        p2, u2 = pad_unpad_ref(bc, ps)
        assert torch.equal(p1, p2) and torch.equal(u1, u2)


def test_neighbors():
    g = torch.unique(torch.randint(0, 6, (200, 3)), dim=0)
    b = torch.zeros(len(g), dtype=torch.long)
    g2 = torch.cat([g, g])
    b2 = torch.cat([b, b + 1])
    for k in (3, 5):
        nbr = build_neighbors(g2, b2, k)
        p = k // 2
        r = range(-p, p + 1)
        offs = list(itertools.product(r, r, r))
        lut = {(int(bb), *map(int, c)): i for i, (bb, c) in enumerate(zip(b2, g2))}
        for i in range(0, len(g2), 7):
            for j, o in enumerate(offs):
                q = (int(b2[i]), *(int(g2[i][d]) + o[d] for d in range(3)))
                assert int(nbr[i, j]) == lut.get(q, len(g2))


def make_data(sizes, gs=0.02):
    cs, gcs, bs = [], [], []
    for i, n in enumerate(sizes):
        g = torch.unique(torch.randint(0, 40, (n, 3)), dim=0)
        gcs.append(g)
        cs.append(g.float() * gs)
        bs.append(torch.full((len(g),), i))
    coord, gc, batch = torch.cat(cs), torch.cat(gcs), torch.cat(bs)
    return dict(coord=coord, feat=torch.cat([coord, torch.rand_like(coord)], 1), batch=batch, grid_coord=gc)


KW = dict(
    in_channels=6,
    enc_depths=(2, 2, 2),
    enc_channels=(16, 32, 64),
    enc_num_head=(2, 4, 4),
    enc_patch_size=(64,) * 3,
    stride=(2, 2),
    dec_depths=(1, 1),
    dec_channels=(16, 32),
    dec_num_head=(2, 4),
    dec_patch_size=(64,) * 2,
)


def test_permutation_equivariance():
    data = make_data([3000, 1500, 700])
    for enc_mode in (False, True):
        m = PointTransformerV3Encoder(enc_mode=enc_mode, **KW).eval()
        with torch.no_grad():
            ref = m(data)
            perm = torch.randperm(len(data["coord"]))
            d2 = {k: v[perm] for k, v in data.items()}
            out = m(d2)
        if enc_mode:  # output order is sorted by voxel key -> compare as sets of (batch, coord, feat)
            key = lambda p: torch.cat([p.batch[:, None].float(), p.coord, p.feat], 1)
            a, b = key(ref), key(out)
            a = a[torch.lexsort(a.T.flip(0).unbind(0)[::-1])] if False else a
            ia = torch.argsort(
                ref.batch * 1e6 + ref.coord[:, 0] * 1e3 + ref.coord[:, 1] * 50 + ref.coord[:, 2] * 2.5, stable=True
            )
            ib = torch.argsort(
                out.batch * 1e6 + out.coord[:, 0] * 1e3 + out.coord[:, 1] * 50 + out.coord[:, 2] * 2.5, stable=True
            )
            err = (ref.feat[ia] - out.feat[ib]).abs().max().item()
        else:
            err = (ref.feat[perm] - out.feat).abs().max().item()
        print(f"permutation err (enc_mode={enc_mode}): {err:.2e}")
        assert err < 1e-4


def test_backward_and_train():
    data = make_data([2000, 900])
    m = PointTransformerV3Encoder(**KW).train()
    out = m(data)
    out.feat.square().mean().backward()
    bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, bad
    assert out.feat.shape == (len(data["coord"]), 16)


def test_small_cloud_and_grid_size():
    data = make_data([800, 30])  # second cloud smaller than the patch size
    m = PointTransformerV3Encoder(**KW).eval()
    with torch.no_grad():
        m(data)
        d = dict(
            coord=data["coord"],
            feat=data["feat"],
            offset=torch.tensor([int((data["batch"] == 0).sum()), len(data["coord"])]),
            grid_size=0.02,
        )
        out = m(d)
    assert out.feat.shape[0] == len(data["coord"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
