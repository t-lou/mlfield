import itertools
from dataclasses import replace

import torch

from components.mmperc.encoder.point_transformer_v3 import (
    GridPooling,
    Point,
    PointGridSpec,
    PointTransformerV3Config,
    PointTransformerV3Encoder,
    SubMConv3d,
    _pad_unpad,
    adapt_ptv3_config_for_input,
    adapt_ptv3_config_for_output,
    adapt_ptv3_stride_for_output,
    build_neighbors,
    hilbert_encode,
    z_order_encode,
)
from components.mmperc.model.transformer_model import point_grid_dimensions

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
        assert torch.equal(nbr, build_neighbors(g2, b2, k, grid_bound=8))
        p = k // 2
        r = range(-p, p + 1)
        offs = list(itertools.product(r, r, r))
        lut = {(int(bb), *map(int, c)): i for i, (bb, c) in enumerate(zip(b2, g2))}
        for i in range(0, len(g2), 7):
            for j, o in enumerate(offs):
                q = (int(b2[i]), *(int(g2[i][d]) + o[d] for d in range(3)))
                assert int(nbr[i, j]) == lut.get(q, len(g2))


def test_submanifold_convolution_matches_reference():
    grid_coord = torch.unique(torch.randint(0, 5, (50, 3)), dim=0)
    batch = torch.zeros(len(grid_coord), dtype=torch.long)
    neighbors = build_neighbors(grid_coord, batch, 3)
    feat = torch.randn(len(grid_coord), 4, requires_grad=True)
    conv = SubMConv3d(4, 8, 3)

    actual = conv(feat, neighbors)
    padded = torch.cat([feat, feat.new_zeros(1, feat.shape[1])])
    expected = sum(padded[neighbors[:, k]] @ conv.weight[k] for k in range(conv.k3)) + conv.bias
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_grid_pooling_decodes_unique_keys():
    grid_coord = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 1, 0], [0, 0, 0]])
    batch = torch.tensor([0, 0, 0, 1])
    point = Point(
        feat=torch.randn(4, 3),
        coord=grid_coord.float(),
        grid_coord=grid_coord,
        batch=batch,
    )
    pooled = GridPooling(3, 4, stride=2).eval()(point)

    assert torch.equal(pooled.grid_coord, torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]))
    assert torch.equal(pooled.batch, torch.tensor([0, 0, 1]))
    assert pooled.feat.shape == (3, 4)


def make_data(sizes, gs=0.02):
    cs, gcs, bs = [], [], []
    for i, n in enumerate(sizes):
        g = torch.unique(torch.randint(0, 40, (n, 3)), dim=0)
        gcs.append(g)
        cs.append(g.float() * gs)
        bs.append(torch.full((len(g),), i))
    coord, gc, batch = torch.cat(cs), torch.cat(gcs), torch.cat(bs)
    return dict(coord=coord, feat=torch.cat([coord, torch.rand_like(coord)], 1), batch=batch, grid_coord=gc)


CONFIG = PointTransformerV3Config(
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
        m = PointTransformerV3Encoder(replace(CONFIG, enc_mode=enc_mode)).eval()
        with torch.no_grad():
            ref = m(data)
            perm = torch.randperm(len(data["coord"]))
            d2 = {k: v[perm] for k, v in data.items()}
            out = m(d2)
        if enc_mode:  # output order is sorted by voxel key -> compare as sets of (batch, coord, feat)
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
    m = PointTransformerV3Encoder(CONFIG).train()
    out = m(data)
    out.feat.square().mean().backward()
    bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, bad
    assert out.feat.shape == (len(data["coord"]), 16)


def test_encoder_stage_outputs():
    data = make_data([2000, 900])
    m = PointTransformerV3Encoder(replace(CONFIG, enc_mode=True)).eval()
    out = m(data)
    assert len(out.encoder_stages) == len(CONFIG.enc_depths)
    assert [stage.feat.shape[1] for stage in out.encoder_stages] == list(CONFIG.enc_channels)
    assert all(stage.feat.shape[0] == stage.coord.shape[0] == stage.batch.shape[0] for stage in out.encoder_stages)


def test_omitted_strides_match_explicit_default_strides():
    data = make_data([1200, 600])
    implicit_config = replace(CONFIG, stride=None)
    explicit_config = replace(CONFIG, stride=(2,) * (len(CONFIG.enc_depths) - 1))

    implicit = PointTransformerV3Encoder(implicit_config).eval()
    explicit = PointTransformerV3Encoder(explicit_config).eval()
    implicit.load_state_dict(explicit.state_dict())

    with torch.no_grad():
        implicit_out = implicit(data)
        explicit_out = explicit(data)

    assert implicit.config.stride == explicit.config.stride
    assert torch.equal(implicit_out.coord, explicit_out.coord)
    assert torch.allclose(implicit_out.feat, explicit_out.feat)


def test_small_cloud_and_grid_size():
    data = make_data([800, 30])  # second cloud smaller than the patch size
    m = PointTransformerV3Encoder(CONFIG).eval()
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


def test_point_grid_adaptation():
    grid = PointGridSpec(
        min_xyz=(0.0, -48.0, -5.0),
        max_xyz=(128.0, 48.0, 3.0),
        voxel_size_xyz=(1.0, 1.0, 8.0),
    )
    assert grid.dimensions == (128, 96, 1)
    assert grid.serial_depth == 7

    adapted = adapt_ptv3_config_for_input(CONFIG, grid)
    assert adapted.serial_depth == 7


def test_point_output_stride_adaptation():
    grid = PointGridSpec(
        min_xyz=(0.0, -48.0, -5.0),
        max_xyz=(128.0, 48.0, 3.0),
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    config = replace(CONFIG, enc_depths=(2, 2, 2, 2), stride=None)

    assert adapt_ptv3_stride_for_output(config, grid, 8.0) == (2, 2, 2)
    adapted = adapt_ptv3_config_for_output(config, grid, 8.0)
    assert adapted.stride == (2, 2, 2)

    try:
        adapt_ptv3_stride_for_output(config, grid, (2.0, 4.0, 2.0))
    except ValueError as err:
        assert "isotropic" in str(err)
    else:
        raise AssertionError("Anisotropic output scaling must be rejected by scalar-stride helper")


def test_lidar_point_grid_dimensions():
    assert point_grid_dimensions(
        (0.0, 128.0),
        (-48.0, 48.0),
        (-5.0, 3.0),
        (1.0, 1.0, 8.0),
    ) == (128, 96, 1)

    try:
        point_grid_dimensions((0.0, 128.0), (-48.0, 48.0), (-5.0, 3.0), (0.0, 1.0, 1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("Zero voxel size must be rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
