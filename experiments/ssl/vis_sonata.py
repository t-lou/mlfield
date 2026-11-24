"""Diagnose whether Sonata self-supervised training produced a useful model.

No downstream labels needed. Renders four complementary views of the same cloud
and prints a PASS/WARN/FAIL health report built from label-free SSL signals:

  pca        DINO-style PCA of encoded features -> RGB. Good features give smooth,
             object/part-coherent colors; garbage gives noise or one flat color.
  prototype  Dominant prototype per point (categorical colors). Should form
             compact, spatially coherent regions, using many prototypes.
  agreement  Does the masked view pick the same prototype as the unmasked view?
             (green = yes). This is the SSL objective itself; also reports
             chance-level agreement so a collapsed model can't fake it.
  confidence Mean max-softmax over both views (robust color range).

Report checks: prototype collapse, mask/unmask agreement vs. chance, prototype
spatial coherence vs. chance, feature locality (neighbor vs. random cosine),
feature effective rank (dimensional collapse), near-uniform assignments.

Caveat: these are collapse / consistency detectors. PASS means "no sign of
failure", not "good downstream". Thresholds are heuristics; the most reliable use
is comparing checkpoints (--json) on the same cloud. Confirm with a linear probe
or kNN on labels when you can.

Use sonata_full.onnx (NOT sonata_teacher_backbone.onnx): only the full export has
mask_logits + unmask_logits.

Examples:
    python -m experiments.ssl.vis_sonata sonata_full.onnx cloud.pcd
    python -m experiments.ssl.vis_sonata sonata_full.onnx cloud.pcd --mode pca
    python -m experiments.ssl.vis_sonata sonata_full.onnx cloud.pcd \\
        --screenshot overview.png --diagnostics plots.png --json step_50k.json
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
import yaml
from components.dataset.pcd_only_dataset import PcdOnlyFolderDataset
from components.definitions.bev_params import BevParams
from components.mmperc.encoder.point_transformer_v3 import PointGridSpec, points_to_voxels, z_order_encode
from components.utils.config import load_yaml
from experiments.ssl.sonata import DEFAULT_IGNORED_FIELDS, SonataConfig, SonataExperimentConfig


def _read_config(config_path: Path) -> SonataConfig:
    with config_path.open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError(f"{config_path}: expected a Sonata configuration mapping")
    if "model" in values:
        return load_yaml(config_path, SonataExperimentConfig).model
    return load_yaml(config_path, SonataConfig)


def _read_ascii_ply_header(file: TextIO, path: Path) -> tuple[int, tuple[str, ...]]:
    if file.readline().strip() != "ply":
        raise ValueError(f"{path}: invalid PLY header")
    header: list[list[str]] = []
    for line in file:
        parts = line.split()
        if parts and parts[0] == "end_header":
            break
        if parts and parts[0] not in {"comment", "obj_info"}:
            header.append(parts)
    else:
        raise ValueError(f"{path}: incomplete PLY header")

    format_lines = [parts for parts in header if parts[0] == "format"]
    vertex_indexes = [
        i for i, parts in enumerate(header) if parts[0] == "element" and len(parts) > 1 and parts[1] == "vertex"
    ]
    if (
        len(format_lines) != 1
        or format_lines[0][1:] != ["ascii", "1.0"]
        or len(vertex_indexes) != 1
        or any(parts[0] == "element" for parts in header[: vertex_indexes[0]])
    ):
        raise ValueError(f"{path}: ASCII fallback requires a vertex element and format ascii 1.0")
    vertex_idx = vertex_indexes[0]
    try:
        vertex_count = int(header[vertex_idx][2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: invalid PLY vertex count") from exc
    if vertex_count < 0:
        raise ValueError(f"{path}: PLY vertex count cannot be negative")
    next_element = next(
        (i for i in range(vertex_idx + 1, len(header)) if header[i][0] == "element"),
        len(header),
    )
    properties = [parts for parts in header[vertex_idx + 1 : next_element] if parts[0] == "property"]
    if any(len(parts) != 3 for parts in properties):
        raise ValueError(f"{path}: ASCII fallback does not support list-valued vertex properties")
    fields = tuple(parts[2] for parts in properties)
    if not fields:
        raise ValueError(f"{path}: PLY vertex element has no scalar properties")
    return vertex_count, fields


def _load_ascii_ply(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    with path.open(encoding="ascii") as file:
        count, fields = _read_ascii_ply_header(file, path)
        if count == 0:
            return np.empty((0, len(fields)), dtype=np.float32), fields
        try:
            data = np.loadtxt(file, dtype=np.float32, max_rows=count, usecols=range(len(fields)), ndmin=2)
        except ValueError as exc:
            raise ValueError(f"{path}: could not decode ASCII PLY vertex properties: {exc}") from exc
        if len(data) != count:
            raise ValueError(f"{path}: truncated PLY vertex data")
        return data, fields


def _load_cloud(
    path: Path,
    coord_fields: tuple[str, str, str],
    selected_features: tuple[str, ...] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not path.is_file() or path.suffix.lower() not in {".pcd", ".ply"}:
        raise ValueError(f"{path}: expected an existing .pcd or .ply file")
    dataset = PcdOnlyFolderDataset(str(path.parent), include_fields=True)
    matches = [i for i, cloud_path in enumerate(dataset.pcd_paths) if cloud_path.resolve() == path.resolve()]
    if not matches:
        raise FileNotFoundError(f"{path}: point cloud was not found by the PCD/PLY loader")
    try:
        values, fields = dataset[matches[0]]
    except ImportError:
        if path.suffix.lower() != ".ply":
            raise
        values, fields = _load_ascii_ply(path)

    return _prepare_cloud_array(path, np.asarray(values), tuple(fields), coord_fields, selected_features)


def _prepare_cloud_array(
    path: Path,
    values: np.ndarray,
    fields: tuple[str, ...],
    coord_fields: tuple[str, str, str],
    selected_features: tuple[str, ...] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    field_to_idx = {field.lower(): i for i, field in enumerate(fields)}
    coord_names = tuple(name.lower() for name in coord_fields)
    if values.ndim != 2 or values.shape[1] != len(fields):
        raise ValueError(f"{path}: point data shape {values.shape} does not match its {len(fields)} fields")
    missing = [name for name in coord_names if name not in field_to_idx]
    if missing:
        raise ValueError(f"{path}: missing coordinate fields {missing}; available fields: {fields}")
    tensor = torch.as_tensor(values, dtype=torch.float32)
    xyz = tensor[:, [field_to_idx[name] for name in coord_names]]
    valid = torch.isfinite(xyz).all(dim=1)
    xyz = xyz[valid]
    if not len(xyz):
        raise ValueError(f"{path}: contains no finite XYZ coordinates")

    ignored_fields = {name.lower() for name in DEFAULT_IGNORED_FIELDS}
    if selected_features is None:
        feature_names = tuple(
            name for name in fields if name.lower() not in coord_names and name.lower() not in ignored_fields
        )
    else:
        feature_names = selected_features
        missing = [name for name in feature_names if name.lower() not in field_to_idx]
        if missing:
            raise ValueError(f"{path}: requested feature fields {missing} are absent; available fields: {fields}")
        if any(name.lower() in coord_names for name in feature_names):
            raise ValueError("Coordinate fields cannot also be listed as feature fields")
    if feature_names:
        feat = torch.stack([tensor[:, field_to_idx[name.lower()]] for name in feature_names], dim=1)[valid]
        feat = torch.nan_to_num(feat)
        feat = (feat - feat.mean(dim=0, keepdim=True)) / feat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        feat = feat.clamp(-10, 10)
    else:
        feat = xyz.clone()
    return xyz.numpy(), feat.numpy(), feature_names


def _select_points(
    xyz: np.ndarray, feat: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("point count must be positive")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or feat.ndim != 2 or feat.shape[0] != len(xyz):
        raise ValueError("coordinates and features must be aligned with shapes (N, 3) and (N, C)")
    if len(xyz) == 0:
        raise ValueError("cannot select points from an empty cloud")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(xyz), size=count, replace=len(xyz) < count))
    return xyz[indices], feat[indices], indices


def _export_lattice(point_count: int) -> np.ndarray:
    if point_count < 1:
        raise ValueError("point count must be positive")
    side = math.ceil(point_count ** (1 / 3))
    point_ids = np.arange(point_count)
    return np.stack((point_ids // (side * side), (point_ids // side) % side, point_ids % side), axis=1).astype(np.int64)


def _map_to_export_lattice(grid_coord: np.ndarray) -> np.ndarray:
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3 or len(grid_coord) == 0:
        raise ValueError(f"grid coordinates must have non-empty shape (N, 3), got {grid_coord.shape}")
    source_grid = grid_coord - grid_coord.min(axis=0, keepdims=True)
    source_max = source_grid.max(axis=0, keepdims=True)
    source_grid = np.divide(
        source_grid * ((1 << 16) - 1),
        source_max,
        out=np.zeros_like(source_grid, dtype=np.float64),
        where=source_max > 0,
    ).astype(np.int64)
    target_grid = _export_lattice(len(grid_coord))
    source_order = z_order_encode(torch.from_numpy(source_grid), 16).numpy().argsort(kind="stable")
    target_order = z_order_encode(torch.from_numpy(target_grid), 16).numpy().argsort(kind="stable")
    mapped_grid = np.empty_like(target_grid)
    mapped_grid[source_order] = target_grid[target_order]
    return mapped_grid


def _voxelize_cloud(
    xyz: np.ndarray, feat: np.ndarray, grid: PointGridSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = torch.cat((torch.from_numpy(xyz), torch.from_numpy(feat)), dim=1)
    data, is_real = points_to_voxels(
        points,
        torch.zeros(len(points), dtype=torch.long),
        grid,
        batch_size=1,
        feature_dim=points.shape[1],
    )
    if not is_real.any():
        raise ValueError("point cloud contains no points inside the configured input grid")
    return data["coord"][is_real].numpy(), data["feat"][is_real, 3:].numpy(), data["grid_coord"][is_real].numpy()


def _effective_config(config: SonataConfig, grid_size: float | None) -> tuple[SonataConfig, PointGridSpec]:
    grid = config.input_grid or PointGridSpec.from_bev_params(BevParams())
    if grid_size is not None:
        grid = replace(grid, voxel_size_xyz=grid_size)
    if config.input_grid != grid:
        config = replace(config, input_grid=grid)
    return config, grid


def output_point_map(
    grid_coord: np.ndarray, output_count: int, stride_factors: tuple[int, ...] | None = None
) -> np.ndarray:
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3 or len(grid_coord) == 0:
        raise ValueError(f"grid coordinates must have non-empty shape (N, 3), got {grid_coord.shape}")
    if output_count < 1:
        raise ValueError("output feature count must be positive")
    if output_count == len(grid_coord):
        return np.arange(len(grid_coord))
    if stride_factors is None:
        factors = []
        factor = 1
        while factor <= int(grid_coord.max()) + 1:
            factors.append(factor)
            factor *= 2
    else:
        factors = list(dict.fromkeys(stride_factors))
    candidates = []
    for factor in factors:
        voxels, inverse = np.unique(grid_coord // factor, axis=0, return_inverse=True)
        if len(voxels) == output_count:
            candidates.append(inverse)
    if not candidates:
        raise ValueError(
            f"Model returned {output_count} features for {len(grid_coord)} points; no pooling stage matches"
        )
    if any(not np.array_equal(candidates[0], candidate) for candidate in candidates[1:]):
        raise ValueError("Model output resolution is ambiguous; use grid coordinates matching the exported model")
    return candidates[0]


def _resolve_onnx_point_count(point_dim: int | str | None, requested_points: int | None) -> int:
    if isinstance(point_dim, int):
        if requested_points is not None and requested_points != point_dim:
            raise ValueError(f"ONNX model requires {point_dim} points, but --num-points={requested_points}")
        count = point_dim
    else:
        if requested_points is None:
            raise ValueError("This ONNX model has a dynamic point count; provide --num-points")
        count = requested_points
    if count < 1:
        raise ValueError("ONNX point count must be positive")
    return count


MODES = ("all", "pca", "prototype", "agreement", "confidence")


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _run_onnx_with_logits(
    path: Path,
    xyz: np.ndarray,
    feat: np.ndarray,
    source_grid_coord: np.ndarray,
    requested_points: int | None = None,
    seed: int = 0,
    config: SonataConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Run ONNX model; return xyz, mask_logits, unmask_logits, features (or None).

    Per-voxel outputs are mapped back to points.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("requires onnxruntime") from exc

    session = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {"features", "grid_coord", "batch"}:
        raise ValueError(f"expected inputs features, grid_coord, batch; found {tuple(inputs)}")

    point_count = _resolve_onnx_point_count(inputs["features"].shape[0], requested_points)
    xyz_input, feat_input, selected = _select_points(xyz, feat, point_count, seed)
    grid_coord = _map_to_export_lattice(source_grid_coord[selected])

    feed = {
        "features": np.ascontiguousarray(feat_input, dtype=np.float32),
        "grid_coord": np.ascontiguousarray(grid_coord),
        "batch": np.zeros(point_count, dtype=np.int64),
    }
    output_names = [item.name for item in session.get_outputs()]
    out = dict(zip(output_names, session.run(output_names, feed)))

    missing = [n for n in ("mask_logits", "unmask_logits") if n not in out]
    if missing:
        raise ValueError(f"model has no {missing} output(s) (found {output_names}); use sonata_full.onnx")

    mask_logits = np.asarray(out["mask_logits"])
    unmask_logits = np.asarray(out["unmask_logits"])
    features = np.asarray(out["encoded_features"]) if "encoded_features" in out else None

    n_points = len(xyz_input)
    output_rows = mask_logits.shape[0]
    if output_rows != n_points:
        stride_factors = None
        if config is not None:
            backbone = config.backbone
            strides = backbone.stride or (2,) * (len(backbone.enc_depths) - 1)
            stage = len(backbone.enc_depths) - 1 - config.up_cast_level
            if 0 <= stage < len(backbone.enc_depths):
                stride_factors = (1, *np.cumprod(strides[:stage]).astype(int).tolist())
        mapping = output_point_map(grid_coord, output_rows, stride_factors)
        mask_logits = mask_logits[mapping]
        unmask_logits = unmask_logits[mapping]
        if features is not None and features.shape[0] == output_rows:
            features = features[mapping]

    if features is not None and features.shape[0] != n_points:
        print(
            f"warning: encoded_features has {features.shape[0]} rows != {n_points} points; skipping PCA/feature metrics"
        )
        features = None
    return xyz_input, mask_logits, unmask_logits, features


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _usage(pid: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Prototype usage distribution and its perplexity (effective #prototypes)."""
    p = np.bincount(pid, minlength=k).astype(np.float64)
    p /= p.sum()
    nz = p[p > 0]
    return p, float(np.exp(-(nz * np.log(nz)).sum()))


def _pca(features: np.ndarray, n_fit: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Top-3 PCA projection of all points; returns (N,3) and full eigenvalue spectrum."""
    f = np.asarray(features, dtype=np.float32)
    sub = f[rng.choice(len(f), size=min(n_fit, len(f)), replace=False)]
    mean = sub.mean(axis=0)
    cov = ((sub - mean).T @ (sub - mean)) / max(1, len(sub) - 1)
    vals, vecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    proj = (f - mean) @ vecs[:, order[:3]].astype(np.float32)
    return proj, vals


def _effective_rank(eigvals: np.ndarray) -> float:
    s = eigvals.sum()
    if s <= 0:
        return 0.0
    p = eigvals / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _knn_indices(xyz: np.ndarray, k: int) -> np.ndarray | None:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    k = min(k, len(xyz) - 1)
    if k < 1:
        return None
    _, idx = cKDTree(xyz).query(xyz, k=k + 1)
    return idx[:, 1:]


def _neighbor_cosines(
    features: np.ndarray, idx: np.ndarray, rng: np.random.Generator, n_query: int = 20000
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine of each query point to its spatial neighbors vs. to random points."""
    f = np.asarray(features, dtype=np.float32)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    q = rng.choice(len(f), size=min(n_query, len(f)), replace=False)
    nbr = np.empty((len(q), idx.shape[1]), dtype=np.float32)
    for s in range(0, len(q), 2000):  # chunk to bound memory (N*k*D)
        qs = q[s : s + 2000]
        nbr[s : s + 2000] = np.einsum("nd,nkd->nk", f[qs], f[idx[qs]])
    rand = np.einsum("nd,nd->n", f[q], f[rng.permutation(len(f))[: len(q)]])
    return nbr.ravel(), rand


# --------------------------------------------------------------------------- #
# Metrics + verdict
# --------------------------------------------------------------------------- #
def compute_metrics(
    xyz: np.ndarray,
    mask_logits: np.ndarray,
    unmask_logits: np.ndarray,
    features: np.ndarray | None,
    knn_k: int = 8,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Label-free SSL health metrics. Returns (scalar metrics, arrays for plots/views)."""
    rng = np.random.default_rng(seed)
    n, k = unmask_logits.shape
    p_mask, p_unmask = _softmax(mask_logits), _softmax(unmask_logits)
    pid_m, pid_u = p_mask.argmax(1), p_unmask.argmax(1)

    use_m, perp_m = _usage(pid_m, k)
    use_u, perp_u = _usage(pid_u, k)
    conf_m, conf_u = p_mask.max(1), p_unmask.max(1)
    ent_u = -(p_unmask * np.log(p_unmask + 1e-10)).sum(1) / np.log(k)

    agree = pid_m == pid_u
    m: dict = {
        "n_points": int(n),
        "n_prototypes": int(k),
        "finite": bool(np.isfinite(mask_logits).all() and np.isfinite(unmask_logits).all()),
        "n_used_mask": int((use_m > 1e-3).sum()),
        "n_used_unmask": int((use_u > 1e-3).sum()),
        "perplexity_mask": perp_m,
        "perplexity_unmask": perp_u,
        "top1_share_mask": float(use_m.max()),
        "top1_share_unmask": float(use_u.max()),
        "agreement": float(agree.mean()),
        "agreement_chance": float((use_m * use_u).sum()),
        "conf_mean": float((conf_m.mean() + conf_u.mean()) / 2),
        "entropy_norm_unmask": float(ent_u.mean()),
    }
    m["agreement_lift"] = m["agreement"] - m["agreement_chance"]

    extras: dict = {
        "pid_mask": pid_m,
        "pid_unmask": pid_u,
        "agree": agree,
        "conf": (conf_m + conf_u) / 2,
        "usage_mask": use_m,
        "usage_unmask": use_u,
        "conf_mask": conf_m,
        "conf_unmask": conf_u,
    }

    idx = _knn_indices(xyz, knn_k)
    if idx is not None:
        same = (pid_u[idx] == pid_u[:, None]).mean()
        m["proto_neighbor_same"] = float(same)
        m["proto_neighbor_chance"] = float((use_u**2).sum())
        m["proto_coherence_lift"] = m["proto_neighbor_same"] - m["proto_neighbor_chance"]

    if features is not None:
        proj, eig = _pca(features, 20000, rng)
        extras["pca_proj"], extras["eigvals"] = proj, eig
        m["feature_dim"] = int(features.shape[1])
        m["effective_rank"] = _effective_rank(eig)
        m["pca_top3_var"] = float(eig[:3].sum() / max(eig.sum(), 1e-12))
        # spread relative to feature magnitude: ~0 means every point maps to the same vector
        m["feature_spread"] = float(np.sqrt(eig.sum()) / (np.linalg.norm(features, axis=1).mean() + 1e-12))
        if idx is not None:
            nbr, rand = _neighbor_cosines(features, idx, rng)
            extras["cos_nbr"], extras["cos_rand"] = nbr, rand
            m["cos_neighbor"], m["cos_random"] = float(nbr.mean()), float(rand.mean())
            m["locality_gap"] = m["cos_neighbor"] - m["cos_random"]
    return m, extras


PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def build_verdict(m: dict) -> tuple[str, list[tuple[str, str, str, str]]]:
    """Return (overall, [(level, check, value, hint)]). Thresholds are heuristics."""
    rows: list[tuple[str, str, str, str]] = []

    def add(level: str, name: str, value: str, hint: str) -> None:
        rows.append((level, name, value, hint if level != PASS else ""))

    if not m["finite"]:
        add(FAIL, "finite logits", "NaN/Inf", "training diverged or export is broken")

    top1 = max(m["top1_share_mask"], m["top1_share_unmask"])
    used = min(m["n_used_mask"], m["n_used_unmask"])
    lvl = FAIL if (top1 > 0.8 or used < 3) else WARN if (top1 > 0.5 or used < 8) else PASS
    add(
        lvl,
        "prototype collapse",
        f"top-1 share {top1:.0%}, {used}/{m['n_prototypes']} used",
        "one/few prototypes win everywhere: classic SSL collapse (check centering/temperature/Sinkhorn)",
    )

    lift = m["agreement_lift"]
    lvl = FAIL if lift < 0.05 else WARN if lift < 0.2 else PASS
    add(
        lvl,
        "mask/unmask agreement",
        f"{m['agreement']:.0%} vs chance {m['agreement_chance']:.0%}",
        "masked view barely predicts the unmasked one: objective not learned",
    )

    if "proto_coherence_lift" in m:
        lift = m["proto_coherence_lift"]
        lvl = FAIL if lift < 0.05 else WARN if lift < 0.2 else PASS
        add(
            lvl,
            "prototype spatial coherence",
            f"{m['proto_neighbor_same']:.0%} neighbors share proto vs chance {m['proto_neighbor_chance']:.0%}",
            "prototype ids carry no spatial structure beyond chance (noise, or a single prototype)",
        )

    ent = m["entropy_norm_unmask"]
    lvl = FAIL if ent > 0.95 else WARN if ent > 0.8 else PASS
    add(
        lvl,
        "assignment sharpness",
        f"normalized entropy {ent:.2f} (1 = uniform)",
        "assignments near uniform: prototypes are not discriminative",
    )

    if "locality_gap" in m:
        gap = m["locality_gap"]
        lvl = FAIL if gap < 0.05 else WARN if gap < 0.15 else PASS
        add(
            lvl,
            "feature locality",
            f"cos neighbor {m['cos_neighbor']:.2f} vs random {m['cos_random']:.2f}",
            "nearby points are no more similar than random ones: no spatial/semantic structure",
        )

    if "feature_spread" in m:
        sp = m["feature_spread"]
        lvl = FAIL if sp < 0.02 else WARN if sp < 0.1 else PASS
        add(
            lvl,
            "feature variance",
            f"std/norm {sp:.3f}",
            "all points map to (almost) the same vector: representation collapse",
        )

    if "effective_rank" in m:
        r = m["effective_rank"]
        lvl = FAIL if r < 3 else WARN if r < 10 else PASS
        add(
            lvl,
            "feature effective rank",
            f"{r:.1f} of {m['feature_dim']} dims (top-3 PCs: {m['pca_top3_var']:.0%} var)",
            "features live in a tiny subspace: dimensional collapse",
        )

    overall = FAIL if any(r[0] == FAIL for r in rows) else WARN if any(r[0] == WARN for r in rows) else PASS
    return overall, rows


def print_report(m: dict, overall: str, rows: list[tuple[str, str, str, str]]) -> None:
    print(f"\n=== SSL health report ({m['n_points']:,} points, {m['n_prototypes']} prototypes) ===")
    for level, name, value, hint in rows:
        print(f"[{level}] {name}: {value}" + (f"\n       -> {hint}" if hint else ""))
    print(f"\nOverall: {overall}  (no sign of collapse != proven useful; confirm with a probe / compare checkpoints)\n")


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
@dataclass
class View:
    title: str
    values: np.ndarray  # (N,3) uint8 RGB or (N,) scalars
    rgb: bool = True
    cmap: str = "viridis"
    clim: tuple[float, float] | None = None
    bar_title: str = ""


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def _pca_rgb(proj: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    return _to_u8((proj - lo) / np.maximum(hi - lo, 1e-8))


def _prototype_rgb(pid: np.ndarray, k: int) -> np.ndarray:
    """Distinct colors by frequency rank (golden-ratio hues); rare prototypes gray."""
    counts = np.bincount(pid, minlength=k)
    palette = np.full((k, 3), 0.45, dtype=np.float32)
    for rank, p in enumerate(np.argsort(-counts)):
        if counts[p] < max(1, 0.002 * len(pid)):
            break
        hue = (rank * 0.61803398875) % 1.0
        palette[p] = colorsys.hsv_to_rgb(hue, (0.95, 0.7, 0.5)[rank % 3], (1.0, 0.85)[rank % 2])
    return _to_u8(palette[pid])


def build_views(mode: str, m: dict, ex: dict, cmap: str, has_features: bool) -> list[View]:
    k = m["n_prototypes"]
    views: dict[str, View] = {}
    if has_features:
        views["pca"] = View(
            f"PCA of features -> RGB (top-3 PCs: {m['pca_top3_var']:.0%} var)\nsmooth, part-coherent colors = good",
            _pca_rgb(ex["pca_proj"]),
        )
    views["prototype"] = View(
        f"Dominant prototype ({m['n_used_unmask']}/{k} used, top-1 {m['top1_share_unmask']:.0%})\n"
        "compact regions = good, noise / one color = bad",
        _prototype_rgb(ex["pid_unmask"], k),
    )
    agree_rgb = np.where(ex["agree"][:, None], np.array([0.2, 0.75, 0.35]), np.array([0.85, 0.2, 0.2]))
    views["agreement"] = View(
        f"Mask/unmask agreement {m['agreement']:.0%} (chance {m['agreement_chance']:.0%})\n"
        "green = same prototype, red = differs",
        _to_u8(agree_rgb),
    )
    c = ex["conf"]
    views["confidence"] = View(
        "Prototype confidence (mean max-softmax, both views)",
        c.astype(np.float32),
        rgb=False,
        cmap=cmap,
        clim=(float(np.percentile(c, 2)), float(np.percentile(c, 98))),
        bar_title="confidence",
    )
    if mode == "all":
        return list(views.values())
    if mode not in views:
        raise ValueError(f"mode '{mode}' needs encoded_features, which this model does not output")
    return [views[mode]]


def _render(
    xyz: np.ndarray, views: list[View], status: str, screenshot: Path | None, background: str, point_size: float
) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("visualization requires pyvista") from exc

    n = len(views)
    shape = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}[n]
    off_screen = screenshot is not None
    size = [min(2400, 900 * shape[1]), 800 * shape[0]]
    plotter = pv.Plotter(shape=shape, off_screen=off_screen, window_size=size)
    plotter.set_background(background)
    r, g, b = pv.Color(background).float_rgb
    fg = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"

    pts = np.asarray(xyz, dtype=np.float32)
    for i, v in enumerate(views):
        plotter.subplot(i // shape[1], i % shape[1])
        cloud = pv.PolyData(pts)
        if v.rgb:
            cloud["rgb"] = v.values
            plotter.add_mesh(cloud, scalars="rgb", rgb=True, point_size=point_size)
        else:
            cloud["values"] = v.values
            plotter.add_mesh(
                cloud,
                scalars="values",
                cmap=v.cmap,
                clim=v.clim,
                point_size=point_size,
                scalar_bar_args={"title": v.bar_title, "color": fg},
            )
        plotter.add_text(v.title, position="upper_edge", font_size=9, color=fg)
        if i == 0:
            plotter.add_text(status, position="lower_left", font_size=9, color=fg)
    if n > 1:
        plotter.link_views()

    if screenshot:
        plotter.screenshot(str(screenshot))
        print(f"Saved view to {screenshot}")
    else:
        plotter.show()


def _save_diagnostics(path: Path, m: dict, ex: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    for key, label in (("usage_mask", "mask"), ("usage_unmask", "unmask")):
        u = np.sort(ex[key])[::-1]
        ax[0, 0].plot(np.arange(1, len(u) + 1), np.maximum(u, 1e-7), label=label)
    ax[0, 0].set(
        yscale="log",
        xlabel="prototype rank",
        ylabel="usage share",
        title="Prototype usage (flat = healthy, cliff after rank 1-3 = collapse)",
    )
    ax[0, 0].legend()

    for key, label in (("conf_mask", "mask"), ("conf_unmask", "unmask")):
        ax[0, 1].hist(ex[key], bins=50, range=(0, 1), alpha=0.6, label=label)
    ax[0, 1].set(xlabel="max softmax", title="Per-point confidence")
    ax[0, 1].legend()

    if "cos_nbr" in ex:
        ax[1, 0].hist(ex["cos_rand"], bins=60, range=(-1, 1), alpha=0.6, density=True, label="random pairs")
        ax[1, 0].hist(ex["cos_nbr"], bins=60, range=(-1, 1), alpha=0.6, density=True, label="spatial neighbors")
        ax[1, 0].set(xlabel="feature cosine", title="Locality (separated = structure)")
        ax[1, 0].legend()
    else:
        ax[1, 0].axis("off")

    if "eigvals" in ex:
        e = ex["eigvals"][:50]
        ax[1, 1].semilogy(np.arange(1, len(e) + 1), np.maximum(e / ex["eigvals"].sum(), 1e-9), marker=".")
        ax[1, 1].set(
            xlabel="PC", ylabel="variance share", title=f"Feature spectrum (eff. rank {m['effective_rank']:.1f})"
        )
    else:
        ax[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved diagnostics to {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="Sonata ONNX model (sonata_full.onnx)")
    parser.add_argument("cloud", type=Path, help="Point cloud file (.pcd or .ply)")
    parser.add_argument("--mode", default="all", choices=MODES, help="View(s) to render (default: all, 2x2 linked)")
    parser.add_argument("--num-points", type=int, default=None, help="Downsample to this many points")
    parser.add_argument("--coord-fields", default="x,y,z", help="Coordinate field names (default: x,y,z)")
    parser.add_argument("--cfg", type=Path, help="Sonata config YAML")
    parser.add_argument("--cmap", default="viridis", help="Colormap for the confidence view")
    parser.add_argument("--background", default="black", help="Background color")
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--knn", type=int, default=8, help="Neighbors for spatial-coherence metrics")
    parser.add_argument("--screenshot", type=Path, help="Save PNG instead of showing")
    parser.add_argument(
        "--diagnostics", type=Path, help="Save a matplotlib diagnostics PNG (usage, confidence, locality, spectrum)"
    )
    parser.add_argument("--json", type=Path, help="Dump metrics + verdict to JSON (compare checkpoints)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for point selection / sampling")
    args = parser.parse_args()

    try:
        coord_fields = tuple(f.strip() for f in args.coord_fields.split(",") if f.strip())
        xyz, feat, _ = _load_cloud(args.cloud, coord_fields, None)
        config = _read_config(args.cfg) if args.cfg is not None else SonataConfig()
        config, input_grid = _effective_config(config, None)
        xyz, feat, grid_coord = _voxelize_cloud(xyz, feat, input_grid)
        print(f"Loaded {len(xyz):,} points from {args.cloud}")

        xyz_in, mask_logits, unmask_logits, features = _run_onnx_with_logits(
            args.model, xyz, feat, grid_coord, args.num_points, args.seed, config
        )
        print(
            f"Model outputs: mask_logits {mask_logits.shape}, unmask_logits {unmask_logits.shape}, "
            f"features {None if features is None else features.shape}"
        )

        metrics, extras = compute_metrics(xyz_in, mask_logits, unmask_logits, features, args.knn, args.seed)
        overall, rows = build_verdict(metrics)
        print_report(metrics, overall, rows)

        if args.json:
            args.json.write_text(
                json.dumps(
                    {
                        "metrics": metrics,
                        "overall": overall,
                        "checks": [dict(level=a, check=b, value=c) for a, b, c, _ in rows],
                    },
                    indent=2,
                )
            )
            print(f"Saved metrics to {args.json}")
        if args.diagnostics:
            _save_diagnostics(args.diagnostics, metrics, extras)

        views = build_views(args.mode, metrics, extras, args.cmap, features is not None)
        n_bad = sum(r[0] != PASS for r in rows)
        _render(
            xyz_in,
            views,
            f"SSL health: {overall} ({n_bad} issue{'s' if n_bad != 1 else ''}, see console)",
            args.screenshot,
            args.background,
            args.point_size,
        )
        return 0

    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
