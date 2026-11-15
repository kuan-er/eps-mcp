"""Surface-area postprocessing of a supplied deposition array, not a simulator.

The caller must establish native execution and UV orientation independently.
Missing pixels remain unknown. Current uncalibrated units are explicit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


VERSION = "eps-surface-metrics/1.0.1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def quadrature(count, seed):
    i = np.arange(count, dtype=np.uint32) ^ np.uint32(seed)
    i = ((i >> 1) & np.uint32(0x55555555)) | ((i & np.uint32(0x55555555)) << 1)
    i = ((i >> 2) & np.uint32(0x33333333)) | ((i & np.uint32(0x33333333)) << 2)
    i = ((i >> 4) & np.uint32(0x0F0F0F0F)) | ((i & np.uint32(0x0F0F0F0F)) << 4)
    i = ((i >> 8) & np.uint32(0x00FF00FF)) | ((i & np.uint32(0x00FF00FF)) << 8)
    i = (i >> 16) | (i << 16)
    v = (i.astype(float) + .5) / 2**32
    s = np.sqrt((np.arange(count) + .5) / count)
    return np.column_stack((1 - s, s * (1 - v), s * v))


def evaluate_uv(values, uv, area_m2, face_ids, target_face_ids, *, row_origin,
                unit, known_mask=None, zero_epsilon=0., minimum=None, maximum=None,
                clip_level=None, samples_per_face=128, seed=9173, centroids_mm=None,
                provenance=None):
    """Return (summary, per_face_arrays), weighting samples by real face area.

    row_origin is 'uv_v0' or 'uv_v1': the array's row zero represents V=0 or V=1.
    Nearest texel uses floor(U*width), floor(V*height), clipped at U/V=1.
    Metrics integrate a finite deterministic area-uniform triangle quadrature;
    they do not prove continuous coverage or calibrated physical film thickness.
    """
    raw = np.asarray(values)
    if raw.ndim != 2 or min(raw.shape) < 2 or raw.dtype.kind not in "uif":
        raise ValueError("values must be a numeric real HxW array with both dimensions >=2")
    if unit not in ("native_raw", "relative"):
        raise ValueError("uncalibrated evaluator accepts only native_raw or relative units")
    if row_origin not in ("uv_v0", "uv_v1"):
        raise ValueError("explicit row_origin=uv_v0 or uv_v1 is required")
    if not isinstance(samples_per_face, int) or not 16 <= samples_per_face <= 4096:
        raise ValueError("samples_per_face must be an integer between 16 and 4096")
    if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a uint32 integer")
    for name, value in (("zero_epsilon", zero_epsilon), ("minimum", minimum),
                        ("maximum", maximum), ("clip_level", clip_level)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    if clip_level is not None and clip_level <= zero_epsilon:
        raise ValueError("clip_level must exceed zero_epsilon")
    uv, area, ids = np.asarray(uv, dtype=float), np.asarray(area_m2, dtype=float), np.asarray(face_ids)
    if area.ndim != 1:
        raise ValueError("area_m2 must be a vector")
    n = len(area)
    if uv.shape != (n, 3, 2) or ids.shape != (n,) or not n or ids.dtype.kind not in "ui":
        raise ValueError("UV, area and integer face IDs must have matching face counts")
    if len(np.unique(ids)) != n or np.any(ids < 0):
        raise ValueError("face IDs must be unique and nonnegative")
    if not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1):
        raise ValueError("UV coordinates must be finite in [0,1]")
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("true surface area_m2 must be finite and positive")
    cross = ((uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
             - (uv[:, 1, 1] - uv[:, 0, 1]) * (uv[:, 2, 0] - uv[:, 0, 0]))
    if np.any(np.abs(cross) <= 1e-15):
        raise ValueError("degenerate UV triangles cannot map physical area")
    if n * samples_per_face > 2_000_000:
        raise ValueError("quadrature budget exceeds two million samples")
    target_ids = np.asarray(target_face_ids)
    if target_ids.ndim != 1 or not len(target_ids) or target_ids.dtype.kind not in "ui":
        raise ValueError("target_face_ids must be a nonempty integer vector")
    if len(np.unique(target_ids)) != len(target_ids) or not np.isin(target_ids, ids).all():
        raise ValueError("target IDs must be unique and present in the atlas")
    target = np.isin(ids, target_ids)
    if known_mask is None:
        pixel_known = np.isfinite(raw)
    else:
        mask = np.asarray(known_mask)
        if mask.shape != raw.shape or mask.dtype != np.bool_:
            raise ValueError("known_mask must be a boolean array matching values")
        pixel_known = mask & np.isfinite(raw)
    if np.any(raw[pixel_known] < -zero_epsilon):
        raise ValueError("negative dose detected; verify before/after state instead of clipping a reset")
    bary = quadrature(samples_per_face, seed)
    h, w = raw.shape
    sample_uv = np.einsum("si,fij->fsj", bary, uv)
    x = np.clip(np.floor(sample_uv[:, :, 0] * w).astype(int), 0, w - 1)
    y = np.clip(np.floor(sample_uv[:, :, 1] * h).astype(int), 0, h - 1)
    if row_origin == "uv_v1":
        y = h - 1 - y
    known = pixel_known[y, x]
    dose = raw[y, x].astype(float)
    dose[~known] = 0.  # used only in sums with an explicit zero weight, never labelled zero dose
    dose[known] = np.maximum(dose[known], 0.)
    weights = np.broadcast_to(area[:, None] / samples_per_face, dose.shape)
    known_weight = weights * known
    positive = (dose > zero_epsilon) & known
    known_area = known_weight.sum(axis=1)
    covered_area = (weights * positive).sum(axis=1)
    integral = (known_weight * dose).sum(axis=1)
    face_mean = np.divide(integral, known_area, out=np.full(n, np.nan), where=known_area > 0)
    face = {"face_ids": ids, "is_target": target, "area_m2": area,
            "known_area_m2": known_area, "covered_area_estimate_m2": covered_area,
            "mean_on_known": face_mean,
            "under_area_estimate_m2": np.full(n, np.nan) if minimum is None else
                (weights * ((dose < minimum) & known)).sum(axis=1),
            "over_area_estimate_m2": np.full(n, np.nan) if maximum is None else
                (weights * ((dose > maximum) & known)).sum(axis=1)}
    if centroids_mm is not None:
        centroids = np.asarray(centroids_mm, dtype=float)
        if centroids.shape != (n, 3) or not np.isfinite(centroids).all():
            raise ValueError("centroids_mm must have one finite XYZ per atlas face")
        face["centroid_mm"] = centroids

    def aggregate(which):
        total = float(area[which].sum())
        ka = float(known_area[which].sum())
        ca = float(covered_area[which].sum())
        unknown = max(0., total - ka)
        mean = float(integral[which].sum() / ka) if ka else None
        var = float((known_weight[which] * (dose[which] - (mean or 0.))**2).sum() / ka) if ka else None
        metrics = {"area_m2": total, "known_area_estimate_m2": ka, "unknown_area_estimate_m2": unknown,
                   "covered_area_estimate_m2": ca,
                   "coverage_lower_bound": float(np.clip(ca / total, 0, 1)) if total else None,
                   "coverage_upper_bound": float(np.clip((ca + unknown) / total, 0, 1)) if total else None,
                   "weighted_mean_on_known": mean, "weighted_std_on_known": math.sqrt(max(0., var)) if var is not None else None,
                   "coefficient_of_variation_on_known": math.sqrt(max(0., var)) / mean if mean is not None and mean > 0 else None,
                   "under_area_estimate_m2": None if minimum is None else float(face["under_area_estimate_m2"][which].sum()),
                   "over_area_estimate_m2": None if maximum is None else float(face["over_area_estimate_m2"][which].sum()),
                   "saturated_area_estimate_m2": None if clip_level is None else float(
                       (weights[which] * ((dose[which] >= clip_level) & known[which])).sum())}
        return metrics

    target_metrics, other_metrics = aggregate(target), aggregate(~target)
    # Bounded localization only. Full per-face arrays are returned separately for artifacts.
    missing = np.maximum(0., area - known_area)
    under = known_area - covered_area if minimum is None else face["under_area_estimate_m2"]
    over = np.zeros(n) if maximum is None else face["over_area_estimate_m2"]
    severity = np.where(target, missing + under + over, covered_area)
    top = np.argsort(-severity)[:20]
    localized = []
    for i in top:
        item = {"face_id": int(ids[i]), "is_target": bool(target[i]), "area_m2": float(area[i]),
                "known_area_estimate_m2": float(known_area[i]),
                "covered_area_estimate_m2": float(covered_area[i]),
                "unknown_area_estimate_m2": max(0., float(area[i] - known_area[i])),
                "under_area_estimate_m2": None if minimum is None else float(under[i]),
                "over_area_estimate_m2": None if maximum is None else float(over[i]),
                "mean_on_known": float(face_mean[i]) if math.isfinite(face_mean[i]) else None}
        if centroids_mm is not None:
            item["centroid_mm"] = face["centroid_mm"][i].tolist()
        localized.append(item)
    policy = {"version": VERSION, "source_sha256": SOURCE_SHA256, "samples_per_face": samples_per_face,
              "seed": seed, "row_origin": row_origin, "texel_sampling": "floor(U*W), floor(V*H), clipped at 1",
              "unit": unit, "zero_epsilon": zero_epsilon, "minimum": minimum,
              "maximum": maximum, "clip_level": clip_level}
    summary = {"evaluation_backend": "deterministic_array_postprocess", "native_execution_verified_by_evaluator": False,
               "calibrated_physical_thickness": False, "provenance_provided_by_caller": provenance,
               "array_shape": list(raw.shape), "array_dtype": raw.dtype.str,
               "array_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
               "known_mask_sha256": hashlib.sha256(np.ascontiguousarray(pixel_known).tobytes()).hexdigest(),
               "mapping_sha256": hashlib.sha256(uv.astype('<f8').tobytes() + area.astype('<f8').tobytes() + ids.astype('<i8').tobytes()).hexdigest(),
               "target_face_ids_sha256": hashlib.sha256(np.sort(target_ids).astype('<i8').tobytes()).hexdigest(),
               "policy": policy, "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
               "target": target_metrics, "non_target": other_metrics,
               "non_target_overspray_area_estimate_m2": other_metrics["covered_area_estimate_m2"],
               "localized_faces": localized,
               "minimum_uv_triangle_pixel_area": float((np.abs(cross) * .5 * h * w).min()),
               "coverage_is_quadrature_estimate": True,
               "uv_overlap_verified_by_evaluator": False,
               "unknown_pixels_are_not_zero": True,
               "thresholds_are_provided_experimental_values": True}
    return summary, face


def evaluate_files(raw_npz, atlas_npz, output_dir, target_face_ids, **settings):
    """Native adapter file entry point; returns summary and exact artifact receipts.

    raw NPZ contains 'values' and optional boolean 'known_mask'. The native adapter
    must export its actual raw array/stride and prove row_origin using a probe.
    """
    raw_npz, atlas_npz, output_dir = Path(raw_npz), Path(atlas_npz), Path(output_dir)
    with np.load(raw_npz, allow_pickle=False) as z:
        values = z["values"].copy()
        known = z["known_mask"].copy() if "known_mask" in z else None
    with np.load(atlas_npz, allow_pickle=False) as z:
        uv, area, ids = z["uv"].copy(), z["area_m2"].copy(), z["face_ids"].copy()
    summary, face = evaluate_uv(values, uv, area, ids, target_face_ids, known_mask=known, **settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary["input_files"] = {name: {"path": str(p.resolve()), "size": p.stat().st_size,
                                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for name, p in (("raw_deposition", raw_npz), ("face_atlas", atlas_npz))}
    np.savez_compressed(output_dir / "face_metrics.npz", **face)
    face_path = output_dir / "face_metrics.npz"
    summary["face_metrics_artifact"] = {"path": str(face_path.resolve()), "size": face_path.stat().st_size,
                                        "sha256": hashlib.sha256(face_path.read_bytes()).hexdigest()}
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                      allow_nan=False) + "\n", encoding="utf-8")
    return summary
