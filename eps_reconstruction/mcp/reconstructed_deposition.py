"""CPU reconstruction of the archived EPS hit -> integer-delta pipeline.

The delta operator is ported separately in eps_kernel.py. This renderer has its
own explicit camera basis, depth tie rule and deterministic simulation clock;
hardware rasterization and whole-scene equivalence are not established.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import numpy as np

from eps_kernel import DEFAULTS, compute

VERSION = "eps-reconstructed-deposition/1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
INT32_MAX = np.iinfo(np.int32).max


def configuration(profile):
    supplied = profile.get("reconstruction")
    if not isinstance(supplied, dict):
        raise ValueError("gun profile must explicitly configure a reconstruction backend")
    defaults = {"camera_pixels": 64, "atlas_texels_per_cell": 8,
                "tick_interval_s": .05, "pose_substeps": 1,
                "fan_width_mm": 200., "fan_height_mm": 200.,
                "near_plane_m": .001, "far_plane_m": .6,
                "max_ticks": 10000, "max_runtime_s": 180.}
    if set(supplied) - (set(defaults) | {"kernel_parameters"}):
        raise ValueError("unknown reconstruction configuration field")
    cfg = {**defaults, **supplied}
    limits = {"camera_pixels": (16, 256), "atlas_texels_per_cell": (4, 32),
              "pose_substeps": (1, 16), "max_ticks": (1, 20000)}
    for key, (lo, hi) in limits.items():
        if type(cfg[key]) is not int or not lo <= cfg[key] <= hi:
            raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
    for key in ("tick_interval_s", "fan_width_mm", "fan_height_mm", "near_plane_m",
                "far_plane_m", "max_runtime_s"):
        v = cfg[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not .005 <= cfg["tick_interval_s"] <= 1 or cfg["max_runtime_s"] > 600:
        raise ValueError("simulation clock/runtime outside supported bounds")
    if cfg["near_plane_m"] >= cfg["far_plane_m"] or max(cfg["fan_width_mm"], cfg["fan_height_mm"]) > 2000:
        raise ValueError("invalid camera volume")
    requested = cfg.get("kernel_parameters")
    if not isinstance(requested, dict) or set(requested) - set(DEFAULTS):
        raise ValueError("explicit kernel_parameters dictionary is required")
    parameters = {**DEFAULTS, **requested}
    for k, v in parameters.items():
        if isinstance(DEFAULTS[k], bool):
            if type(v) is not bool:
                raise ValueError(f"{k} must be boolean")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"invalid kernel parameter {k}")
    for k in ("SprayOkDistanceMeters", "EllipseAxisA", "EllipseAxisB"):
        if parameters[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if not 0 <= parameters["FluxScale"] <= 100000 or type(parameters["MaxDeltaPerHit"]) is not int or not 0 < parameters["MaxDeltaPerHit"] <= 100000:
        raise ValueError("invalid flux or delta limit")
    # The independently configured orthographic volume supplies these parameters.
    # Segment-distance stretching is intentionally not silently substituted.
    if parameters["UseSegmentDistance"]:
        raise ValueError("this backend currently requires pose_substeps, not stretched-camera sampling")
    parameters.update(CameraHalfWidthMeters=cfg["fan_width_mm"] / 2000.,
                      CameraHalfHeightMeters=cfg["fan_height_mm"] / 2000.,
                      BaseFanHalfWidthMeters=cfg["fan_width_mm"] / 2000.,
                      BaseFanHalfHeightMeters=cfg["fan_height_mm"] / 2000.)
    cfg["kernel_parameters"] = parameters
    return cfg


def _unit(v):
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError("degenerate camera direction")
    return v / length


def camera_basis(forward, up):
    forward = _unit(np.asarray(forward, dtype=float))
    # Right-handed look-at convention: the camera looks down negative view Z.
    right = _unit(np.cross(forward, up))
    return right, _unit(np.cross(right, forward)), forward


def raster_hits(triangles_m, normals, uv, position_m, forward, up, cfg, atlas_side, *, deadline=None):
    """Z-buffer all faces, including non-target occluders, at pixel centres.

    Returns per-texel aggregate hit inputs in the shader's quantized units.
    Equal-depth shared edges belong to the first original face deterministically.
    """
    right, camera_up, forward = camera_basis(forward, up)
    basis = np.column_stack((right, camera_up, forward))
    projected = (triangles_m - position_m) @ basis
    size = cfg["camera_pixels"]
    half_x, half_y = cfg["fan_width_mm"] / 2000., cfg["fan_height_mm"] / 2000.
    depth = np.full((size, size), np.inf, dtype=np.float64)
    face_map = np.full((size, size), -1, dtype=np.int64)
    bary_map = np.zeros((size, size, 3), dtype=np.float64)
    bounds_min, bounds_max = projected.min(axis=1), projected.max(axis=1)
    keep = ((bounds_max[:, 0] >= -half_x) & (bounds_min[:, 0] <= half_x)
            & (bounds_max[:, 1] >= -half_y) & (bounds_min[:, 1] <= half_y)
            & (bounds_max[:, 2] >= cfg["near_plane_m"])
            & (bounds_min[:, 2] <= cfg["far_plane_m"]))
    for ordinal, face in enumerate(np.flatnonzero(keep)):
        if deadline is not None and ordinal % 64 == 0 and time.monotonic() > deadline:
            raise TimeoutError("reconstruction runtime budget exceeded during visibility computation")
        t = projected[face]
        screen = np.column_stack(((t[:, 0] / half_x + 1) * size / 2,
                                  (1 - t[:, 1] / half_y) * size / 2))
        lo = np.maximum(0, np.ceil(screen.min(0) - .5).astype(int))
        hi = np.minimum(size - 1, np.floor(screen.max(0) - .5).astype(int))
        if np.any(lo > hi):
            continue
        a, b, c = screen
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        x, y = np.meshgrid(np.arange(lo[0], hi[0] + 1) + .5,
                           np.arange(lo[1], hi[1] + 1) + .5)
        u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        bary = np.stack((u, v, 1 - u - v), axis=-1)
        z = bary @ t[:, 2]
        patch = np.s_[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        visible = ((bary.min(-1) >= -1e-10) & (z >= cfg["near_plane_m"])
                   & (z <= cfg["far_plane_m"]) & (z < depth[patch] - 1e-10))
        depth[patch][visible] = z[visible]
        face_map[patch][visible] = face
        bary_map[patch][visible] = bary[visible]
    yy, xx = np.nonzero(face_map >= 0)
    if not len(xx):
        return None
    faces, bary = face_map[yy, xx], bary_map[yy, xx]
    positions = np.einsum("ni,nij->nj", bary, triangles_m[faces])
    directions = np.asarray(position_m) - positions
    distance = np.linalg.norm(directions, axis=1)
    to_camera = directions / distance[:, None]
    n = normals[faces].copy()
    raw_z = n[:, 2].copy()
    n[np.einsum("ij,ij->i", n, to_camera) < 0] *= -1
    incidence = np.clip(np.einsum("ij,ij->i", n, to_camera), -1, 1)
    camera_cosine = np.clip(n @ -forward, -1, 1)
    texture = np.mod(np.einsum("ni,nij->nj", bary, uv[faces]), 1.)
    pixel = np.minimum(atlas_side - 1, np.floor(texture * atlas_side).astype(np.int64))
    texel = pixel[:, 1] * atlas_side + pixel[:, 0]
    unique, inverse, count = np.unique(texel, return_inverse=True, return_counts=True)
    ndcx = ((xx + .5) / size) * 2 - 1
    ndcy = 1 - ((yy + .5) / size) * 2

    def mean_quantized(values, scale):
        integer = np.rint(np.asarray(values, dtype=np.float32) * np.float32(scale)).astype(np.int64)
        sums = np.bincount(inverse, weights=integer, minlength=len(unique)).astype(np.int64)
        # HLSL InterlockedAdd uses 32-bit integers, including wraparound.
        sums = sums.astype(np.int32).astype(np.float32)
        return sums * np.float32(1 / scale) * (np.float32(1) / count.astype(np.float32))

    return {"texel": unique, "count": count, "distance": mean_quantized(distance, 1000),
            "cosinc": mean_quantized(incidence, 1e6), "cosforward": mean_quantized(camera_cosine, 1e6),
            "cosnormalz": mean_quantized(raw_z, 1e6),
            "ndcx": mean_quantized(ndcx, 1e6), "ndcy": mean_quantized(ndcy, 1e6),
            "visible_pixels": len(xx)}


def sampling_events(trajectory, cfg, *, gun_mode="trajectory"):
    """Yield a deterministic tick/substep schedule; time is simulated seconds."""
    if gun_mode not in ("trajectory", "off"):
        raise ValueError("gun_mode must be trajectory or off")
    points, segments = trajectory["points"], trajectory["segments"]
    times = np.asarray([p["time_s"] for p in points], dtype=float)
    positions = np.asarray([p["position_mm"] for p in points], dtype=float) / 1000.
    rotations = np.asarray([p["rotation_matrix"] for p in points], dtype=float)
    if (len(points) < 2 or len(segments) != len(points) - 1 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)
            or positions.shape != (len(points), 3) or not np.isfinite(positions).all()
            or rotations.shape != (len(points), 3, 3) or not np.isfinite(rotations).all()):
        raise ValueError("invalid timed trajectory")
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6):
        raise ValueError("trajectory rotations must be right-handed orthonormal matrices")
    for i, s in enumerate(segments):
        if s["start"] != i or s["end"] != i + 1 or type(s["spray_on"]) is not bool:
            raise ValueError("segments must be contiguous with explicit gun states")
    count = int(math.ceil(times[-1] / cfg["tick_interval_s"]))
    if count > cfg["max_ticks"]:
        raise ValueError("trajectory exceeds configured tick budget")
    previous = None
    for tick in range(count):
        t = tick * cfg["tick_interval_s"]
        i = min(int(np.searchsorted(times, t, side="right") - 1), len(segments) - 1)
        alpha = (t - times[i]) / (times[i + 1] - times[i])
        p = positions[i] * (1 - alpha) + positions[i + 1] * alpha
        f = _unit(rotations[i, :, 2] * (1 - alpha) + rotations[i + 1, :, 2] * alpha)
        u = _unit(rotations[i, :, 1] * (1 - alpha) + rotations[i + 1, :, 1] * alpha)
        active = segments[i]["spray_on"] and gun_mode != "off"
        if active:
            n = cfg["pose_substeps"]
            # Archived StepWithSubsteps sends a newly active tool only at the
            # final substep, without dividing its original flux.
            if previous is None or n == 1:
                yield tick, t, p, f, u, 1.
            else:
                for k in range(1, n + 1):
                    a = k / n
                    yield tick, t, previous[0] * (1 - a) + p * a, _unit(previous[1] * (1 - a) + f * a), _unit(previous[2] * (1 - a) + u * a), 1 / n
            previous = (p, f, u)
        else:
            previous = None


def simulate(mesh, atlas_uv, trajectory, cfg, *, gun_mode="trajectory"):
    transform = np.asarray(trajectory["world_transform"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("invalid world transform")
    vertices = (mesh["vertices"] @ transform[:3, :3].T + transform[:3, 3]) / 1000.
    triangles = vertices[mesh["faces"]]
    normals = mesh["normals"] @ transform[:3, :3].T
    side = max(2, math.ceil(math.sqrt(len(triangles))) * cfg["atlas_texels_per_cell"])
    if side > 4096:
        raise ValueError("atlas exceeds 4096; reduce mesh size or texels per cell")
    values = np.zeros(side * side, dtype=np.int32)
    history = []
    started = time.monotonic()
    for tick, simulated_time, position, forward, up, weight in sampling_events(trajectory, cfg, gun_mode=gun_mode):
        if time.monotonic() - started > cfg["max_runtime_s"]:
            raise TimeoutError("reconstruction runtime budget exceeded; no completed result")
        hits = raster_hits(triangles, normals, atlas_uv, position, forward, up, cfg, side,
                           deadline=started + cfg["max_runtime_s"])
        changed = 0
        if hits is not None:
            parameters = {**cfg["kernel_parameters"], "FluxScale": cfg["kernel_parameters"]["FluxScale"] * weight}
            delta = compute(parameters, hits["ndcx"], hits["ndcy"], hits["distance"],
                            hits["cosinc"], hits["cosforward"], hits["count"])
            ix = hits["texel"]
            updated = np.clip(values[ix].astype(np.int64) + delta.astype(np.int64), 0, INT32_MAX).astype(np.int32)
            changed = int(np.count_nonzero(updated != values[ix]))
            values[ix] = updated
        history.append({"tick": tick, "simulation_time_s": simulated_time, "flux_weight": weight,
                        "position_m": position.tolist(), "forward": forward.tolist(), "up": up.tolist(),
                        "gun_on": True, "visible_pixels": 0 if hits is None else hits["visible_pixels"],
                        "texels_updated": changed})
    raw = values.reshape(side, side)
    stats = {"version": VERSION, "source_sha256": SOURCE_SHA256,
             "raw_array_sha256": hashlib.sha256(raw.astype('<i4').tobytes()).hexdigest(),
             "simulation_duration_s": trajectory["duration_s"], "wall_seconds": time.monotonic() - started,
             "paint_samples": len(history), "gun_mode": gun_mode,
             "nonzero_texels": int(np.count_nonzero(raw)), "raw_sum": int(raw.astype(np.int64).sum()),
             "raw_max": int(raw.max()), "shape": list(raw.shape), "dtype": raw.dtype.str,
             "raw_unit": "relative", "row_origin": "uv_v0", "native_eps_executed": False,
             "calibrated_physical_thickness": False,
             "fidelity": {"delta_operator": "82 archived original-DLL cases matched in separate audit",
                          "visibility": "CPU pixel-centre orthographic z-buffer with original face-order ties",
                          "temporal": "configured deterministic pose_substeps; active-tool cache resets on off ticks",
                          "unverified": ["hardware raster edge/depth equality", "camera factory equivalence",
                                         "GPU asynchronous readback timing", "full native scene equivalence"]}}
    return raw, stats, history
