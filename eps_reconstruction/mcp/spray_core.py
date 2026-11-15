"""Portable geometry contribution for EPS MCP. Native EPS execution is separate.

Positions describe the nozzle in bbox-centred millimetres. Rotation columns are
tool axes; tool +Z points nozzle-to-surface. All segment states are explicit.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path
import zipfile

import numpy as np


VERSION = "eps-portable-core/1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def rigid_transform(value=None):
    m = np.eye(4) if value is None else np.asarray(value, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("world_transform must be a finite 4x4 matrix")
    r = m[:3, :3]
    if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-9) or not np.allclose(
            r.T @ r, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(r), 1, atol=1e-8):
        raise ValueError("world_transform must be rigid, right-handed, without scale")
    return m


def view_basis(axis):
    w = np.asarray(axis, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or np.linalg.norm(w) < 1e-12:
        raise ValueError("view axis must be a finite nonzero 3-vector")
    w = w / np.linalg.norm(w)
    ref = np.eye(3)[np.argmin(np.abs(w))]
    u = ref - w * np.dot(ref, w)
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(w, u), w))


def load_mesh(mesh_path, summary_path):
    mesh_path, summary_path = Path(mesh_path), Path(summary_path)
    if mesh_path.stat().st_size > 64 * 1024**2:
        raise ValueError("mesh archive exceeds 64 MiB limit")
    with zipfile.ZipFile(mesh_path) as archive:
        if sum(x.file_size for x in archive.infolist()) > 256 * 1024**2:
            raise ValueError("mesh archive expands beyond 256 MiB")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    legend = {int(k): str(v) for k, v in summary["part_legend"].items()}
    with np.load(mesh_path, allow_pickle=False) as z:
        v = np.asarray(z["vertices"], dtype=float)
        f, labels, target = z["faces"].copy(), z["part_id"].copy(), z["spray_target"].copy()
        if "spray_part" in z and not np.array_equal(z["spray_part"], labels):
            raise ValueError("spray_part and part_id disagree")
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError("vertices must be finite nonempty Nx3")
    if f.ndim != 2 or f.shape[1] != 3 or not len(f) or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("faces must be nonempty integer Mx3")
    if f.min() < 0 or f.max() >= len(v) or len(f) > 200000:
        raise ValueError("face indices invalid or face count exceeds 200000")
    if labels.shape != (len(f),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("part_id must contain one integer per face")
    if target.shape != (len(f),) or target.dtype != np.bool_:
        raise ValueError("spray_target must be one boolean per face")
    if not set(np.unique(labels)).issubset(legend):
        raise ValueError("legend does not describe all part IDs")
    center = (v.min(0) + v.max(0)) * 500.0
    vertices = v * 1000.0 - center
    tri = vertices[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if np.any(twice_area <= 1e-10):
        raise ValueError("degenerate triangles must be repaired before planning")
    return {"vertices": vertices, "faces": f.astype(np.int64), "part_id": labels,
            "target": target, "normals": cross / twice_area[:, None],
            "area_mm2": twice_area / 2, "legend": legend,
            "category": summary.get("category_id"), "source_center_mm": center}


def select_faces(mesh, semantic, axes, max_slope_deg=35):
    if not math.isfinite(max_slope_deg) or not 0 < max_slope_deg < 89:
        raise ValueError("max_slope_deg must be between 0 and 89")
    if not 1 <= len(axes) <= 12:
        raise ValueError("provide 1 to 12 outward view axes")
    bases = [view_basis(a) for a in axes]
    ids = [i for i, name in mesh["legend"].items() if name == semantic]
    if not ids:
        raise ValueError(f"semantic {semantic!r} is absent from this category's legend")
    requested = mesh["target"] & np.isin(mesh["part_id"], ids)
    dots = mesh["normals"] @ np.stack([b[:, 2] for b in bases], axis=1)
    assignment = np.argmax(dots, axis=1)
    chosen = requested & (dots.max(axis=1) >= np.cos(np.deg2rad(max_slope_deg)))
    if not chosen.any():
        raise ValueError("no target faces match the selected semantic and view axes")
    assignment[~chosen] = -1
    return assignment, bases, requested


def write_obj_atlas(mesh, path, mapping_path):
    """Nonoverlapping per-face UV triangles; IDs preserve original face order."""
    faces, vertices = mesh["faces"], mesh["vertices"]
    side = math.ceil(math.sqrt(len(faces)))
    cell = np.column_stack((np.arange(len(faces)) % side, np.arange(len(faces)) // side))
    uv = (cell[:, None, :] + np.array([[.08, .08], [.92, .08], [.08, .92]])) / side
    with Path(path).open("w", encoding="utf-8", newline="\n") as out:
        out.write("# bbox-centred millimetres; independent per-face UV atlas\n")
        for p in vertices:
            out.write("v %.12g %.12g %.12g\n" % tuple(p))
        for p in uv.reshape(-1, 2):
            out.write("vt %.12g %.12g\n" % tuple(p))
        for n in mesh["normals"]:
            out.write("vn %.12g %.12g %.12g\n" % tuple(n))
        for i, face in enumerate(faces):
            out.write("f " + " ".join(f"{int(v)+1}/{3*i+j+1}/{i+1}"
                                      for j, v in enumerate(face)) + "\n")
    np.savez_compressed(mapping_path, uv=uv, face_ids=np.arange(len(faces)),
                        area_m2=mesh["area_mm2"] / 1e6)


def _row_intervals(tri, y):
    """Analytical triangle/scanline intervals; no point sampling across holes."""
    rows = []
    for face, t in enumerate(tri):
        xs = []
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            dy = b[1] - a[1]
            if abs(dy) < 1e-12:
                if abs(y - a[1]) < 1e-9:
                    xs.extend([a[0], b[0]])
            elif min(a[1], b[1]) - 1e-9 <= y <= max(a[1], b[1]) + 1e-9:
                xs.append(a[0] + (y - a[1]) * (b[0] - a[0]) / dy)
        if len(xs) < 2 or max(xs) - min(xs) < 1e-8:
            continue
        n = np.cross(t[1] - t[0], t[2] - t[0])
        if abs(n[2]) < 1e-10:
            continue
        m = -n[0] / n[2]
        c = t[0, 2] + (n[0] * t[0, 0] - n[1] * (y - t[0, 1])) / n[2]
        rows.append((min(xs), max(xs), m, c, face))
    return rows


def raster(mesh, assignment, bases, spacing_mm, standoff_mm, point_step_mm, max_points=100000):
    strokes, count = [], 0
    for view, basis in enumerate(bases):
        selected = assignment == view
        if not selected.any():
            continue
        tri = mesh["vertices"][mesh["faces"]] @ basis
        points = tri[selected].reshape(-1, 3)
        lo, hi = points.min(0), points.max(0)
        nr = max(1, math.ceil((hi[1] - lo[1]) / spacing_mm))
        if nr > 10000:
            raise ValueError("scanline count exceeds 10000; change planning resolution")
        if len(tri) * nr > 2000000:
            raise ValueError("triangle/scanline budget exceeded; use a spatial accelerator")
        ys = lo[1] + (np.arange(nr) + .5) * (hi[1] - lo[1]) / nr
        for row, y in enumerate(ys):
            intervals = _row_intervals(tri, float(y))
            if not intervals:
                continue
            a = np.asarray(intervals)
            breaks = np.unique(a[:, :2])
            row_strokes = []
            for left, right in zip(breaks[:-1], breaks[1:]):
                if right - left < 1e-7:
                    continue
                mid = (left + right) / 2
                active = np.flatnonzero((a[:, 0] <= mid) & (a[:, 1] >= mid))
                if not len(active):
                    continue
                top = active[np.argmax(a[active, 2] * mid + a[active, 3])]
                face = int(a[top, 4])
                if not selected[face]:
                    continue
                nx = max(2, math.ceil((right - left) / point_step_mm) + 1)
                count += nx
                if count > max_points:
                    raise ValueError("trajectory exceeds point budget; increase spacing or point step")
                x = np.linspace(left, right, nx)
                p = np.column_stack((x, np.full(nx, y), a[top, 2] * x + a[top, 3])) @ basis.T
                n = mesh["normals"][face]
                # Join only contiguous coplanar pieces, keeping holes and sharp edges separate.
                if row_strokes and np.linalg.norm(row_strokes[-1]["surface"][-1] - p[0]) < 1e-6 and np.dot(row_strokes[-1]["normal"], n) > 1 - 1e-9:
                    old = row_strokes[-1]
                    old["surface"] = np.concatenate((old["surface"], p[1:]))
                    old["faces"].add(face)
                else:
                    row_strokes.append({"surface": p, "normal": n, "faces": {face}, "view": view})
            if row % 2:
                row_strokes.reverse()
                for s in row_strokes:
                    s["surface"] = s["surface"][::-1]
            for s in row_strokes:
                s["nozzle"] = s["surface"] + s["normal"] * standoff_mm
                strokes.append(s)
    if not strokes:
        raise ValueError("selected target has no visible continuous scanline")
    return strokes


def segment_hits_mesh(a, b, triangles):
    d = np.asarray(b) - np.asarray(a)
    if np.linalg.norm(d) < 1e-10:
        return False
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    p = np.cross(np.broadcast_to(d, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, p)
    good = np.abs(det) > 1e-10
    inv = np.zeros_like(det)
    inv[good] = 1 / det[good]
    s = np.asarray(a) - triangles[:, 0]
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = q @ d * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    return bool(np.any(good & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
                       & (t >= 0) & (t <= 1)))


def _enters_box(a, b, lo, hi):
    """Whether a segment enters the open interior of a clearance box."""
    t0, t1 = 0., 1.
    for p, d, low, high in zip(a, b - a, lo + 1e-7, hi - 1e-7):
        if abs(d) < 1e-12:
            if p < low or p > high:
                return False
        else:
            x, y = sorted(((low - p) / d, (high - p) / d))
            t0, t1 = max(t0, x), min(t1, y)
            if t0 > t1:
                return False
    return t0 <= t1


def outside_route(a, b, lo, hi):
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    nodes = np.vstack((a, b, corners))
    edges = [[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i):
            if not _enters_box(nodes[i], nodes[j], lo, hi):
                length = float(np.linalg.norm(nodes[i] - nodes[j]))
                edges[i].append((j, length))
                edges[j].append((i, length))
    queue, best = [(0., 0, [0])], {0: 0.}
    while queue:
        cost, i, path = heapq.heappop(queue)
        if i == 1:
            return nodes[path]
        if cost > best[i] + 1e-10:
            continue
        for j, length in edges[i]:
            nc = cost + length
            if nc < best.get(j, math.inf) - 1e-10:
                best[j] = nc
                heapq.heappush(queue, (nc, j, path + [j]))
    raise ValueError("no external clearance-box transition found")


def tool_rotation(aim, reference):
    z = np.asarray(aim, dtype=float)
    z = z / np.linalg.norm(z)
    x = reference - z * np.dot(reference, z)
    if np.linalg.norm(x) < 1e-8:
        reference = np.eye(3)[np.argmin(np.abs(z))]
        x = reference - z * np.dot(reference, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def euler_zyx(matrix):
    b = math.asin(float(np.clip(-matrix[2, 0], -1, 1)))
    if abs(math.cos(b)) > 1e-8:
        a, c = math.atan2(matrix[1, 0], matrix[0, 0]), math.atan2(matrix[2, 1], matrix[2, 2])
    else:
        a, c = math.atan2(-matrix[0, 1], matrix[1, 1]), 0.
    return np.rad2deg([a, b, c]).tolist()


def assemble(mesh, strokes, bases, speed_mm_s, transit_speed_mm_s, clearance_mm,
             world_transform=None, angular_speed_deg_s=90.):
    lo = mesh["vertices"].min(0) - clearance_mm
    hi = mesh["vertices"].max(0) + clearance_mm
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    transform = rigid_transform(world_transform)
    world_r, world_t = transform[:3, :3], transform[:3, 3]
    points, segments = [], []

    def append(p, rotation, on=False, stroke_id=-1, kind="transit"):
        p, rotation = world_r @ p + world_t, world_r @ rotation
        time = 0.
        if points:
            old = points[-1]
            length = float(np.linalg.norm(p - np.asarray(old["position_mm"])))
            delta = np.asarray(old["rotation_matrix"]).T @ rotation
            angle = math.degrees(math.acos(float(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
            if length < 1e-9 and angle < 1e-7:
                return
            requested_speed = speed_mm_s if on else transit_speed_mm_s
            dt = max(length / requested_speed, angle / angular_speed_deg_s, .001)
            time = old["time_s"] + dt
            segments.append({"start": len(points) - 1, "end": len(points), "spray_on": bool(on),
                             "speed_mm_s": length / dt, "requested_speed_mm_s": requested_speed,
                             "duration_s": dt, "stroke_id": stroke_id, "kind": kind})
        points.append({"position_mm": p.tolist(), "rotation_matrix": rotation.tolist(), "time_s": time})

    for sid, s in enumerate(strokes):
        w = bases[s["view"]][:, 2]
        r = tool_rotation(-s["normal"], bases[s["view"]][:, 0])
        start, end = s["nozzle"][0], s["nozzle"][-1]
        safe_height = float(max((corners @ w).max(), start @ w, end @ w) + clearance_mm)
        safe_start = start + w * (safe_height - np.dot(start, w))
        safe_end = end + w * (safe_height - np.dot(end, w))
        if points:
            previous = world_r.T @ (np.asarray(points[-1]["position_mm"]) - world_t)
            route = outside_route(previous, safe_start, lo, hi)
            old_r = world_r.T @ np.asarray(points[-1]["rotation_matrix"])
            for p in route[1:]:
                append(p, old_r, kind="external_transfer")
        else:
            append(safe_start, r)
        # Orientation changes at a safe external pose with the gun off.
        append(safe_start, r, kind="orient")
        append(start, r, kind="approach")
        for p in s["nozzle"][1:]:
            append(p, r, on=True, stroke_id=sid, kind="spray")
        append(safe_end, r, kind="retract")
    tri = mesh["vertices"][mesh["faces"]]
    if len(segments) * len(tri) > 150_000_000:
        raise ValueError("centerline collision audit exceeds budget; use a spatial accelerator")
    collisions = []
    for i, seg in enumerate(segments):
        a, b = (world_r.T @ (np.asarray(points[seg[k]]["position_mm"]) - world_t)
                for k in ("start", "end"))
        if segment_hits_mesh(a, b, tri):
            collisions.append(i)
    if collisions:
        raise ValueError(f"candidate rejected: {len(collisions)} centerline segments intersect mesh; first {collisions[:8]}")
    return {"schema": "eps-nozzle-trajectory/1", "generator": VERSION, "generator_sha256": SOURCE_SHA256,
            "frame": "EPS world transform of bbox-centred millimetres",
            "position_semantics": "nozzle position; standoff already included",
            "orientation_semantics": "rotation columns are tool XYZ; +Z is nozzle-to-surface",
            "segment_semantics": "spray_on applies throughout start->end; final gun state off",
            "points": points, "segments": segments, "final_spray_on": False,
            "stroke_count": len(strokes), "duration_s": points[-1]["time_s"],
            "geometry_checks": {"centerline_intersection_count": 0,
                                "all_spray_and_off_segments_checked": True,
                                "transition_clearance_box_mm": [lo.tolist(), hi.tolist()]},
            "robot_ready": False,
            "not_validated": ["robot current-pose to first safe pose", "robot IK and joint limits",
                              "gun-body and robot swept collision", "controller dynamics",
                              "native EPS execution", "physical deposition calibration",
                              "mesh self-intersections"],
            "world_transform": transform.tolist()}


def plan(mesh, semantic, axes, spacing_mm=80., standoff_mm=200., point_step_mm=12.,
         speed_mm_s=200., transit_speed_mm_s=300., clearance_mm=50.,
         max_slope_deg=35., world_transform=None):
    for name, value, upper in (("spacing", spacing_mm, 500), ("standoff", standoff_mm, 1000),
                               ("point_step", point_step_mm, 100), ("speed", speed_mm_s, 5000),
                               ("transit_speed", transit_speed_mm_s, 5000), ("clearance", clearance_mm, 1000)):
        if not math.isfinite(value) or not 1 <= value <= upper:
            raise ValueError(f"{name} must be finite and between 1 and {upper}")
    assignment, bases, requested = select_faces(mesh, semantic, axes, max_slope_deg)
    strokes = raster(mesh, assignment, bases, spacing_mm, standoff_mm, point_step_mm)
    result = assemble(mesh, strokes, bases, speed_mm_s, transit_speed_mm_s, clearance_mm, world_transform)
    selected = np.flatnonzero(assignment >= 0)
    result["target_scope"] = {"semantic": semantic, "view_axes": [b[:, 2].tolist() for b in bases],
                              "max_slope_deg": max_slope_deg, "face_ids": selected.tolist(),
                              "face_ids_sha256": hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
                              "selected_area_m2": float(mesh["area_mm2"][selected].sum() / 1e6),
                              "requested_semantic_area_m2": float(mesh["area_mm2"][requested].sum() / 1e6),
                              "whole_mesh_area_m2": float(mesh["area_mm2"].sum() / 1e6),
                              "whole_object_validation": False}
    result["parameters"] = {"spacing_mm": spacing_mm, "standoff_mm": standoff_mm,
                            "point_step_mm": point_step_mm, "speed_mm_s": speed_mm_s,
                            "transit_speed_mm_s": transit_speed_mm_s, "clearance_mm": clearance_mm}
    result["surface_strokes"] = [{"stroke_id": i, "surface_points_mm": s["surface"].tolist(),
                                   "nozzle_points_mm": s["nozzle"].tolist(),
                                   "face_ids": sorted(s["faces"]), "view_index": s["view"]}
                                  for i, s in enumerate(strokes)]
    return result


def write_trajectory_csv(path, trajectory):
    by_start = {s["start"]: s for s in trajectory["segments"]}
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        f.write("X;Y;Z;A;B;C;strokeId;time_s;speed_mm_s;spray_on\n")
        for i, p in enumerate(trajectory["points"]):
            s = by_start.get(i, {})
            values = p["position_mm"] + euler_zyx(np.asarray(p["rotation_matrix"]))
            values += [s.get("stroke_id", -1), p["time_s"], s.get("speed_mm_s", 0), int(s.get("spray_on", False))]
            f.write(";".join(format(float(v), ".12g") for v in values) + "\n")
