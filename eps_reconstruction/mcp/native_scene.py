"""Bind registered native EPS inputs to the MCP run and result contracts."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from native_results import current_evaluator, freeze_policy, policy_hash
from run_store import RUN_MANAGER, validate_inputs
from spray_core import digest, load_mesh, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def receipt(path):
    return {"path": str(path), "sha256": digest(path), "size": path.stat().st_size}


def registered_scene(scene_id):
    registry = os.environ.get("EPS_MCP_NATIVE_SCENES")
    if not registry or not Path(registry).is_absolute():
        raise ValueError("Set EPS_MCP_NATIVE_SCENES to an absolute installer registry path")
    document = read(registry)
    if scene_id not in document["scenes"]:
        raise ValueError("Native EPS scene is not registered")
    scene = document["scenes"][scene_id]
    source = {key: Path(scene[key]).resolve(strict=True) for key in ("project", "solution", "program", "mesh")}
    project_root = source["project"].parent
    if any(not path.is_file() or not path.is_relative_to(project_root) for path in source.values()):
        raise ValueError("All native scene files must stay in the isolated project")
    if tuple(path.suffix.lower() for path in source.values()) != (".cmaproj", ".cmasln", ".xpl", ".obj"):
        raise ValueError("The native scene needs an EPS project, solution, XPL program and UV OBJ")
    project, solution = read(source["project"]), read(source["solution"])
    robots = project.get("RobotInfos") or []
    if (len(robots) != 1 or robots[0].get("IpAddress") not in ("127.0.0.1", "::1")
            or Path(project["FileName"]).resolve() != source["project"]
            or Path(solution["FileName"]).resolve() != source["solution"]):
        raise ValueError("Expected a self-contained project with one loopback robot")
    programs, pieces = solution["RobotsPrograms"], solution["PiecesFeatures"]
    if (len(programs) != 1 or programs[0].get("ProgramName") != source["program"].name
            or len(pieces) != 1 or pieces[0].get("Name") != source["mesh"].name or not pieces[0].get("Paintable")):
        raise ValueError("The registered solution must bind the exact XPL program and paintable OBJ")
    piece = pieces[0]
    offset = np.asarray(piece["Offset"], dtype=float)
    if (offset.shape != (6,) or not np.isfinite(offset).all() or np.any(offset[3:]) or piece.get("FrameName")
            or piece.get("Rotation", {}).get("Enabled") or piece.get("Dynamic", {}).get("Enabled")):
        raise ValueError("This EPS adapter currently supports one stationary workpiece with a translation")
    origin = scene["journal_row_origin"]
    if origin not in ("uv_v0", "uv_v1"):
        raise ValueError("Configure a verified native journal row origin")
    return document["worker"], scene, source, robots[0], offset[:3]


def parse_obj(path):
    vertices, uv, faces, texture_faces = [], [], [], []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            words = line.partition("#")[0].split()
            if not words:
                continue
            kind, *values = words
            if kind == "v" and len(values) >= 3:
                vertices.append(tuple(map(float, values[:3])))
            elif kind == "vt" and len(values) >= 2:
                uv.append(tuple(map(float, values[:2])))
            elif kind == "f":
                if len(values) != 3:
                    raise ValueError("The native paintable OBJ must be triangulated")
                pairs = [value.split("/") for value in values]
                faces.append(tuple(int(pair[0]) for pair in pairs))
                texture_faces.append(tuple(int(pair[1]) for pair in pairs))
    v, t, f, tf = np.asarray(vertices), np.asarray(uv), np.asarray(faces), np.asarray(texture_faces)
    if (not len(v) or not len(t) or not len(f) or not np.isfinite(v).all() or not np.isfinite(t).all()
            or len(f) > 200000 or f.min() <= 0 or f.max() > len(v) or tf.min() <= 0 or tf.max() > len(t)
            or t.min() < 0 or t.max() > 1):
        raise ValueError("The native paintable OBJ requires finite vertices, triangles and UVs in [0, 1]")
    return v, f - 1, t[tf - 1]


def prepare(workflow, scene_id):
    _, scene, source, robot, translation = registered_scene(scene_id)
    sources = {name: receipt(path) for name, path in source.items()}
    v, f, uv = parse_obj(source["mesh"])
    cid, tid = "case-" + uuid.uuid4().hex, "trajectory-" + uuid.uuid4().hex
    folder = workflow.root / cid
    path = folder / "trajectories" / tid
    path.mkdir(parents=True, exist_ok=False)
    mesh_path, summary_path = folder / "mesh.npz", folder / "summary.json"
    np.savez_compressed(mesh_path, vertices=v / 1000, faces=f, part_id=np.zeros(len(f), dtype=np.int64),
                        spray_target=np.ones(len(f), dtype=bool))
    write_json(summary_path, {"category_id": "native_eps_workpiece", "part_legend": {"0": "workpiece"}})
    mesh = load_mesh(mesh_path, summary_path)
    mapping = folder / "face_atlas.npz"
    np.savez_compressed(mapping, uv=uv, face_ids=np.arange(len(f)), area_m2=mesh["area_mm2"] / 1e6)
    transform = np.eye(4)
    transform[:3, 3] = mesh["source_center_mm"] + translation
    profile = folder / "gun_profile.json"
    write_json(profile, {"profile_id": scene_id + ":project", "version": sources["project"]["sha256"],
                         "data_origin": "native_eps_project", "output_unit": "relative",
                         "calibrated": False, "settings": robot.get("GunsFeatures")})
    policy = freeze_policy({**read(workflow.catalog_path)["evaluation_policy"], "row_origin": scene["journal_row_origin"]})
    inputs = {"mesh": digest(mesh_path), "summary": digest(summary_path)}
    case = {"schema": "eps-workpiece-case/1", "case_id": cid, "created_at": now(), "asset_id": scene_id,
            "input_sha256": inputs, "gun_profile_sha256": digest(profile), "planning_to_world": transform.tolist(),
            "evaluation_policy": policy, "evaluation_policy_sha256": policy_hash(policy),
            "evaluation_implementation": current_evaluator(), "artifacts": {"face_atlas.npz": receipt(mapping)},
            "native_scene": {"id": scene_id, "sources": sources, "prim_path": scene["prim_path"],
                             "journal_row_origin": scene["journal_row_origin"]}}
    write_json(path / "trajectory.json", {"schema": "eps-native-program/1", "case_id": cid, "trajectory_id": tid,
                "input_sha256": inputs, "gun_profile_sha256": case["gun_profile_sha256"],
                "world_transform": case["planning_to_world"], "native_program": sources["program"],
                "target_scope": {"face_ids": list(range(len(f)))}, "duration_s": None})
    if any(digest(source[name]) != info["sha256"] for name, info in sources.items()):
        raise RuntimeError("A native input changed during case preparation")
    write_json(folder / "case.json", case)
    return {"case_id": cid, "trajectory_id": tid, "state": "native_case_prepared", "native_scene": case["native_scene"]}


def same_mesh(source, world_translation, captured):
    v, f, uv = parse_obj(source)

    def signature(points, uv_coordinates):
        corners = np.concatenate((np.round(points, 3), np.round(uv_coordinates, 6)), axis=-1)
        return sorted(tuple(sorted(tuple(corner) for corner in triangle)) for triangle in corners)

    with np.load(captured, allow_pickle=False) as data:
        if signature(v[f] + world_translation, uv) != signature(data["triangles_mm"], data["uv"]):
            raise ValueError("Recorded EPS mesh geometry, world position or UV mapping differs from the registered input")


def run(workflow, case_id, trajectory_id):
    from native_export import run_export
    from native_worker import ensure_worker
    folder, case, trajectory, hashes = validate_inputs(workflow, case_id, trajectory_id)
    binding = case["native_scene"]
    run_id = "run-" + uuid.uuid4().hex
    output = folder / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "eps-run/1", "case_id": case_id, "trajectory_id": trajectory_id, "run_id": run_id,
                "backend": "eps_native_scene", "state": "queued", "created_at": now(), "updated_at": now(),
                "run_manager": RUN_MANAGER, "owner_pid": os.getpid(),
                "deadline_utc": (datetime.now(timezone.utc) + timedelta(seconds=170)).isoformat(),
                "native_control_attempted": False, "effect_status": "NO_EFFECT", "automatic_retry": False,
                "execution_evidence": [], "evaluation_policy_sha256": case["evaluation_policy_sha256"], **hashes}

    def publish():
        temp = output / ("run-" + uuid.uuid4().hex + ".tmp")
        write_json(temp, manifest)
        temp.replace(output / "run.json")

    def before_control():
        manifest.update(state="running", native_control_attempted=True, effect_status="UNKNOWN", updated_at=now())
        publish()

    publish()
    try:
        worker, scene, source, _, translation = registered_scene(binding["id"])
        if (binding["prim_path"] != scene["prim_path"] or binding["journal_row_origin"] != scene["journal_row_origin"]
                or trajectory.get("native_program") != binding["sources"]["program"]
                or binding["sources"] != {name: receipt(path) for name, path in source.items()}):
            raise ValueError("Native scene binding changed since case preparation")
        pid = ensure_worker(worker, source["project"])
        raw = run_export(project=source["project"], solution=source["solution"], executable=worker["executable"],
                         blender=worker["blender"], bridge_root=worker["bridge_root"], pid=pid, piece=source["mesh"].name,
                         program=source["program"].name, prim_path=scene["prim_path"], output=output,
                         before_control=before_control, module_sha256=worker["module_sha256"])
        same_mesh(source["mesh"], translation, output / "native-paint-journal.mesh.npz")
        if hashes != validate_inputs(workflow, case_id, trajectory_id)[3]:
            raise RuntimeError("Frozen case inputs changed while EPS was running")
        if binding["sources"] != {name: receipt(path) for name, path in source.items()}:
            raise RuntimeError("Native project, program, solution or workpiece changed while EPS was running")
        manifest.update(state="completed", effect_status="NATIVE_EXPORT_WRITTEN", raw_sha256=raw["raw_sha256"],
                        execution_evidence=[{"relative_path": p.name, "size": p.stat().st_size, "sha256": digest(p)}
                                            for p in sorted(output.iterdir()) if p.is_file() and p.name != "run.json"],
                        execution_validation={"native_scene_verified": True, "recorded_mesh_uv_and_position_verified": True,
                                              "controller_program": source["program"].name,
                                              "physical_thickness_calibrated": False}, updated_at=now())
    except Exception as exc:
        attempted = manifest["native_control_attempted"]
        manifest.update(state="interrupted" if attempted else "blocked", reason=str(exc),
                        error_code="NATIVE_EXECUTION_UNCONFIRMED" if attempted else "NATIVE_PREFLIGHT_FAILED",
                        updated_at=now())
    publish()
    return {key: manifest[key] for key in ("state", "backend", "case_id", "run_id", "trajectory_id", "effect_status")}
