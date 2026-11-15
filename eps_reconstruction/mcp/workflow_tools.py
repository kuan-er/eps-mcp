"""Immutable case and trajectory tools for reconstructed and native EPS runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone
import uuid

import numpy as np

from spray_core import (VERSION, digest, load_mesh, plan, rigid_transform,
                        write_json, write_obj_atlas, write_trajectory_csv)
from native_results import ResultReader, freeze_policy, policy_hash, current_evaluator
from run_store import admit, recover_pending


def now():
    return datetime.now(timezone.utc).isoformat()


class Workflow:
    def __init__(self, catalog_path, output_root):
        self.catalog_path, self.root = Path(catalog_path).resolve(), Path(output_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.results = ResultReader(self)

    def _case(self, case_id):
        if not re.fullmatch(r"case-[a-f0-9]{32}", case_id):
            raise ValueError("invalid case_id")
        p = self.root / case_id
        if p.is_symlink() or not (p / "case.json").is_file():
            raise ValueError("unknown case_id")
        return p

    def prepare_case(self, asset_id: str, world_transform: list | None = None,
                     gun_profile_id: str = "eps_reconstruction_reference_v1") -> dict:
        """Prepare a registered mesh with millimetre frame and face/UV mapping.

        This writes immutable files; native EPS scene loading remains a separate
        recorded step. world_transform is rigid planning-mm -> EPS-world-mm.
        """
        transform = rigid_transform(world_transform)
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8-sig"))
        evaluation_policy = freeze_policy(catalog.get("evaluation_policy"))
        asset = catalog["assets"].get(asset_id)
        if not isinstance(asset, dict):
            raise ValueError("asset_id is not registered in the configured catalogue")
        if asset.get("units") != "m" or asset.get("up_axis") != "+Z":
            raise ValueError("asset must explicitly declare units=m and up_axis=+Z")
        sources = {n: Path(asset[n]).expanduser() for n in ("mesh", "summary")}
        if not all(p.is_absolute() and p.is_file() for p in sources.values()):
            raise ValueError("registered input files are missing")
        sources = {n: p.resolve() for n, p in sources.items()}
        if sources["summary"].stat().st_size > 1024**2:
            raise ValueError("summary exceeds 1 MiB")
        initial = {n: digest(p) for n, p in sources.items()}
        profile_path = Path(catalog.get("profiles", {}).get(gun_profile_id, ""))
        if not profile_path.is_absolute() or not profile_path.is_file() or profile_path.stat().st_size > 1024**2:
            raise ValueError("gun_profile_id must have a configured profile file")
        profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        if profile.get("profile_id") != gun_profile_id or not profile.get("version"):
            raise ValueError("profile identity/version missing or inconsistent")
        profile_hash = digest(profile_path)
        # Validate the complete source before allocating the new case.
        load_mesh(sources["mesh"], sources["summary"])
        case_id = "case-" + uuid.uuid4().hex
        directory = self.root / case_id
        directory.mkdir(exist_ok=False)
        try:
            for name, p in sources.items():
                dest = directory / ("mesh.npz" if name == "mesh" else "summary.json")
                shutil.copyfile(p, dest)
                if digest(dest) != initial[name] or digest(p) != initial[name]:
                    raise ValueError("source changed during copy; case rejected")
            mesh = load_mesh(directory / "mesh.npz", directory / "summary.json")
            shutil.copyfile(profile_path, directory / "gun_profile.json")
            if digest(directory / "gun_profile.json") != profile_hash or digest(profile_path) != profile_hash:
                raise ValueError("gun profile changed during copy")
            write_obj_atlas(mesh, directory / "mesh_mm.obj", directory / "face_atlas.npz")
            case = {"schema": "eps-workpiece-case/1", "case_id": case_id, "asset_id": asset_id,
                    "created_at": now(), "core_version": VERSION, "input_sha256": initial,
                    "gun_profile_id": gun_profile_id, "gun_profile_version": profile["version"],
                    "gun_profile_sha256": profile_hash,
                    "gun_profile_description": {key: profile.get(key) for key in
                        ("data_origin", "mechanism_family", "status", "hardware_identity", "calibrated",
                         "output_unit", "eps_native_scene_equivalence_verified")},
                    "evaluation_policy": evaluation_policy,
                    "evaluation_policy_sha256": None if evaluation_policy is None else policy_hash(evaluation_policy),
                    "evaluation_implementation": current_evaluator(),
                    "category": mesh["category"], "legend": mesh["legend"],
                    "source_frame": {"units": "m", "up_axis": "+Z"},
                    "planning_frame": {"units": "mm", "up_axis": "+Z", "origin": "source bounding-box centre",
                                       "source_center_mm": mesh["source_center_mm"].tolist(),
                                       "formula": "planning_mm = source_metres * 1000 - source_center_mm"},
                    "planning_to_world": transform.tolist(), "world_frame_verified": False,
                    "extents_mm": np.ptp(mesh["vertices"], axis=0).tolist(),
                    "whole_mesh_area_m2": float(mesh["area_mm2"].sum() / 1e6),
                    "face_count": len(mesh["faces"]), "vertex_count": len(mesh["vertices"]),
                    "state": "files_prepared", "native_scene_state": "not_loaded",
                    "mesh_obj_frame": "planning millimetres; apply planning_to_world exactly once",
                    "uv_mapping": "face_atlas.npz: original face ID, UV triangle, true area_m2",
                    "native_import_requirements": ["load isolated project/solution", "mark intended workpiece Paintable",
                                                   "verify current frame, gun axis, TCP and controller binding"]}
            case["artifacts"] = {p.name: {"path": str(p), "size": p.stat().st_size, "sha256": digest(p)}
                                 for p in sorted(directory.iterdir()) if p.is_file()}
            write_json(directory / "case.json", case)
            return case
        except Exception as exc:
            write_json(directory / "failure.json", {"at": now(), "stage": "prepare_case", "error": str(exc)})
            raise

    def generate_trajectory(self, case_id: str, semantic: str, view_axes: list,
                            spacing_mm: float = 80., standoff_mm: float = 200.,
                            point_step_mm: float = 12., speed_mm_s: float = 200.,
                            transit_speed_mm_s: float = 300., clearance_mm: float = 50.,
                            parent_trajectory_id: str | None = None, decision: dict | None = None) -> dict:
        """Generate an immutable candidate with nozzle pose, time and per-segment spray state.

        A revision requires the Agent's reason/expected_effect/changes. Generation
        and centreline geometry checks do not establish native EPS/robot validity.
        """
        directory = self._case(case_id)
        case = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        if parent_trajectory_id is not None:
            if not re.fullmatch(r"trajectory-[a-f0-9]{32}", parent_trajectory_id):
                raise ValueError("invalid parent_trajectory_id")
            if not (directory / "trajectories" / parent_trajectory_id / "trajectory.json").is_file():
                raise ValueError("parent must be an existing trajectory of the same case")
            if not isinstance(decision, dict) or any(not decision.get(k) for k in ("reason", "expected_effect", "changes")):
                raise ValueError("revision needs decision.reason, expected_effect and changes")
            parent = json.loads((directory / "trajectories" / parent_trajectory_id / "trajectory.json").read_text(encoding="utf-8"))
            previous = {**parent["parameters"], "semantic": parent["target_scope"]["semantic"],
                        "view_axes": parent["target_scope"]["view_axes"]}
            from spray_core import view_basis
            requested = {"spacing_mm": spacing_mm, "standoff_mm": standoff_mm, "point_step_mm": point_step_mm,
                         "speed_mm_s": speed_mm_s, "transit_speed_mm_s": transit_speed_mm_s,
                         "clearance_mm": clearance_mm, "semantic": semantic,
                         "view_axes": [view_basis(a)[:, 2].tolist() for a in view_axes]}
            changed = {k: v for k, v in requested.items() if previous.get(k) != v}
            if not changed or decision["changes"] != changed:
                raise ValueError(f"decision.changes must match actual changes: {changed}")
        for name, filename in (("mesh", "mesh.npz"), ("summary", "summary.json")):
            if digest(directory / filename) != case["input_sha256"][name]:
                raise ValueError("case input hash changed; refusing to revise its geometry")
        if digest(directory / "gun_profile.json") != case["gun_profile_sha256"]:
            raise ValueError("case gun profile hash changed")
        tid = "trajectory-" + uuid.uuid4().hex
        target = directory / "trajectories" / tid
        target.mkdir(parents=True, exist_ok=False)
        try:
            mesh = load_mesh(directory / "mesh.npz", directory / "summary.json")
            trajectory = plan(mesh, semantic, view_axes, spacing_mm, standoff_mm, point_step_mm,
                              speed_mm_s, transit_speed_mm_s, clearance_mm,
                              world_transform=case["planning_to_world"])
            trajectory.update(case_id=case_id, trajectory_id=tid, parent_trajectory_id=parent_trajectory_id,
                              created_at=now(), input_sha256=case["input_sha256"], decision=decision,
                              gun_profile_id=case["gun_profile_id"], gun_profile_sha256=case["gun_profile_sha256"])
            trajectory["same_target_scope_as_parent"] = (None if parent_trajectory_id is None else
                trajectory["target_scope"]["face_ids_sha256"] == parent["target_scope"]["face_ids_sha256"])
            write_json(target / "trajectory.json", trajectory)
            write_trajectory_csv(target / "trajectory.csv", trajectory)
            result = {"case_id": case_id, "trajectory_id": tid, "parent_trajectory_id": parent_trajectory_id,
                      "state": "candidate_generated", "robot_ready": False,
                      "points": len(trajectory["points"]), "segments": len(trajectory["segments"]),
                      "duration_s": trajectory["duration_s"], "stroke_count": trajectory["stroke_count"],
                      "target_scope": trajectory["target_scope"], "geometry_checks": trajectory["geometry_checks"],
                      "native_eps_verified": False, "not_validated": trajectory["not_validated"],
                      "artifacts": {p.name: {"path": str(p), "size": p.stat().st_size, "sha256": digest(p)}
                                    for p in sorted(target.iterdir()) if p.is_file()}}
            write_json(target / "generation.json", result)
            return result
        except Exception as exc:
            write_json(target / "failure.json", {"at": now(), "stage": "generate_trajectory", "error": str(exc),
                                                 "parent_trajectory_id": parent_trajectory_id, "decision": decision})
            raise


    def get_result(self, case_id: str, run_id: str, baseline_run_id: str | None = None) -> dict:
        """Read backend state, verify immutable artifacts, and evaluate completed raw results."""
        recover_pending(self._case(case_id), run_id)
        return self.results.read(case_id, run_id, baseline_run_id)

    def run_simulation(self, case_id: str, trajectory_id: str,
                       backend: str = "cpu_reference", gun_mode: str = "trajectory") -> dict:
        """Execute the selected deposition backend."""
        case = json.loads((self._case(case_id) / "case.json").read_text(encoding="utf-8"))
        if backend == "cpu_reference":
            if case.get("native_scene"):
                raise ValueError("This case binds an EPS XPL program. Use backend=eps_native_scene")
            from reconstruction_runner import run
            return run(self, case_id, trajectory_id, gun_mode=gun_mode)
        if backend == "eps_native_scene" and gun_mode == "trajectory":
            if case.get("native_scene"):
                from native_scene import run
                return run(self, case_id, trajectory_id)
            return admit(self, case_id, trajectory_id)
        raise ValueError("backend must be cpu_reference, or eps_native_scene with trajectory gun mode")


def register_tools(mcp):
    """Register business tools without altering the existing EPS read tools."""
    workflow = None

    def configured():
        nonlocal workflow
        if workflow is None:
            catalog, root = os.environ.get("EPS_MCP_ASSET_CATALOG"), os.environ.get("EPS_MCP_CASE_ROOT")
            if not catalog or not root or not Path(catalog).is_absolute() or not Path(root).is_absolute():
                raise RuntimeError("configure absolute EPS_MCP_ASSET_CATALOG and EPS_MCP_CASE_ROOT")
            workflow = Workflow(catalog, root)
        return workflow

    @mcp.tool()
    def prepare_case(asset_id: str, world_transform: list | None = None,
                     gun_profile_id: str = "eps_reconstruction_reference_v1") -> dict:
        """Prepare registered workpiece files, explicit units/frame and nonoverlapping UV mapping."""
        return configured().prepare_case(asset_id, world_transform, gun_profile_id)

    @mcp.tool()
    def prepare_native_scene(scene_id: str) -> dict:
        """Freeze a registered EPS project, solution, XPL program and paintable UV mesh for a native run."""
        from native_scene import prepare
        return prepare(configured(), scene_id)

    @mcp.tool()
    def generate_trajectory(case_id: str, semantic: str, view_axes: list,
                            spacing_mm: float = 80., standoff_mm: float = 200.,
                            point_step_mm: float = 12., speed_mm_s: float = 200.,
                            transit_speed_mm_s: float = 300., clearance_mm: float = 50.,
                            parent_trajectory_id: str | None = None, decision: dict | None = None) -> dict:
        """Generate/revise a candidate with full pose, speed/time, off-gun transitions and immutable history."""
        return configured().generate_trajectory(case_id, semantic, view_axes, spacing_mm, standoff_mm,
                                                point_step_mm, speed_mm_s, transit_speed_mm_s,
                                                clearance_mm, parent_trajectory_id, decision)

    @mcp.tool()
    def run_simulation(case_id: str, trajectory_id: str,
                       backend: str = "cpu_reference", gun_mode: str = "trajectory") -> dict:
        """Simulate timed trajectories with explicit cpu_reference or eps_native_scene backend.

        cpu_reference runs an offline reconstruction. gun_mode=off is its control
        experiment. eps_native_scene runs an isolated EPS project prepared by prepare_native_scene.
        """
        return configured().run_simulation(case_id, trajectory_id, backend, gun_mode)

    @mcp.tool()
    def get_result(case_id: str, run_id: str, baseline_run_id: str | None = None) -> dict:
        """Read a registered backend run and area-weighted results using the frozen case policy.

        Backend labels and execution claims remain separate from artifact integrity.
        Missing/running/failed jobs cannot be reported as completed deposition.
        Optional baseline_run_id compares completed runs of this case with the
        same backend, scope, mapping, policy, array layout and known-pixel mask.
        """
        return configured().get_result(case_id, run_id, baseline_run_id)
