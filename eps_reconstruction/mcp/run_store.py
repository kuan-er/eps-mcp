"""Persistent run admission and recovery for EPS simulation backends."""
from __future__ import annotations

import json
import re
import uuid
import os
import psutil
from datetime import datetime, timezone
from pathlib import Path

from spray_core import digest, write_json

RUN_MANAGER = "eps_mcp_native_v1"


def _owner_is_alive(pid):
    """Query process state without sending a Windows console control event."""
    if type(pid) is not int or pid <= 0:
        return False
    try:
        return psutil.Process(pid).is_running()
    except (psutil.Error, OSError, OverflowError, ValueError, TypeError):
        return False


def recover_pending(case_dir, run_id: str, *, at=None, process_alive=None):
    """Reconcile a stale admission without ever replaying native control.

    Only records explicitly owned by this manager may be recovered. External
    backend records remain read-only. A future adapter must set run_manager,
    owner_pid and deadline_utc before entering queued or running.
    """
    if not re.fullmatch(r"run-[a-f0-9]{32}", run_id):
        raise ValueError("invalid run_id")
    path = Path(case_dir) / "runs" / run_id / "run.json"
    if path.is_symlink() or not path.is_file():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("run_id") != run_id or run.get("state") not in ("queued", "running"):
        return run
    own_native = run.get("run_manager") == RUN_MANAGER and run.get("backend") == "eps_native_scene"
    own_reference = run.get("run_manager") == "eps_mcp_reconstruction_v1" and run.get("backend") == "cpu_reference"
    if not (own_native or own_reference):
        return run
    now = at or datetime.now(timezone.utc)
    deadline = run.get("deadline_utc")
    expired = deadline is not None and now >= datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    pid = run.get("owner_pid")
    if process_alive is None:
        process_alive = _owner_is_alive
    orphaned = type(pid) is not int or pid <= 0 or not process_alive(pid)
    if not (expired or orphaned):
        return run
    attempted = own_native and run.get("native_control_attempted") is True
    run.update(state="interrupted" if attempted or own_reference else "blocked",
               error_code="RUN_TIMEOUT" if expired else "RUN_OWNER_LOST",
               reason="run deadline passed" if expired else "run owner unavailable after restart",
               effect_status="LOCAL_PARTIAL" if own_reference else "UNKNOWN" if attempted else "NO_EFFECT",
               updated_at=now.isoformat(), recovered_at=now.isoformat(),
               automatic_retry=False)
    temporary = path.with_suffix(".recovery.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(run, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return run


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("registered immutable artifact is missing or a symlink")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(workflow, case_id: str, trajectory_id: str):
    case_dir = workflow._case(case_id)
    if not re.fullmatch(r"trajectory-[a-f0-9]{32}", trajectory_id):
        raise ValueError("invalid trajectory_id")
    trajectory_path = case_dir / "trajectories" / trajectory_id / "trajectory.json"
    case_path = case_dir / "case.json"
    case, trajectory = _read(case_path), _read(trajectory_path)
    if trajectory.get("case_id") != case_id or trajectory.get("trajectory_id") != trajectory_id:
        raise ValueError("trajectory does not belong to case")
    if trajectory.get("input_sha256") != case.get("input_sha256"):
        raise ValueError("trajectory input hashes differ from case")
    if trajectory.get("gun_profile_sha256") != case.get("gun_profile_sha256"):
        raise ValueError("trajectory gun profile differs from case")
    if trajectory.get("world_transform") != case.get("planning_to_world"):
        raise ValueError("trajectory world transform differs from case")
    files = {
        "case_manifest_sha256": case_path,
        "input_mesh_sha256": case_dir / "mesh.npz",
        "input_summary_sha256": case_dir / "summary.json",
        "gun_profile_sha256": case_dir / "gun_profile.json",
        "face_atlas_sha256": case_dir / "face_atlas.npz",
        "trajectory_sha256": trajectory_path,
    }
    hashes = {name: digest(path) for name, path in files.items()}
    for name, expected in (("input_mesh_sha256", case["input_sha256"]["mesh"]),
                           ("input_summary_sha256", case["input_sha256"]["summary"]),
                           ("gun_profile_sha256", case["gun_profile_sha256"]),
                           ("face_atlas_sha256", case["artifacts"]["face_atlas.npz"]["sha256"])):
        if hashes[name] != expected:
            raise ValueError(name + " changed since case preparation")
    return case_dir, case, trajectory, hashes


def admit(workflow, case_id: str, trajectory_id: str) -> dict:
    case_dir, case, trajectory, hashes = validate_inputs(workflow, case_id, trajectory_id)
    # A unique run directory is the admission record. No native action occurs here.
    run_id = "run-" + uuid.uuid4().hex
    run_dir = case_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    from workflow_tools import now
    reason = "native EPS scene, coordinate/controller binding and control adapter are not verified"
    run = {"schema": "eps-run/1", "case_id": case_id, "run_id": run_id,
           "run_manager": RUN_MANAGER,
           "trajectory_id": trajectory_id, "backend": "eps_native_scene",
           "state": "blocked", "created_at": now(), "updated_at": now(),
           "reason": reason, "error_code": "NATIVE_ADAPTER_UNVERIFIED",
           "native_control_attempted": False, "effect_status": "NO_EFFECT",
           "automatic_retry": False,
           "execution_validation": {"native_scene_verified": False,
                                    "controller_verified": False,
                                    "coordinate_mapping_verified": False,
                                    "native_adapter_verified": False},
           "evaluation_policy_sha256": case.get("evaluation_policy_sha256"),
           "execution_evidence": [], **hashes}
    write_json(run_dir / "run.json", run)
    return {"case_id": case_id, "trajectory_id": trajectory_id, "run_id": run_id,
            "state": "blocked", "reason": reason, "error_code": run["error_code"],
            "backend": run["backend"], "native_control_attempted": False,
            "effect_status": "NO_EFFECT"}
