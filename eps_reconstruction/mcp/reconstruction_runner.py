"""Execute a reconstructed run and publish artifacts before its final manifest."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

from reconstructed_deposition import configuration, simulate, VERSION, SOURCE_SHA256
import eps_kernel
from run_store import validate_inputs
from spray_core import load_mesh, digest, write_json
from native_results import policy_hash


def now():
    return datetime.now(timezone.utc).isoformat()


def _publish(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    write_json(temporary, value)
    os.replace(temporary, path)


def run(workflow, case_id, trajectory_id, *, gun_mode="trajectory"):
    case_dir, case, trajectory, hashes = validate_inputs(workflow, case_id, trajectory_id)
    profile = json.loads((case_dir / "gun_profile.json").read_text(encoding="utf-8-sig"))
    cfg = configuration(profile)
    if gun_mode not in ("trajectory", "off"):
        raise ValueError("gun_mode must be trajectory or off")
    policy = case.get("evaluation_policy")
    if not policy or policy["row_origin"] != "uv_v0" or policy["unit"] != "relative":
        raise ValueError("reconstruction needs a frozen relative-unit policy with row_origin=uv_v0")
    run_id = "run-" + uuid.uuid4().hex
    root = case_dir / "runs" / run_id
    if root.parent.is_symlink() or not root.parent.resolve().is_relative_to(case_dir.resolve()):
        raise ValueError("run directory escapes the case")
    root.mkdir(parents=True, exist_ok=False)
    started = now()
    manifest = {"schema": "eps-run/1", "case_id": case_id, "run_id": run_id,
                "trajectory_id": trajectory_id, "backend": "cpu_reference",
                "implementation": VERSION, "implementation_sha256": SOURCE_SHA256,
                "kernel_sha256": digest(Path(eps_kernel.__file__)),
                "numpy_version": np.__version__, "configuration": cfg,
                "run_manager": "eps_mcp_reconstruction_v1",
                "deadline_utc": (datetime.now(timezone.utc) + timedelta(seconds=cfg["max_runtime_s"] + 10)).isoformat(),
                "configuration_sha256": policy_hash(cfg), "gun_mode": gun_mode,
                "state": "running", "created_at": started, "updated_at": started,
                "owner_pid": os.getpid(), "native_control_attempted": False,
                "automatic_retry": False, "effect_status": "LOCAL_COMPUTATION",
                "execution_validation": {"native_scene_verified": False, "robot_executed": False,
                                         "reconstructed_backend": True, "calibrated_physical_thickness": False},
                "evaluation_policy_sha256": case["evaluation_policy_sha256"],
                "execution_evidence": [], **hashes}
    _publish(root / "run.json", manifest)
    try:
        mesh = load_mesh(case_dir / "mesh.npz", case_dir / "summary.json")
        with np.load(case_dir / "face_atlas.npz", allow_pickle=False) as atlas:
            uv = atlas["uv"].copy()
        raw, stats, history = simulate(mesh, uv, trajectory, cfg, gun_mode=gun_mode)
        # Every pixel is defined by this finite numerical model, including zero.
        # This is distinct from a missing/unknown external measurement.
        np.savez_compressed(root / "raw_deposition.npz", values=raw,
                            known_mask=np.ones(raw.shape, dtype=bool))
        write_json(root / "execution.json", stats)
        with (root / "paint-samples.jsonl").open("x", encoding="utf-8") as stream:
            for event in history:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
        write_json(root / "configuration.json", {"configuration": cfg, "gun_mode": gun_mode,
                                                 "trajectory_sha256": hashes["trajectory_sha256"]})
        # Recheck frozen inputs before publishing completed state.
        _, _, _, after = validate_inputs(workflow, case_id, trajectory_id)
        if hashes != after:
            raise ValueError("immutable inputs changed during reconstruction")
        evidence = []
        for filename in ("execution.json", "paint-samples.jsonl", "configuration.json"):
            p = root / filename
            evidence.append({"relative_path": filename, "size": p.stat().st_size, "sha256": digest(p)})
        manifest.update(state="completed", updated_at=now(), raw_sha256=digest(root / "raw_deposition.npz"),
                        execution_evidence=evidence, raw_array_sha256=stats["raw_array_sha256"])
        _publish(root / "run.json", manifest)
        return {"case_id": case_id, "run_id": run_id, "trajectory_id": trajectory_id,
                "state": "completed", "backend": "cpu_reference", "implementation": VERSION,
                "native_eps_executed": False, "summary": stats,
                "run_manifest": {"path": str(root / "run.json"), "sha256": digest(root / "run.json")}}
    except Exception as exc:
        manifest.update(state="failed", updated_at=now(), reason=str(exc),
                        error_code="RECONSTRUCTION_FAILED")
        _publish(root / "run.json", manifest)
        write_json(root / "failure.json", {"error": str(exc), "at": now(),
                                            "completed_raw_result_available": False})
        return {"case_id": case_id, "run_id": run_id, "trajectory_id": trajectory_id,
                "state": "failed", "backend": "cpu_reference", "reason": str(exc),
                "error_code": "RECONSTRUCTION_FAILED", "native_eps_executed": False}
