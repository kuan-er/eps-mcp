"""Read backend run manifests and evaluate immutable scalar-array snapshots.

This module does not launch EPS. Backend execution claims and evidence remain
separate from the file-integrity and numerical checks performed here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import threading
import uuid

import numpy as np

from result_metrics import evaluate_files, SOURCE_SHA256 as EVALUATOR_SHA256, VERSION as EVALUATOR_VERSION
from spray_core import digest, load_mesh, write_json


def policy_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def current_evaluator():
    return {"version": EVALUATOR_VERSION, "source_sha256": EVALUATOR_SHA256,
            "numpy_version": np.__version__}


def freeze_policy(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("evaluation_policy must be a dictionary")
    defaults = {"minimum": None, "maximum": None, "clip_level": None,
                "zero_epsilon": 0., "samples_per_face": 128, "seed": 9173}
    allowed = set(defaults) | {"unit", "row_origin"}
    if set(value) - allowed:
        raise ValueError("unknown evaluation policy keys")
    p = {**defaults, **value}
    if p.get("unit") not in ("relative", "native_raw") or p.get("row_origin") not in ("uv_v0", "uv_v1"):
        raise ValueError("evaluation policy needs explicit uncalibrated unit and UV row origin")
    for name in ("minimum", "maximum", "clip_level", "zero_epsilon"):
        if p[name] is not None and (not isinstance(p[name], (int, float)) or isinstance(p[name], bool)
                                    or not math.isfinite(p[name]) or p[name] < 0):
            raise ValueError(f"invalid evaluation policy {name}")
    if p["minimum"] is not None and p["maximum"] is not None and p["minimum"] > p["maximum"]:
        raise ValueError("minimum cannot exceed maximum")
    if p["clip_level"] is not None and p["clip_level"] <= p["zero_epsilon"]:
        raise ValueError("clip_level must exceed zero_epsilon")
    if type(p["samples_per_face"]) is not int or not 16 <= p["samples_per_face"] <= 4096:
        raise ValueError("samples_per_face outside supported range")
    if type(p["seed"]) is not int or not 0 <= p["seed"] < 2**32:
        raise ValueError("seed must be a uint32")
    return p


class ResultReader:
    def __init__(self, workflow):
        self.workflow = workflow
        self.lock = threading.Lock()

    def _file(self, root, relative):
        if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
            raise ValueError("artifact path must be a portable relative path")
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("artifact path escapes the run")
        p = root / p
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not p.is_file():
            raise ValueError("artifact is missing or escapes its registered directory")
        return p

    def read(self, case_id, run_id, baseline_run_id=None):
        with self.lock:
            result = self._read(case_id, run_id)
            if baseline_run_id is not None:
                baseline = self._read(case_id, baseline_run_id)
                result["comparison"] = self._compare(result, baseline)
            return result

    @staticmethod
    def _compare(candidate, baseline):
        """Compare fixed-scope results; never award a coating-quality pass here."""
        result = {"baseline_run_id": baseline["run_id"],
                  "baseline_trajectory_id": baseline["trajectory_id"],
                  "baseline_run_manifest_sha256": baseline["run_manifest_sha256"],
                  "candidate_run_id": candidate["run_id"],
                  "delta_convention": "candidate_minus_baseline",
                  "native_execution_verified_by_comparison": False,
                  "quality_pass_decided_by_comparison": False}
        reasons = []
        for label, value in (("candidate", candidate), ("baseline", baseline)):
            if value["state"] != "completed":
                reasons.append(label + " is " + value["state"])
        if candidate["deposition_backend"] != baseline["deposition_backend"]:
            reasons.append("deposition backend differs")
        if candidate["deposition_backend"] == "cpu_reference" and candidate.get("reconstruction_identity") != baseline.get("reconstruction_identity"):
            reasons.append("reconstruction implementation or configuration differs")
        if not reasons:
            a, b = candidate["metrics"], baseline["metrics"]
            for field in ("target_face_ids_sha256", "mapping_sha256", "policy_sha256",
                          "array_shape", "array_dtype", "known_mask_sha256"):
                if a[field] != b[field]:
                    reasons.append(field + " differs")
        if reasons:
            return {**result, "state": "incomparable", "reasons": reasons, "metric_deltas": None}
        a, b = candidate["metrics"], baseline["metrics"]
        deltas = {name: (None if value is None or b["target"][name] is None else value - b["target"][name])
                  for name, value in a["target"].items()}
        return {**result, "state": "comparable", "reasons": [],
                "baseline_metrics_artifact": baseline["metrics_artifact"],
                "baseline_target": b["target"], "candidate_target": a["target"],
                "evaluation_policy_sha256": candidate["evaluation_policy_sha256"],
                "evaluation_implementation": candidate["evaluation_implementation"],
                "metric_deltas": {"target": deltas,
                    "non_target_overspray_area_estimate_m2": a["non_target_overspray_area_estimate_m2"] - b["non_target_overspray_area_estimate_m2"],
                                  "planned_duration_s": (None if candidate["planned_duration_s"] is None
                                                         or baseline["planned_duration_s"] is None else
                                                         candidate["planned_duration_s"] - baseline["planned_duration_s"])},
                "duration_is_planned_not_observed_runtime": True}

    def _read(self, case_id, run_id):
        case_dir = self.workflow._case(case_id)
        if not re.fullmatch(r"run-[a-f0-9]{32}", run_id):
            raise ValueError("invalid run_id")
        run_dir = case_dir / "runs" / run_id
        if run_dir.is_symlink() or not run_dir.resolve().is_relative_to(case_dir.resolve()):
            raise ValueError("run directory escapes the case")
        manifest_path = self._file(run_dir, "run.json")
        if manifest_path.stat().st_size > 4 * 1024**2:
            raise ValueError("run manifest is too large")
        run = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        case = json.loads(self._file(case_dir, "case.json").read_text(encoding="utf-8"))
        if run.get("case_id") != case_id or run.get("run_id") != run_id:
            raise ValueError("run IDs do not match its registered location")
        if run.get("backend") not in ("eps_native_scene", "eps_native_operator", "cpu_reference", "synthetic_test"):
            raise ValueError("run backend must explicitly distinguish scene/operator/reference/test")
        if run.get("state") not in ("queued", "running", "completed", "failed", "blocked", "interrupted"):
            raise ValueError("invalid backend run state")
        tid = run.get("trajectory_id", "")
        if not re.fullmatch(r"trajectory-[a-f0-9]{32}", tid):
            raise ValueError("invalid run trajectory_id")
        traj_path = self._file(case_dir, "trajectories/" + tid + "/trajectory.json")
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        if traj.get("case_id") != case_id or traj.get("trajectory_id") != tid or traj.get("input_sha256") != case["input_sha256"]:
            raise ValueError("trajectory belongs to another case or input version")
        if traj.get("gun_profile_sha256") != case["gun_profile_sha256"] or not np.allclose(
                traj["world_transform"], case["planning_to_world"], atol=1e-10):
            raise ValueError("trajectory gun profile or world transform differs from case")
        policy = case.get("evaluation_policy")
        if policy is None:
            if run["state"] == "completed":
                raise ValueError("case has no frozen evaluation policy; prepare a new case with a configured policy")
            if case.get("evaluation_policy_sha256") is not None:
                raise ValueError("case evaluation policy hash changed")
        else:
            policy = freeze_policy(policy)
            if policy_hash(policy) != case.get("evaluation_policy_sha256"):
                raise ValueError("case evaluation policy hash changed")
        # Status/error inspection performs no evaluation. In particular, native
        # preflight can fail before a UV origin has been verified or after the
        # originally pinned evaluator environment is no longer active.
        if run["state"] == "completed" and case.get("evaluation_implementation") != current_evaluator():
            raise ValueError("frozen evaluation implementation differs; use its original code/environment or prepare a new case")
        files = {"case_manifest_sha256": self._file(case_dir, "case.json"),
                 "input_mesh_sha256": self._file(case_dir, "mesh.npz"),
                 "input_summary_sha256": self._file(case_dir, "summary.json"),
                 "gun_profile_sha256": self._file(case_dir, "gun_profile.json"),
                 "face_atlas_sha256": self._file(case_dir, "face_atlas.npz"),
                 "trajectory_sha256": traj_path}
        for name, path in files.items():
            actual = digest(path)
            if actual != run.get(name):
                raise ValueError(f"run {name} does not match current immutable artifact")
        if run["input_mesh_sha256"] != case["input_sha256"]["mesh"] or run["gun_profile_sha256"] != case["gun_profile_sha256"]:
            raise ValueError("case source mesh or gun profile changed")
        if run["input_summary_sha256"] != case["input_sha256"]["summary"]:
            raise ValueError("case input summary changed")
        if run["face_atlas_sha256"] != case["artifacts"]["face_atlas.npz"]["sha256"]:
            raise ValueError("case UV atlas changed")
        if run.get("evaluation_policy_sha256") != case["evaluation_policy_sha256"]:
            raise ValueError("run evaluation policy differs from frozen case policy")
        profile = json.loads(files["gun_profile_sha256"].read_text(encoding="utf-8-sig"))
        base = {"case_id": case_id, "run_id": run_id, "trajectory_id": tid,
                "gun_profile": {key: profile.get(key) for key in
                    ("profile_id", "version", "data_origin", "mechanism_family", "status",
                     "hardware_identity", "calibrated", "output_unit", "eps_native_scene_equivalence_verified")},
                "state": run["state"], "deposition_backend": run["backend"],
                "backend_reported_reason": run.get("reason"),
                "backend_reported_error_code": run.get("error_code"),
                "backend_reported_effect_status": run.get("effect_status"),
                "backend_reported_native_control_attempted": run.get("native_control_attempted"),
                "backend_reported_automatic_retry": run.get("automatic_retry"),
                "backend_reported_times": {key: run.get(key) for key in
                                           ("created_at", "updated_at", "deadline_utc", "recovered_at")},
                "execution_claims_from_backend": run.get("execution_validation"),
                "execution_verified_by_result_reader": False,
                "reconstruction_identity": ({key: run.get(key) for key in
                    ("implementation", "implementation_sha256", "kernel_sha256", "numpy_version", "configuration_sha256")}
                    if run["backend"] == "cpu_reference" else None),
                "evaluation_policy_sha256": case["evaluation_policy_sha256"],
                "evaluation_implementation": case["evaluation_implementation"],
                "planned_duration_s": traj["duration_s"],
                "parent_trajectory_id": traj.get("parent_trajectory_id"),
                "run_manifest_sha256": digest(manifest_path)}
        if run["state"] != "completed":
            return {**base, "raw_result_available": False, "metrics": None}
        raw = self._file(run_dir, "raw_deposition.npz")
        if raw.stat().st_size > 128 * 1024**2 or digest(raw) != run.get("raw_sha256"):
            raise ValueError("raw deposition size/hash mismatch")
        evidence = []
        for item in run.get("execution_evidence", []):
            path = self._file(run_dir, item["relative_path"])
            if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
                raise ValueError("execution evidence size/hash mismatch")
            evidence.append({**item, "path": str(path), "integrity_verified": True})
        if run["backend"] == "eps_native_scene" and not any(x["size"] > 0 for x in evidence):
            raise ValueError("native scene run has no non-empty runtime evidence")
        key = policy_hash({"raw": run["raw_sha256"], "atlas": run["face_atlas_sha256"],
                           "trajectory": run["trajectory_sha256"], "evaluator": current_evaluator(),
                           "policy": case["evaluation_policy_sha256"], "run_manifest": base["run_manifest_sha256"]})
        index = run_dir / "evaluations" / (key + ".json")
        if index.is_file():
            cached = json.loads(index.read_text(encoding="utf-8"))
            summary_path = self._file(run_dir, cached["relative_path"])
            if digest(summary_path) != cached["sha256"]:
                raise ValueError("cached metric artifact changed")
            metrics = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            attempt = run_dir / "evaluations" / (key[:16] + "-" + uuid.uuid4().hex)
            inputs = attempt / "inputs"
            inputs.mkdir(parents=True, exist_ok=False)
            try:
                for src, filename, expected in ((raw, "raw_deposition.npz", run["raw_sha256"]),
                                                (files["face_atlas_sha256"], "face_atlas.npz", run["face_atlas_sha256"])):
                    target = inputs / filename
                    with src.open("rb") as a, target.open("xb") as b:
                        shutil.copyfileobj(a, b)
                    if digest(src) != expected or digest(target) != expected:
                        raise ValueError("source changed during immutable result snapshot")
                mesh = load_mesh(case_dir / "mesh.npz", case_dir / "summary.json")
                transform = np.asarray(case["planning_to_world"])
                centroids = mesh["vertices"][mesh["faces"]].mean(axis=1) @ transform[:3, :3].T + transform[:3, 3]
                metrics = evaluate_files(inputs / "raw_deposition.npz", inputs / "face_atlas.npz", attempt / "output",
                    np.asarray(traj["target_scope"]["face_ids"], dtype=np.int64), **policy, centroids_mm=centroids,
                    provenance={"backend": run["backend"], "run_id": run_id,
                                "run_manifest_sha256": base["run_manifest_sha256"],
                                "trajectory_sha256": run["trajectory_sha256"]})
                if digest(manifest_path) != base["run_manifest_sha256"] or any(digest(p) != run[k] for k, p in files.items()):
                    raise ValueError("run or case artifacts changed during result analysis")
                summary_path = attempt / "output/metrics.json"
                temp_index = index.with_name("tmp-" + uuid.uuid4().hex + ".json")
                write_json(temp_index, {"relative_path": summary_path.relative_to(run_dir).as_posix(),
                                        "sha256": digest(summary_path)})
                temp_index.replace(index)
            except Exception as exc:
                write_json(attempt / "failure.json", {"error": str(exc), "run_id": run_id})
                raise
        for name, receipt in metrics["input_files"].items():
            snapshot = Path(receipt["path"])
            if snapshot.is_symlink() or not snapshot.resolve().is_relative_to(run_dir.resolve()):
                raise ValueError("cached input snapshot escapes run directory")
            if not snapshot.is_file() or snapshot.stat().st_size != receipt["size"] or digest(snapshot) != receipt["sha256"]:
                raise ValueError("cached input snapshot changed: " + name)
        face_artifact = Path(metrics["face_metrics_artifact"]["path"])
        if face_artifact.is_symlink() or not face_artifact.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError("per-face result artifact escapes run directory")
        if digest(face_artifact) != metrics["face_metrics_artifact"]["sha256"]:
            raise ValueError("per-face result artifact changed")
        if digest(manifest_path) != base["run_manifest_sha256"]:
            raise ValueError("run manifest changed during result query")
        return {**base, "raw_result_available": True, "array_and_artifact_integrity_verified": True,
                "execution_evidence": evidence, "metrics": metrics,
                "metrics_artifact": {"path": str(summary_path), "size": summary_path.stat().st_size,
                                      "sha256": digest(summary_path)}}
