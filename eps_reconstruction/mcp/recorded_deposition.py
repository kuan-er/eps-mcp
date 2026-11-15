"""Decode EPS schema-1 recorded integer fields; this does not simulate spraying.

Input is the exact cma:paintRecording attribute extracted by the native adapter.
Source identity, runtime execution and UV origin need separate native evidence.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from spray_core import digest, write_json


VERSION = "eps-recorded-deposition/1"
MAX_JSON_BYTES = 128 * 1024**2
MAX_PIXELS = 4096**2
MAX_FRAMES = 100000
INT64_MAX = np.iinfo(np.int64).max


def _integers(values, name, *, upper=INT64_MAX):
    if not isinstance(values, list) or any(type(x) is not int or not 0 <= x <= upper for x in values):
        raise ValueError(name + " must be nonnegative bounded JSON integers")
    return np.asarray(values, dtype=np.int64)


def reconstruct(encoded):
    """Return baseline/final fields and a delta only across continuous history.

    An identity reset (e.g. shader snapshot) is harmless. Any different full reset
    or texture shape change invalidates this-run delta, even if final>=baseline.
    """
    if not isinstance(encoded, str) or len(encoded) > MAX_JSON_BYTES * 2:
        raise ValueError("encoded journal is missing or too large")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            original_json = stream.read(MAX_JSON_BYTES + 1)
        if len(original_json) > MAX_JSON_BYTES:
            raise ValueError("expanded journal exceeds byte limit")
        journal = json.loads(original_json)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid encoded paint journal") from exc
    if not isinstance(journal, dict) or type(journal.get("schemaVersion")) is not int or journal["schemaVersion"] != 1:
        raise ValueError("unsupported paint journal schema")
    frames = journal.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError("paint journal needs a bounded nonempty frame list")
    raw = baseline = None
    previous_time = -1.
    previous_shape = None
    resets, discontinuities = [], []
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("invalid frame")
        time = frame.get("timeCode")
        width, height = frame.get("width"), frame.get("height")
        try:
            finite_time = type(time) in (int, float) and math.isfinite(time)
        except OverflowError:
            finite_time = False
        if (not finite_time or time < 0 or time < previous_time
                or type(width) is not int or type(height) is not int or min(width, height) <= 0
                or width * height > MAX_PIXELS or type(frame.get("reset")) is not bool):
            raise ValueError("invalid frame time, dimensions or reset flag")
        shape = (height, width)
        values = _integers(frame.get("values"), "values")
        if frame["reset"]:
            if values.size != width * height:
                raise ValueError("reset snapshot size differs from dimensions")
            resets.append(i)
            if i and (shape != previous_shape or not np.array_equal(raw, values)):
                discontinuities.append(i)
            raw = values.copy()
            if i == 0:
                baseline = raw.reshape(shape).copy()
        else:
            if raw is None:
                raise ValueError("paint journal has no initial reset baseline")
            if shape != previous_shape:
                raise ValueError("sparse frame changed texture shape")
            indices = _integers(frame.get("indices"), "indices", upper=raw.size - 1)
            if indices.size != values.size:
                raise ValueError("sparse indices and values differ in length")
            # Conservative positive-sum bound also covers repeated indices.
            if int(raw.max(initial=0)) + sum(frame["values"]) > INT64_MAX:
                raise ValueError("recorded accumulation could overflow int64")
            np.add.at(raw, indices, values)
        previous_time, previous_shape = time, shape
    final = raw.reshape(previous_shape).copy()
    continuous = not discontinuities and baseline.shape == final.shape
    increment = final - baseline if continuous else None
    if increment is not None and (increment < 0).any():
        raise ValueError("negative recorded increment")
    summary = {"version": VERSION, "source_encoded_sha256": hashlib.sha256(encoded.encode("ascii")).hexdigest(),
               "source_compressed_sha256": hashlib.sha256(compressed).hexdigest(),
               "source_json_sha256": hashlib.sha256(original_json).hexdigest(),
               "frames": len(frames), "first_time_code": frames[0]["timeCode"], "last_time_code": previous_time,
               "time_codes_are_not_simulation_seconds": True, "reset_frame_indices": resets,
               "discontinuous_reset_frame_indices": discontinuities, "increment_available": continuous,
               "baseline_shape": list(baseline.shape), "final_shape": list(final.shape),
               "baseline_all_zero": not bool(np.any(baseline)), "reconstructed_dtype": "int64",
               "original_native_buffer_dtype": None, "original_native_buffer_stride": None,
               "value_semantics": "recorded integer field; display smoothing/clamping not applied",
               "native_execution_verified_by_decoder": False, "source_identity_verified_by_decoder": False}
    return {"baseline": baseline, "final": final, "increment": increment,
            "summary": summary, "original_json": original_json}


def export_increment(encoded, output_dir, *, row_origin, provenance=None):
    """Native-adapter helper; preserve source/failures and never overwrite a run.

    row_origin must come from the native adapter's verified asymmetric probe.
    This decoder records it as a caller claim, not as independent verification.
    """
    if row_origin not in ("uv_v0", "uv_v1"):
        raise ValueError("explicit verified-by-adapter row_origin required")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        source = directory / "source_journal.b64"
        source.write_text(encoded, encoding="ascii")
        data = reconstruct(encoded)
        (directory / "source_journal.json").write_bytes(data["original_json"])
        np.savez_compressed(directory / "recorded_fields.npz", baseline=data["baseline"], final=data["final"])
        write_json(directory / "journal_summary.json", data["summary"])
        if data["increment"] is None:
            raise ValueError("recording reset or texture change prevents a continuous this-run increment")
        raw = directory / "raw_deposition.npz"
        np.savez_compressed(raw, values=data["increment"])
        result = {**data["summary"], "output_semantics": "final_minus_recording_initial_baseline",
                  "row_origin_provided_by_caller": row_origin, "row_origin_verified_by_decoder": False,
                  "provenance_provided_by_caller": provenance,
                  "calibrated_physical_thickness": False,
                  "artifacts": {p.name: {"path": str(p.resolve()), "size": p.stat().st_size, "sha256": digest(p)}
                                for p in sorted(directory.iterdir()) if p.is_file()}}
        write_json(directory / "recording_receipt.json", result)
        return result
    except Exception as exc:
        write_json(directory / "failure.json", {"error": str(exc), "native_execution_verified_by_decoder": False})
        raise
