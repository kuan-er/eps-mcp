"""Run a loaded, isolated native EPS scene and retain its OpenUSD evidence."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import psutil

from native_bridge import request
from recorded_deposition import reconstruct


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files_under_render(project):
    render = project.parent / "Render"
    return {path.resolve() for path in render.glob(f"*/{project.stem}.usda")
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(render.resolve())}


def run_export(*, project, solution, executable, blender, bridge_root, pid, piece, program, prim_path,
               output, before_control, module_sha256, timeout=85):
    """Drive the EPS module and publish the original export plus its integer increment."""
    project, solution, executable, blender, bridge_root = (Path(p).resolve(strict=True) for p in
                                                  (project, solution, executable, blender, bridge_root))
    output = Path(output).resolve(strict=True)
    process = psutil.Process(pid)
    if Path(process.exe()).resolve() != executable or len(process.cmdline()) != 2 or (
            Path(process.cmdline()[1]).resolve() != project):
        raise RuntimeError("Native EPS is not the configured isolated process and project")
    process_started = process.create_time()
    deadline = time.monotonic() + timeout

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("EPS native execution and export timed out")
        return left

    def call(operation, filename, **args):
        if not process.is_running() or psutil.Process(pid).create_time() != process_started:
            raise RuntimeError("The isolated EPS process exited or restarted")
        value = request(bridge_root, pid, operation, min(remaining(), 10), module_sha256=module_sha256, **args)
        with (output / filename).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        return value

    lock = bridge_root / "scene-run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"eps_pid": pid, "output": str(output), "started": datetime.now(timezone.utc).isoformat()}, stream)
    attempted = completed = False
    try:
        scene = call("snapshot", "native-worker.json")
        toolbar = scene.get("simulation_toolbar") or {}
        flags = (scene.get("recording_api") or {}).get("flags") or {}
        if (Path(scene["project_folder"]).resolve() != project.parent or toolbar.get("SimulationRunning") is not False
                or not flags.get("CanToggle") or flags.get("IsEnabled") is not False or scene.get("open_windows")):
            raise RuntimeError("The isolated EPS worker needs an idle project and an available OpenUSD recorder")
        before_control()
        attempted = True
        loaded = call("load_solution", "native-load.json", solution_path=solution)
        if loaded["state"] != "loaded":
            raise RuntimeError("EPS did not load the registered solution: " + str(loaded.get("error") or loaded.get("reason")))
        scene = call("snapshot", "native-preflight.json")
        toolbar = scene.get("simulation_toolbar") or {}
        flags = (scene.get("recording_api") or {}).get("flags") or {}
        if (Path(scene["project_folder"]).resolve() != project.parent or not scene.get("host_is_ready")
                or not scene.get("solution_path") or Path(scene["solution_path"]).resolve() != solution
                or scene.get("solution_piece_names") != [piece]
                or Path(scene.get("editor_program_name") or "").stem != Path(program).stem
                or [item["program"] for item in scene["solution_programs"]] != [program]
                or not toolbar.get("PaintSimulationEnabled") or toolbar.get("SimulationRunning") is not False
                or not (scene.get("public_native_state") or {}).get("paint_runtime_available")
                or not flags.get("CanToggle") or flags.get("IsEnabled") is not False
                or scene.get("open_windows")):
            raise RuntimeError("The expected isolated workpiece and program must be loaded with an idle EPS recorder")
        existing = files_under_render(project)
        if call("arm_recording", "native-arm.json")["state"] != "armed":
            raise RuntimeError("The native OpenUSD recorder did not arm")
        if call("start_scene", "native-play.json")["state"] != "requested":
            raise RuntimeError("EPS did not accept Play")
        for i in range(1000):
            scene = call("snapshot", f"native-simulation-{i:03}.json")
            windows = scene.get("open_windows") or []
            if any(window["type"] == "Olpp.Infrastructure.OpenUSD.RecordingExportWindow" for window in windows):
                break
            if windows:
                raise RuntimeError("An unexpected native EPS dialog interrupted simulation")
            time.sleep(min(remaining(), 0.4))
        else:
            raise TimeoutError("EPS did not present a recording export")

        if call("save_usda", "native-save.json")["state"] != "requested":
            raise RuntimeError("EPS did not accept OpenUSD Save")
        for i in range(1000):
            scene = call("snapshot", f"native-export-{i:03}.json")
            flags = (scene.get("recording_api") or {}).get("flags") or {}
            created = files_under_render(project) - existing
            if len(created) > 1:
                raise RuntimeError("Multiple new recordings appeared in the isolated project")
            if created and flags.get("IsBusy") is False and not scene.get("open_windows"):
                source = created.pop()
                break
            time.sleep(min(remaining(), 0.4))
        else:
            raise TimeoutError("EPS did not finish the OpenUSD export")

        recording = output / "native-recording.usda"
        shutil.copyfile(source, recording)
        if digest(recording) != digest(source):
            raise RuntimeError("EPS recording changed during snapshot")
        journal = output / "native-paint-journal.b64"
        extract = Path(__file__).with_name("extract_paint_journal.py")
        with (output / "native-extract.log").open("xb") as log:
            extraction = subprocess.run([str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                                         "--python", str(extract), "--", str(recording), "--prim", prim_path,
                                         "--out", str(journal)], cwd=output, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=min(remaining(), 25),
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        extraction.check_returncode()
        decoded = reconstruct(journal.read_text(encoding="ascii"))
        raw = decoded["increment"]
        if raw is None:
            raise ValueError("The native recording did not preserve a continuous deposition history")
        np.savez_compressed(output / "raw_deposition.npz", values=raw)
        result = {**decoded["summary"], "recording_sha256": digest(recording), "prim_path": prim_path,
                  "raw_sha256": digest(output / "raw_deposition.npz"), "nonzero_texels": int(np.count_nonzero(raw)),
                  "increment_sum": int(raw.sum()), "max_increment": int(raw.max())}
        (output / "native-deposition.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        completed = True
        return result
    finally:
        if completed or not attempted:
            lock.unlink()
