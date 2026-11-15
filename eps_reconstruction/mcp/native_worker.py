"""Launch or reuse the dedicated Windows EPS process for a registered project."""

import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from native_bridge import request
from spray_core import digest


def ensure_worker(worker, project, timeout=70):
    if os.name != "nt":
        raise RuntimeError("The EPS native scene worker needs a Windows desktop")
    executable = Path(worker["executable"]).resolve(strict=True)
    module = Path(worker["module"]).resolve(strict=True)
    module_sha256 = worker["module_sha256"].lower()
    if digest(module).lower() != module_sha256:
        raise RuntimeError("The native worker module file does not match its configured hash")
    root = Path(worker["bridge_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    launch_lock = root / "worker-launch.lock"
    with launch_lock.open("x", encoding="ascii") as lock:
        lock.write(str(os.getpid()))
    try:
        instances = [p for p in psutil.process_iter(["name", "exe"]) if
                     (p.info["name"] or "").lower() == executable.name.lower()]
        if instances:
            if (len(instances) != 1 or not instances[0].info["exe"]
                    or Path(instances[0].info["exe"]).resolve() != executable):
                raise RuntimeError("Another EPS instance is open. Close it before starting the isolated worker")
            process = instances[0]
        else:
            launched = subprocess.Popen([str(executable), str(project)], cwd=executable.parent,
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
            process = psutil.Process(launched.pid)
        args = process.cmdline()
        if len(args) != 2 or Path(args[1]).resolve() != project:
            raise RuntimeError("The isolated EPS worker has a different startup project")
    finally:
        launch_lock.unlink()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process.is_running():
            raise RuntimeError("The isolated EPS worker exited during startup")
        markers = list(root.glob(f"ready-{process.pid}-*.json"))
        if len(markers) == 1:
            try:
                ready = json.loads(markers[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.2)
                continue
            if ready["eps_pid"] != process.pid or Path(ready["project_folder"]).resolve() != project.parent:
                raise RuntimeError("EPS reported an unexpected active project folder")
            scene = request(root, process.pid, "snapshot", 5, module_sha256=module_sha256)
            if (scene.get("simulation_toolbar") or {}).get("SimulationRunning") is False:
                return process.pid
        elif len(markers) > 1:
            raise RuntimeError("Multiple readiness receipts match the EPS process")
        time.sleep(0.6)
    raise TimeoutError("The isolated EPS worker did not reach its ready state")
