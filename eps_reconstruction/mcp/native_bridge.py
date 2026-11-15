"""Local client for the isolated EPS recording module's file protocol."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid


def request(root: Path, pid: int, operation: str, timeout: float, *, solution_path: Path | None = None,
            module_sha256: str | None = None) -> dict:
    root = root.resolve(strict=True)
    markers = list(root.glob(f"loaded-{pid}-*.json"))
    if len(markers) != 1:
        raise RuntimeError("Expected exactly one module constructor receipt for the launched EPS process")
    marker = json.loads(markers[0].read_text(encoding="utf-8"))
    if marker["eps_pid"] != pid or marker["module_version"] != "2.2-recording-adapter":
        raise RuntimeError("EPS process or recording module version does not match")
    if module_sha256 is not None and marker["dll_sha256"].lower() != module_sha256.lower():
        raise RuntimeError("The loaded native EPS module differs from the configured binary")

    nonce = uuid.uuid4().hex
    response_prefix = {"snapshot": "snapshot", "load_solution": "load", "arm_recording": "arm",
                       "start_scene": "play", "save_usda": "save"}[operation]
    response = root / f"{response_prefix}-{nonce}.json"
    lock = root / "request.lock"
    temporary = root / f"request-{nonce}.tmp"
    request_path = root / "request.json"
    with lock.open("x", encoding="ascii") as owner:
        owner.write(f"{os.getpid()} {nonce}\n")
    resolved = False
    sent = False
    try:
        if request_path.exists():
            raise RuntimeError("A previous module request is still present")
        message = {"Nonce": nonce, "Operation": operation, "CreatedUtc": datetime.now(timezone.utc).isoformat()}
        if operation == "load_solution":
            if solution_path is None or not solution_path.is_absolute():
                raise ValueError("load_solution needs the registered absolute solution path")
            message["SolutionPath"] = str(solution_path)
        temporary.write_text(json.dumps(message), encoding="utf-8")
        temporary.rename(request_path)
        sent = True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if response.exists():
                try:
                    result = json.loads(response.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.05)
                    continue
                if result["eps_pid"] != pid or result["module_instance"] != marker["module_instance"]:
                    raise RuntimeError("Response came from a different EPS process or module instance")
                resolved = True
                return result
            time.sleep(0.05)
        raise TimeoutError(f"EPS did not confirm {operation}; inspect request {nonce} before retrying")
    finally:
        temporary.unlink(missing_ok=True)
        if resolved:
            request_path.unlink()
        if resolved or not sent:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("operation", choices=("snapshot", "load_solution", "arm_recording", "start_scene", "save_usda"))
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if not 0 < args.timeout <= 30:
        parser.error("timeout must be between 0 and 30 seconds")
    print(json.dumps(request(args.root, args.pid, args.operation, args.timeout, solution_path=args.solution), ensure_ascii=False))
