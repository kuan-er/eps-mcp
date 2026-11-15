"""Read-only EFORT PaintStudio MCP server. No robot or controller commands."""
import hashlib
import json
import logging
import os
import threading
import ctypes
from pathlib import Path
from datetime import datetime, timezone

import psutil
from mcp.server.fastmcp import FastMCP
from pywinauto import Desktop

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "identity-and-environment.json"
OUTPUT = ROOT / "screenshots"
LOG = logging.getLogger("eps_mcp")
logging.basicConfig(level=logging.INFO, stream=__import__("sys").stderr)
MCP = FastMCP("eps", instructions="Read-only EPS observations; disk values and live GUI are separate sources.")
UI_LOCK = threading.Lock()


def _session_id(pid):
    result = ctypes.c_ulong()
    if not ctypes.windll.kernel32.ProcessIdToSessionId(pid, ctypes.byref(result)):
        raise ctypes.WinError()
    return result.value


def _file_version(path):
    """Read Windows executable version metadata without trusting the path name."""
    try:
        import win32api
        info = win32api.GetFileVersionInfo(str(path), "\\")
        ms, ls = info["FileVersionMS"], info["FileVersionLS"]
        return ".".join(str(part) for part in
                        (ms >> 16, ms & 0xffff, ls >> 16, ls & 0xffff)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _eps_process():
    candidates = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            if (proc.info["name"] or "").lower() == "efort paintstudio.exe":
                candidates.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one EPS process, found {len(candidates)}")
    return candidates[0]


def _project_path():
    data = json.loads(IDENTITY.read_text(encoding="utf-8-sig"))
    configured = Path(data["eps"]["startup_project"])
    if not configured.is_absolute() or configured.suffix.lower() != ".cmaproj":
        raise ValueError("Configured startup project is not an absolute .cmaproj path")
    path = configured.resolve(strict=True)
    if not path.is_file():
        raise ValueError("Configured startup project is not a file")
    return path


def _window():
    proc = _eps_process()
    windows = Desktop(backend="uia").windows(process=proc.pid, visible_only=True)
    if not windows:
        raise RuntimeError("EPS GUI is inaccessible in this session; no visible UIA window")
    guns = [w for w in windows if w.window_text() == "GunsWindow"]
    if len(guns) != 1:
        raise RuntimeError(f"Expected one live GunsWindow, found {len(guns)}; no disk fallback")
    return proc, guns[0]


@MCP.tool()
def eps_status() -> dict:
    """Read current EPS process, session, executable and visible window titles."""
    proc = _eps_process()
    try:
        with UI_LOCK:
            windows = [w.window_text() for w in Desktop(backend="uia").windows(process=proc.pid, visible_only=True)]
        gui_error = None
    except Exception as exc:
        windows, gui_error = [], f"{type(exc).__name__}: {exc}"
    executable = proc.exe()
    file_version, version_error = _file_version(executable)
    return {"source": "live_process", "read_at": datetime.now(timezone.utc).isoformat(),
            "pid": proc.pid, "session_id": _session_id(proc.pid),
            "executable": executable, "file_version": file_version,
            "file_version_error": version_error, "windows": windows, "gui_error": gui_error,
            "version_basis": "Windows executable metadata; application UI version is separate"}


@MCP.tool()
def eps_disk_project() -> dict:
    """Read saved gun and spray settings from the fixed audited startup project; not live GUI."""
    path = _project_path()
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8-sig"))
    # The project is JSON; return only relevant read-only records, not unrelated project data.
    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                yield node[key]
            for val in node.values():
                yield from find(val, key)
        elif isinstance(node, list):
            for val in node:
                yield from find(val, key)
    guns = [g for group in find(obj, "GunsFeatures") if isinstance(group, list)
            for g in group if isinstance(g, dict) and "GunID" in g]
    spray = list(find(obj, "PaintSpraySettings"))
    return {"source": "disk_saved_startup_project", "active_project_verified": False,
            "path": str(path), "sha256": hashlib.sha256(raw).hexdigest().upper(),
            "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "guns": guns, "paint_spray_settings": spray}


@MCP.tool()
def eps_live_gun() -> dict:
    """Read current GunsWindow UIA controls. Raises an error if live GUI is unavailable."""
    with UI_LOCK:
        proc, window = _window()
        values = []
        for item in window.descendants():
            info = item.element_info
            if info.control_type in ("Text", "Edit", "Button", "ComboBox"):
                value = None
                if info.control_type == "Edit":
                    try:
                        value = item.get_value()
                    except Exception as exc:
                        value = f"READ_ERROR: {type(exc).__name__}: {exc}"
                values.append({"type": info.control_type, "name": info.name,
                               "automation_id": info.automation_id, "value": value})
    if not values:
        raise RuntimeError("GunsWindow UIA returned no controls; no disk fallback")
    return {"source": "live_gui_uia", "read_at": datetime.now(timezone.utc).isoformat(),
            "pid": proc.pid, "window": window.window_text(), "controls": values}


@MCP.tool()
def eps_window_screenshot() -> dict:
    """Capture the current live GunsWindow into the delivery screenshots directory."""
    with UI_LOCK:
        proc, window = _window()
        OUTPUT.mkdir(exist_ok=True)
        path = OUTPUT / f"mcp-guns-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.png"
        image = window.capture_as_image()
        if image is None:
            raise RuntimeError("UIA screenshot capture returned no image")
        image.save(path)
    return {"source": "live_gui_capture", "pid": proc.pid, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(), "bytes": path.stat().st_size}


if __name__ == "__main__":
    MCP.run(transport="stdio")
