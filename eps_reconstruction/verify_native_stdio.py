"""Verify a native EPS run through the MCP entry point and result reader."""

import argparse
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tomllib

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def verify(output, scene_id, repeats, config):
    if config:
        with config.open("rb") as stream:
            registered = tomllib.load(stream)["mcp_servers"]["eps_reconstruction"]
        parameters = StdioServerParameters(command=registered["command"], args=registered["args"],
                                           env={**os.environ, **registered.get("env", {})})
    else:
        parameters = StdioServerParameters(command=sys.executable,
            args=[str(Path(__file__).resolve().parent / "mcp" / "offline_server.py")], env=os.environ.copy())
    with output.open("x", encoding="utf-8") as log:
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=150)) as client:
                await client.initialize()
                print("Native MCP initialized", flush=True)

                async def call(name, arguments):
                    result = await client.call_tool(name, arguments)
                    log.write(json.dumps({"method": name, "arguments": arguments,
                                          "result": result.model_dump(mode="json")}) + "\n")
                    log.flush()
                    if result.isError:
                        raise RuntimeError(result.content[0].text)
                    return result.structuredContent or json.loads(result.content[0].text)

                if scene_id == "cpu_demo":
                    binding = await call("prepare_case", {"asset_id": "demo_plane"})
                    trajectory = await call("generate_trajectory", {"case_id": binding["case_id"],
                        "semantic": "top", "view_axes": [[0, 0, 1]], "spacing_mm": 50})
                    binding["trajectory_id"] = trajectory["trajectory_id"]
                    backend = "cpu_reference"
                else:
                    binding = await call("prepare_native_scene", {"scene_id": scene_id})
                    backend = "eps_native_scene"
                baseline = None
                for _ in range(repeats):
                    run = await call("run_simulation", {"case_id": binding["case_id"],
                        "trajectory_id": binding["trajectory_id"], "backend": backend})
                    args = {"case_id": binding["case_id"], "run_id": run["run_id"]}
                    if baseline:
                        args["baseline_run_id"] = baseline
                    result = await call("get_result", args)
                    print(json.dumps({"run": run, "reason": result.get("backend_reported_reason"),
                        "verified": result.get("array_and_artifact_integrity_verified"),
                        "metrics_artifact": result.get("metrics_artifact"),
                        "comparison": result.get("comparison")}), flush=True)
                    if run["state"] != "completed" or not result.get("array_and_artifact_integrity_verified"):
                        raise RuntimeError("The native run or result validation did not complete")
                    baseline = run["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="native_spray_target")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    asyncio.run(verify(args.output, args.scene, args.repeats, args.config))
