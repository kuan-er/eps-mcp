"""Apply an explicit Agent decision to an existing case over real MCP stdio."""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def revise(baseline_path, decision_path, output):
    baseline = json.loads(baseline_path.read_text())
    decision = json.loads(decision_path.read_text())
    case = next(c for c in baseline["cases"] if c["case_id"] == decision["case_id"])
    if not case["passed"] or case["run"]["run_id"] != decision["baseline_run_id"] or case["trajectory_id"] != decision["parent_trajectory_id"]:
        raise ValueError("decision does not identify a completed baseline")
    catalog_path = Path(baseline["catalog"])
    if hashlib.sha256(catalog_path.read_bytes()).hexdigest() != baseline["catalog_sha256"]:
        raise ValueError("baseline catalogue changed")
    catalogue = json.loads(catalog_path.read_text())
    asset = catalogue["assets"][case["asset_id"]]
    output.mkdir(parents=True, exist_ok=False)
    # Preserve the decision exactly; it is data, never executable instructions.
    (output / "agent-decision.json").write_bytes(decision_path.read_bytes())
    args = {"case_id": case["case_id"], "semantic": asset["semantic"], "view_axes": asset["view_axes"],
            **baseline["planning"], **decision["decision"]["changes"],
            "parent_trajectory_id": case["trajectory_id"], "decision": decision["decision"]}
    parameters = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp/server.py")],
        env={**os.environ, "EPS_MCP_ASSET_CATALOG": str(catalog_path), "EPS_MCP_CASE_ROOT": baseline["case_root"]})
    transcript = output / "mcp-transcript.jsonl"

    def record(method, arguments, response):
        with transcript.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "method": method,
                                     "arguments": arguments, "response": response}, ensure_ascii=False) + "\n")

    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=630)) as client:
            init = await client.initialize()
            record("initialize", {}, init.model_dump(mode="json"))

            async def call(name, arguments):
                r = await client.call_tool(name, arguments)
                record(name, arguments, r.model_dump(mode="json"))
                if r.isError:
                    raise RuntimeError(r.content[0].text)
                return r.structuredContent or json.loads(r.content[0].text)

            generated = await call("generate_trajectory", args)
            run = await call("run_simulation", {"case_id": case["case_id"], "trajectory_id": generated["trajectory_id"]})
            result = await call("get_result", {"case_id": case["case_id"], "run_id": run["run_id"], "baseline_run_id": case["run"]["run_id"]})
    report = {"decision": decision, "generation": generated, "run": run, "result": result,
              "agent_decision_was_explicit_input": True, "native_eps_executed": False,
              "feedback_loop_executed": run["state"] == "completed" and result.get("comparison", {}).get("state") == "comparable"}
    (output / "feedback-result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"feedback_loop_executed": report["feedback_loop_executed"], "state": run["state"],
                      "comparison": result.get("comparison"), "reason": run.get("reason")}, ensure_ascii=False))
    return report["feedback_loop_executed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--decision", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    sys.exit(0 if asyncio.run(revise(a.baseline.resolve(strict=True), a.decision.resolve(strict=True), a.output.resolve())) else 1)
