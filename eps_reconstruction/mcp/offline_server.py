"""Standalone deployment entry point for existing reconstructed workflow tools."""
import logging
import sys
from mcp.server.fastmcp import FastMCP
from workflow_tools import register_tools

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
MCP = FastMCP(
    "eps_reconstruction",
    instructions=(
        "Offline reconstructed deposition, not native EPS execution or physical coating. "
        "Use prepare_case -> generate_trajectory -> run_simulation(backend='cpu_reference') "
        "-> get_result. Configure an absolute asset catalogue and case output directory. "
        "Use gun_mode='off' for a zero-deposition control. Existing cases are immutable. "
        "Relative numerical outputs are not calibrated film thickness. "
        "For registered native scenes, call prepare_native_scene, then run_simulation(backend='eps_native_scene') "
        "and get_result. Native execution requires the isolated EPS worker."
    ),
)
register_tools(MCP)

if __name__ == "__main__":
    MCP.run(transport="stdio")
