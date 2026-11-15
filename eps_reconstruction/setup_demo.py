"""Create a small, permanent demo catalogue for this isolated release."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
demo = root / "demo"
demo.mkdir(exist_ok=True)
for name in ("mesh.npz", "summary.json", "catalog.json"):
    if (demo / name).exists():
        raise FileExistsError("demo already exists; do not overwrite frozen inputs: " + name)
np.savez(demo / "mesh.npz",
         vertices=np.array([[-.05, -.05, 0], [.05, -.05, 0], [.05, .05, 0], [-.05, .05, 0]]),
         faces=np.array([[0, 1, 2], [0, 2, 3]]),
         part_id=np.ones(2, dtype=int), spray_target=np.ones(2, dtype=bool))
(demo / "summary.json").write_text(json.dumps({"part_legend": {"1": "top"}}), encoding="utf-8")
catalog = {
    "assets": {"demo_plane": {"mesh": str(demo / "mesh.npz"),
                              "summary": str(demo / "summary.json"), "units": "m", "up_axis": "+Z"}},
    "profiles": {p.stem: str(p) for p in sorted((root / "profiles").glob("*.json"))},
    "evaluation_policy": {"unit": "relative", "row_origin": "uv_v0", "samples_per_face": 32, "seed": 9173},
}
(demo / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
print(json.dumps({"catalog": str(demo / "catalog.json"), "case_root": str(root / "runs"),
                  "asset_id": "demo_plane", "semantic": "top", "native_eps_executed": False}))
