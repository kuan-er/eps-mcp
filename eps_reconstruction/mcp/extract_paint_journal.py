"""Read EPS paint journals with the USD bindings bundled with Blender."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from pxr import Sdf, Usd, UsdGeom
import numpy as np


PAINT_ATTRIBUTE = "cma:paintRecording"


def open_recording(path: Path):
    if not path.is_file() or path.suffix.lower() != ".usda":
        raise ValueError("Expected a native EPS .usda recording")
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    if not stage:
        raise ValueError("USD could not open the native recording")
    return stage


def paintable_prims(stage):
    for prim in stage.Traverse():
        attribute = prim.GetAttribute(PAINT_ATTRIBUTE)
        if prim.IsA(UsdGeom.Mesh) and attribute and attribute.HasAuthoredValueOpinion():
            yield prim


def extract(stage, source: Path, prim_path: str, out: Path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError("The requested USD prim is not a mesh")
    attribute = prim.GetAttribute(PAINT_ATTRIBUTE)
    if not attribute or attribute.GetTypeName() != Sdf.ValueTypeNames.String or attribute.GetNumTimeSamples():
        raise ValueError("Expected a single EPS paint journal string")
    encoded = attribute.Get()
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("The selected mesh contains no paint journal")
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with out.open("x", encoding="ascii") as stream:
        stream.write(encoded)
    mesh = UsdGeom.Mesh(prim)
    uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3) or uv.GetInterpolation() != UsdGeom.Tokens.faceVarying:
        raise ValueError("EPS recording must contain triangles with per-corner UV coordinates")
    transform = np.asarray(UsdGeom.XformCache(stage.GetStartTimeCode()).GetLocalToWorldTransform(prim), dtype=float)
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0) or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("EPS recording did not declare metres and Z-up")
    positions = np.asarray(mesh.GetPointsAttr().Get(), dtype=float) @ transform[:3, :3] + transform[3, :3]
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    coordinates = np.asarray(uv.ComputeFlattened(), dtype=float).reshape(-1, 3, 2)
    if len(faces) != len(coordinates):
        raise ValueError("EPS recording UV count differs from its triangle count")
    # Eyeshot exports V inverted. The paint journal uses the source OBJ V.
    coordinates[..., 1] = 1 - coordinates[..., 1]
    np.savez_compressed(out.with_suffix(".mesh.npz"), triangles_mm=positions[faces] * 1000, uv=coordinates)
    return {"source": str(source), "source_sha256": source_hash, "prim_path": str(prim.GetPath()),
            "journal_sha256": hashlib.sha256(encoded.encode("ascii")).hexdigest(), "journal": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--prim")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if bool(args.prim) != bool(args.out):
        parser.error("use both --prim and --out to extract, or omit both to list")
    source = args.recording.resolve(strict=True)
    stage = open_recording(source)
    result = (extract(stage, source, args.prim, args.out) if args.prim else
              {"paintable_prims": [str(prim.GetPath()) for prim in paintable_prims(stage)]})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
