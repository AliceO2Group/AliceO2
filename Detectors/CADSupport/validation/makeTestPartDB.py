#!/usr/bin/env python3

# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.
# Author: Sandro Wenzel <sandro.wenzel@cern.ch>
# Since: 2026-07

"""Build a test-part database for the solid-navigation harness.

For each input CAD model it runs O2_CADtoTGeo.py with `--exact-surfaces auto --mesh
--surface-report --dump-brep --csg auto` and indexes, per leaf volume, the paired artefacts into
one `manifest.json`: `surfaces_<VOL>_<LID>.bin`, `facets_<VOL>_<LID>.bin`, `brep_<VOL>_<LID>.brep`
when present, and `shape_<VOL>_<LID>.root` (a TGeoShape under the key "shape", in cm). A part
enters only when both the sidecar and the mesh exist.

Each part's `"shipped"` block copies the converter's cascade decision from `csg_report.json`, so
the gate judges the representation the part ships in; `"decidedBy"` records the source. Leaf
solids with no sidecar are listed under `"unscoredParts"`.

Usage:
  python3 makeTestPartDB.py --output <db-dir>
  python3 makeTestPartDB.py --models ExcavatorArm.step as1-oc-214.stp --output <db-dir> --force

Requires the O2 + pythonOCC environment, as O2_CADtoTGeo.py does.
"""

import argparse
import datetime
import json
import re
import shutil
import struct
import subprocess
import sys
from typing import Optional
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONVERTER = _SCRIPT_DIR.parent / "tools" / "O2_CADtoTGeo.py"
_DEFAULT_MODEL_DIR = _SCRIPT_DIR.parent / "examples"

# Models that ship in examples/.
_DEFAULT_MODELS = [
    "ExcavatorArm.step",
    "as1-oc-214.stp",
]


def _sanitize_filename(s: str) -> str:
    """Mirror of O2_CADtoTGeo.py's sanitize_filename(); keep in sync with that copy."""
    safe = re.sub(r"[^0-9a-zA-Z]", "_", s)
    return safe or "x"


def _slugify_model(model_path: Path) -> str:
    return _sanitize_filename(model_path.stem)


def _resolve_model(model_arg: str) -> Path:
    p = Path(model_arg)
    if not p.is_absolute():
        candidate = _DEFAULT_MODEL_DIR / model_arg
        if candidate.exists():
            p = candidate
        else:
            p = Path(model_arg).expanduser().resolve()
    return p.resolve()


def _read_facets_summary(path: Path):
    """Return (nTriangles, bboxMin, bboxMax) by scanning a facets_*.bin file."""
    with open(path, "rb") as f:
        header = f.read(4)
        (n_tri,) = struct.unpack("<I", header)
        bbox_min = [float("inf")] * 3
        bbox_max = [float("-inf")] * 3
        rec_size = 9 * 4
        for _ in range(n_tri):
            data = f.read(rec_size)
            if len(data) != rec_size:
                raise ValueError(f"{path}: truncated facet record (expected {n_tri} triangles)")
            v = struct.unpack("<9f", data)
            for vi in range(3):
                x, y, z = v[3 * vi], v[3 * vi + 1], v[3 * vi + 2]
                bbox_min[0] = min(bbox_min[0], x)
                bbox_min[1] = min(bbox_min[1], y)
                bbox_min[2] = min(bbox_min[2], z)
                bbox_max[0] = max(bbox_max[0], x)
                bbox_max[1] = max(bbox_max[1], y)
                bbox_max[2] = max(bbox_max[2], z)
    if n_tri == 0:
        bbox_min = [0.0, 0.0, 0.0]
        bbox_max = [0.0, 0.0, 0.0]
    return n_tri, bbox_min, bbox_max


def _convert_model(model_path: Path, out_dir: Path, skip_existing: bool, force: bool,
                   csg_mode: str = "auto", mesh_prec: Optional[str] = None,
                   include_name: Optional[list] = None):
    report_path = out_dir / "surface_report.json"

    if out_dir.exists() and any(out_dir.iterdir()):
        if force:
            shutil.rmtree(out_dir)
        elif skip_existing and report_path.exists():
            print(f"  [skip] {out_dir} already converted (--skip-existing)")
            return json.loads(report_path.read_text()), None
        else:
            raise RuntimeError(
                f"{out_dir} already exists and is non-empty. Pass --skip-existing to reuse it "
                "or --force to regenerate it."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_CONVERTER),
        str(model_path),
        "--exact-surfaces", "auto",
        "--mesh",
        "--dump-brep",
        "--surface-report", str(report_path),
        "--output-folder", str(out_dir),
        "-o", "geom.C",
    ]
    # `--csg auto` builds the converter's full cascade; `off` reproduces the pre-cascade database.
    if csg_mode != "off":
        cmd += ["--csg", csg_mode]
    # Unset keeps the converter's default of 0.1 and the identical command line.
    if mesh_prec is not None:
        cmd += ["--mesh-prec", str(mesh_prec)]
    # Passed straight through; the converter owns the matching rule.
    for pat in (include_name or []):
        cmd += ["--include-name", pat]
    print(f"  running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not report_path.exists():
        raise RuntimeError(f"{model_path}: converter did not produce {report_path}")
    return json.loads(report_path.read_text()), cmd


# The cascade tier under the name `gate.json` gives that representation.
_TIER_TO_REPRESENTATION = {"csg": "shape", "surface": "surface", "mesh": "mesh"}


def _read_cascade(out_dir: Path):
    """Read the converter's own cascade decision, per leaf solid, from `csg_report.json`.

    Returns `(by_suffix, by_lid, meta)`; `meta` is None when the converter ran without `--csg`.
    """
    path = out_dir / "csg_report.json"
    if not path.exists():
        return {}, {}, None
    doc = json.loads(path.read_text())
    by_suffix, by_lid = {}, {}
    for row in doc.get("parts", []):
        entry = dict(row)
        entry["source"] = str(path.resolve())
        if row.get("part"):
            by_suffix[row["part"]] = entry
        if row.get("lid"):
            by_lid[row["lid"]] = entry
    meta = {"path": str(path.resolve()), "tiers": doc.get("tiers", {}),
            "nLeafSolids": doc.get("nLeafSolids")}
    return by_suffix, by_lid, meta


def _shape_placement(suffix: str, lid, cascade_by_suffix: dict, cascade_by_lid: dict):
    """The shape's rigid placement, 3x4 row-major `[R | t]` (local -> part), or None for identity.

    Read from `csg_report.json`; the `TGeoHMatrix` in `shape_<part>.root` is the same transform.
    """
    row = cascade_by_suffix.get(suffix)
    if row is None and lid is not None:
        row = cascade_by_lid.get(lid)
    return None if row is None else row.get("shapePlacement")


def _shipped_entry(suffix: str, lid, cascade_by_suffix: dict, cascade_by_lid: dict,
                   cascade_meta: dict, out_dir: Path):
    """What representation this part ships in, from the converter's cascade decision only."""
    row = cascade_by_suffix.get(suffix)
    if row is None and lid is not None:
        row = cascade_by_lid.get(lid)
    if row is not None:
        tier = row.get("representation", "mesh")
        entry = {
            "representation": _TIER_TO_REPRESENTATION.get(tier, tier),
            "tier": tier,
            "decidedBy": "csg_report.json (converter cascade)",
            "source": row.get("source"),
            "evidence": row.get("evidence", {}),
        }
        if row.get("shapeDeferred"):
            entry["shapeDeferred"] = True
        return entry
    # No cascade report: the converter ran without --csg, so its cascade is the older
    # exact-surfaces -> tessellated one and a part in this database has a sidecar by construction.
    return {
        "representation": "surface" if (out_dir / f"surfaces_{suffix}.bin").exists() else "mesh",
        "tier": "surface" if (out_dir / f"surfaces_{suffix}.bin").exists() else "mesh",
        "decidedBy": "artifact presence (converter ran without --csg)",
        "source": str((out_dir / "surface_report.json").resolve()),
        "evidence": {},
    }


def _unscored_leaf_solids(slug: str, out_dir: Path, report: dict, indexed_suffixes: set,
                          cascade_by_lid: dict):
    """Leaf solids the model has that never enter the part database, such as ExcavatorArm's `Bucket`."""
    missing = []
    for lid, info in report.get("volumes", {}).items():
        name = info.get("name") or ""
        volname = _sanitize_filename(name) if name else "vol"
        suffix = f"{volname}_{_sanitize_filename(lid)}"
        if suffix in indexed_suffixes:
            continue
        row = cascade_by_lid.get(lid, {})
        tier = row.get("representation", "mesh")
        missing.append({
            "id": f"{slug}/{suffix}",
            "volume": name,
            "lid": lid,
            "nFaces": info.get("n_faces"),
            "eligible": info.get("eligible"),
            "shipped": {
                "representation": _TIER_TO_REPRESENTATION.get(tier, tier),
                "tier": tier,
                "decidedBy": ("csg_report.json (converter cascade)" if row
                              else "artifact presence (no exact sidecar was written)"),
                "source": row.get("source", str((out_dir / "surface_report.json").resolve())),
                "evidence": row.get("evidence", {}),
            },
            "reason": "no surfaces_*.bin sidecar: the harness cannot score this part",
            "facets": (str(out_dir / f"facets_{suffix}.bin")
                       if (out_dir / f"facets_{suffix}.bin").exists() else None),
        })
    return missing


def _index_parts(model_name: str, slug: str, out_dir: Path, report: dict):
    """Pair surfaces_*.bin / facets_*.bin by <VOL>_<LID> suffix and read the report's
    (raw lid -> volume name) map to recover the manifest's `volume`/`lid` fields."""
    suffix_to_lid = {}
    for lid, info in report.get("volumes", {}).items():
        name = info.get("name") or ""
        volname = _sanitize_filename(name) if name else "vol"
        lidname = _sanitize_filename(lid)
        suffix_to_lid[f"{volname}_{lidname}"] = (lid, name)

    cascade_by_suffix, cascade_by_lid, cascade_meta = _read_cascade(out_dir)

    parts = []
    warnings = []
    indexed_suffixes = set()
    for surf_path in sorted(out_dir.glob("surfaces_*.bin")):
        suffix = surf_path.name[len("surfaces_"):-len(".bin")]
        facet_path = out_dir / f"facets_{suffix}.bin"
        if not facet_path.exists():
            warnings.append(f"{surf_path.name}: no matching facets_{suffix}.bin, skipped")
            continue
        lid, volname = suffix_to_lid.get(suffix, (None, None))
        if lid is None:
            warnings.append(f"{surf_path.name}: suffix not found in surface_report.json volumes")
        n_tri, bbox_min, bbox_max = _read_facets_summary(facet_path)
        part = {
            "id": f"{slug}/{suffix}",
            "model": model_name,
            "volume": volname,
            "lid": lid,
            "surfaces": str(surf_path),
            "facets": str(facet_path),
            "nTriangles": n_tri,
            "bbox": {"min": bbox_min, "max": bbox_max},
        }
        # OCCT reference solid in cm (converter --dump-brep); absent for older databases.
        brep_path = out_dir / f"brep_{suffix}.brep"
        if brep_path.exists():
            part["brep"] = str(brep_path)
        # A third representation: a TGeoShape under "shape", in cm, with an optional "placement".
        shape_path = out_dir / f"shape_{suffix}.root"
        if shape_path.exists():
            part["shape"] = str(shape_path)
            # Mirrored from the converter's record; absent means identity.
            placement = _shape_placement(suffix, lid, cascade_by_suffix, cascade_by_lid)
            if placement is not None:
                part["shapePlacement"] = placement
        # The representation the converter shipped; the gate judges this one.
        part["shipped"] = _shipped_entry(suffix, lid, cascade_by_suffix, cascade_by_lid,
                                         cascade_meta, out_dir)
        if part["shipped"]["representation"] == "shape" and "shape" not in part:
            warnings.append(f"{surf_path.name}: cascade says CSG but no shape_{suffix}.root exists")
        parts.append(part)
        indexed_suffixes.add(suffix)
    unscored = _unscored_leaf_solids(slug, out_dir, report, indexed_suffixes, cascade_by_lid)
    return parts, warnings, unscored, cascade_meta


def build_db(models, output: Path, skip_existing: bool, force: bool, csg_mode: str = "auto",
             mesh_prec: Optional[str] = None, include_name: Optional[list] = None):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "output_dir": str(output.resolve()),
        "csg_mode": csg_mode,
        "mesh_prec": mesh_prec,
        "include_name": include_name,
        "models": [],
        "parts": [],
        # Leaf solids this database cannot hold; read by runOracleGate.py, ignored by the harness.
        "unscoredParts": [],
    }

    for model_arg in models:
        model_path = _resolve_model(model_arg)
        if not model_path.exists():
            raise RuntimeError(f"Model not found: {model_arg} (resolved to {model_path})")
        slug = _slugify_model(model_path)
        out_dir = output / slug
        print(f"[{slug}] {model_path}")

        report, cmd = _convert_model(model_path, out_dir, skip_existing, force, csg_mode,
                                     mesh_prec, include_name)
        parts, warnings, unscored, cascade_meta = _index_parts(
            model_path.name, slug, out_dir, report)
        for w in warnings:
            print(f"  [warn] {w}")

        summary = report.get("summary", {})
        model_entry = {
            "model": model_path.name,
            "model_path": str(model_path),
            "slug": slug,
            "output_dir": str(out_dir.resolve()),
            "command": cmd,
            "surface_report": str((out_dir / "surface_report.json").resolve()),
            "n_volumes": summary.get("n_volumes"),
            "n_eligible": summary.get("n_eligible"),
            "n_paired": len(parts),
            "n_unscored": len(unscored),
            "warnings": warnings,
        }
        if cascade_meta:
            model_entry["csg_report"] = cascade_meta["path"]
            model_entry["cascade_tiers"] = cascade_meta["tiers"]
            model_entry["n_leaf_solids"] = cascade_meta["nLeafSolids"]
        manifest["models"].append(model_entry)
        manifest["parts"].extend(parts)
        manifest["unscoredParts"].extend(unscored)
        tier_counts = {}
        for part in parts:
            key = part["shipped"]["representation"]
            tier_counts[key] = tier_counts.get(key, 0) + 1
        print(f"  -> {len(parts)} parts paired (of {summary.get('n_eligible')} exact-eligible / "
              f"{summary.get('n_volumes')} total volumes)")
        print(f"     shipped representation: "
              + ", ".join(f"{k}={v}" for k, v in sorted(tier_counts.items()))
              + (f"; {len(unscored)} leaf solid(s) not scoreable" if unscored else ""))

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"\nWrote {manifest_path} ({len(manifest['parts'])} parts across {len(manifest['models'])} models)")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=_DEFAULT_MODELS,
                    help="CAD model files (relative names are resolved against "
                         f"{_DEFAULT_MODEL_DIR}). Default: the models that ship in examples/.")
    ap.add_argument("--output", default=str(_SCRIPT_DIR / "test_part_db"),
                    help="Database output directory (default: %(default)s)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Reuse a model's output directory if already converted, re-indexing only.")
    ap.add_argument("--force", action="store_true",
                    help="Delete and regenerate a model's output directory even if it exists.")
    ap.add_argument("--csg", default="auto", choices=["off", "auto", "required"],
                    help="Converter CSG mode (default: %(default)s). 'auto' runs the production "
                         "cascade CSG -> exact surfaces -> tessellated and records the per-part "
                         "choice in csg_report.json, which is what the gate reads to decide which "
                         "representation each part's verdict is computed on. 'off' reproduces the "
                         "pre-cascade database.")
    ap.add_argument("--include-name", action="append", default=None,
                    help="Passed to the converter: only convert CAD labels matching this regex. "
                         "May be repeated. Lets a database be built for one part of a module.")
    ap.add_argument("--mesh-prec", default=None,
                    help="Meshing precision handed to the converter. Unset (default) means the "
                         "converter's own 0.1, which is what every database built before this "
                         "argument existed used, so an existing gate result does not move. Set it "
                         "for a model 0.1 is not safe on -- ALICE3 IRIS needs 0.25.")
    args = ap.parse_args()

    build_db(args.models, Path(args.output), args.skip_existing, args.force, args.csg,
             args.mesh_prec, args.include_name)


if __name__ == "__main__":
    main()
