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
# Since: 2026-08

"""Per-part decline catalogue: part x {ships-as, whyNotCSG, whyNotSurface}.

Joins csg_report.json and surface_report.json of converter output directories into
decline_reasons.json. `--source NAME=FILE` records which CAD file a run was converted from.

Usage
-----
  decline_catalogue.py --run ExcavatorArm=/path/to/converted/ExcavatorArm \
                       --run ALICE3=/path/to/converted/ALICE3 \
                       [--source ALICE3=/path/to/CAD_noETA.stp] \
                       [--gate-db /path/to/gate/workdir/db] \
                       --out decline_reasons.json [--markdown]

`--gate-db` adds every model subdirectory of a gate database (each holds the two reports) as a
run named after the subdirectory.
"""

import argparse
import json
import sys
from pathlib import Path


def load_run(name, out_dir):
    out_dir = Path(out_dir)
    csg_path = out_dir / "csg_report.json"
    surf_path = out_dir / "surface_report.json"
    if not csg_path.exists():
        raise SystemExit(f"{name}: {csg_path} does not exist (convert with --csg auto)")
    if not surf_path.exists():
        raise SystemExit(f"{name}: {surf_path} does not exist (convert with --surface-report)")
    csg = json.loads(csg_path.read_text())
    surf = json.loads(surf_path.read_text())
    volumes = surf.get("volumes", {})
    rows = []
    for part in csg.get("parts", []):
        lid = part.get("lid")
        vol = volumes.get(lid, {})
        why_not_csg = part.get("whyNotCSG")
        rows.append({
            "name": part.get("volume") or lid,
            "model": name,
            "lid": lid,
            "shipsAs": part.get("representation"),
            "whyNotCSG": why_not_csg,
            "whyNotSurface": vol.get("why_not_surface"),
            "nFaces": vol.get("n_faces"),
        })
    # A leaf solid can, in principle, appear in the surface report only (it never reached the
    # CSG hook). Keep it, so the catalogue counts every part the converter saw.
    seen = {r["lid"] for r in rows}
    for lid, vol in volumes.items():
        if lid in seen:
            continue
        rows.append({
            "name": vol.get("name") or lid,
            "model": name,
            "lid": lid,
            "shipsAs": "surface" if vol.get("emitted") else "mesh",
            "whyNotCSG": "not assessed (part never reached the CSG hook)",
            "whyNotSurface": vol.get("why_not_surface"),
            "nFaces": vol.get("n_faces"),
        })
    return rows


def markdown(rows):
    out = ["| model | part | faces | ships as | why not CSG | why not SurfaceSolid |",
           "| --- | --- | ---: | --- | --- | --- |"]
    for r in rows:
        why_csg = r["whyNotCSG"] or "—"
        why_surf = r["whyNotSurface"] or "—"
        out.append(f"| {r['model']} | `{r['name']}` | {r['nFaces'] if r['nFaces'] is not None else '?'} "
                   f"| **{r['shipsAs']}** | {why_csg} | {why_surf} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="NAME=DIR",
                    help="a converter output directory holding csg_report.json and "
                         "surface_report.json; repeatable")
    ap.add_argument("--source", action="append", default=[], metavar="NAME=FILE",
                    help="the CAD file a run was converted from, recorded in the output's "
                         "sourceModel map; repeatable")
    ap.add_argument("--gate-db", type=Path,
                    help="a gate workdir's db/ directory: every model subdirectory becomes a run")
    ap.add_argument("--out", type=Path, help="write decline_reasons.json here")
    ap.add_argument("--markdown", action="store_true", help="print the table as markdown")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        name, _, folder = spec.partition("=")
        if not folder:
            ap.error(f"--run wants NAME=DIR, got {spec!r}")
        runs.append((name, folder))
    sources = {}
    for spec in args.source:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--source wants NAME=FILE, got {spec!r}")
        sources[name] = path
    if args.gate_db:
        for sub in sorted(args.gate_db.iterdir()):
            if sub.is_dir() and (sub / "csg_report.json").exists():
                runs.append((sub.name, sub))
    if not runs:
        ap.error("give --run and/or --gate-db")
    unknown = sorted(set(sources) - {name for name, _ in runs})
    if unknown:
        ap.error(f"--source names no such run: {', '.join(unknown)}")

    rows = []
    for name, folder in runs:
        rows.extend(load_run(name, folder))

    n_csg = sum(1 for r in rows if r["shipsAs"] == "csg")
    n_surface = sum(1 for r in rows if r["shipsAs"] == "surface")
    n_mesh = sum(1 for r in rows if r["shipsAs"] == "mesh")
    missing = [r for r in rows
               if (r["shipsAs"] != "csg" and not r["whyNotCSG"])
               or (r["shipsAs"] == "mesh" and not r["whyNotSurface"])]
    print(f"{len(rows)} part(s) over {len(runs)} run(s): "
          f"csg {n_csg}, surface {n_surface}, mesh {n_mesh}; "
          f"{len(missing)} row(s) with a missing decline reason")
    for r in missing:
        print(f"  [missing] {r['model']}/{r['name']}: shipsAs={r['shipsAs']} "
              f"whyNotCSG={r['whyNotCSG']!r} whyNotSurface={r['whyNotSurface']!r}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # `null` for a run nobody named a source for: an absent statement, never a guess.
        source_map = {name: sources.get(name) for name, _ in runs}
        args.out.write_text(json.dumps({"sourceModel": source_map, "parts": rows}, indent=1))
        print(f"Wrote {args.out}")
    if args.markdown:
        print(markdown(rows))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
