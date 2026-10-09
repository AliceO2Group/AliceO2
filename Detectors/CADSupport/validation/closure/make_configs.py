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

"""Write the o2-sim configuration that runs the round-tripped geometry.

PIPE, TPC and MAG go in as o2::passive::ExternalModule and ITS as o2::ext::ExternalDetector on the
ITS DetID slot. Each piece is anchored where roundtrip_module.py found it. The external names
are not the real module names, so the native modules are not built as well.

Usage:
  make_configs.py <studydir> [--name CADCLOSURE]
"""

import argparse
import json
import os
import sys

# module -> (external name, kind); ITS is the only sensitive one.
MODULES = [
    ("PIPE", "CPIPE", "passive", "CAD round-tripped beam pipe"),
    ("TPC",  "CTPC",  "passive", "CAD round-tripped TPC (material only)"),
    ("MAG",  "CMAG",  "passive", "CAD round-tripped L3 magnet"),
    ("ITS",  "CITS",  "sensitive", "CAD round-tripped ITS"),
]

SENSITIVE_VOLUMES = {"CITS": ["ITSUSensor"]}   # substring match: ITSUSensor0..6
DET_ID = {"CITS": "ITS"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("studydir")
    p.add_argument("--name", default="CADCLOSURE",
                   help="the detector-list key o2-sim is pointed at")
    p.add_argument("--variant", default="csg", choices=("csg", "mesh"),
                   help="which back-conversion to configure: the shipped cascade "
                        "(csg) or tessellated-only (mesh), the fallback every other "
                        "CAD pipeline uses and the benchmark for the exact path")
    p.add_argument("--out-prefix", default="",
                   help="prefix for the written JSON file names, so two variants "
                        "can live side by side in one study directory")
    args = p.parse_args()

    study = os.path.abspath(args.studydir)
    modules, detectors, names, missing = [], [], [], []

    for mod, name, kind, title in MODULES:
        frag = os.path.join(study, "cad", mod, "module_entries.json")
        if not os.path.exists(frag):
            missing.append(frag)
            continue
        # One external module or detector per placement, named <name> or <name>_<tag>.
        frag_entries = json.load(open(frag))["entries"]
        if isinstance(frag_entries, dict):          # variants
            if args.variant not in frag_entries:
                missing.append(f"{frag} (no '{args.variant}' variant)")
                continue
            frag_entries = frag_entries[args.variant]
        for e in frag_entries:
            suffix = "" if e["tag"] == "barrel" else "_" + e["tag"][:8].upper()
            ename = (name + suffix)[:15]
            entry = {"name": ename, "title": f"{title} [{e['tag']}]",
                     "macro": e["macro"], "anchor": e["anchor"]}
            if e.get("placement"):
                entry["placement"] = e["placement"]
            if kind == "sensitive":
                entry["detID"] = DET_ID[name]
                entry["sensitiveVolumes"] = SENSITIVE_VOLUMES[name]
                detectors.append(entry)
            else:
                modules.append(entry)
            names.append(ename)

    if missing:
        raise SystemExit("no module_entries.json for:\n  " + "\n  ".join(missing)
                         + "\n(run roundtrip_module.py for each module first)")

    pre = args.out_prefix
    ext_path = os.path.join(study, f"{pre}externalDetectors.json")
    det_path = os.path.join(study, f"{pre}detectorlist.json")
    with open(ext_path, "w") as fh:
        json.dump({"externalModules": modules, "externalDetectors": detectors},
                  fh, indent=2)
    with open(det_path, "w") as fh:
        json.dump({args.name: names}, fh, indent=2)

    print(f"wrote {ext_path}")
    print(f"  {len(modules)} passive external module(s): "
          f"{', '.join(e['name'] for e in modules)}")
    print(f"  {len(detectors)} sensitive external detector(s): "
          + ", ".join(f"{e['name']} on DetID {e['detID']} "
                      f"(sensitive: {', '.join(e['sensitiveVolumes'])})"
                      for e in detectors))
    print(f"wrote {det_path}: {args.name} = {names}")
    print()
    print("run it with:")
    print(f"  o2-sim-serial -n <N> -g boxgen \\")
    print(f"      --detectorList {args.name}:{det_path} \\")
    print(f"      --extGeomFile {ext_path} \\")
    print(f"      --seed <SEED> --configKeyValues 'SimCutParams.trackSeed=true'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
