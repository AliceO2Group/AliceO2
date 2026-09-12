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

"""Carry the baseline's Geant cuts and processes over to the CAD run, by medium name.

A loaded cut is resolved by (module, local index), which the CAD run does not share with the
baseline; the medium NAME survives the round trip. So the CAD module prefix is stripped, names are
matched, and the baseline's cuts are written under the CAD run's module and local index:

  baseline   module ITS,   local 1, medium `ITS_AIR$`
  CAD run    module CITS,  local 7, medium `CITS_ITS_AIR$`

  remap_cuts.py --baseline base.json --cad-dump cad_out.json --out cad_in.json
  remap_cuts.py --compare  base.json cad_out2.json
"""

import argparse
import json
import sys


def index_baseline(doc):
    """medium name (unprefixed by module) -> its cuts/processes record."""
    out = {}
    for key, entries in doc.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            name = e.get("medium_name")
            if not name:
                continue
            # `ITS_AIR$` under module `ITS` -> key on both the full name and the
            # part after the module prefix, so either spelling matches later.
            out.setdefault(name, e)
            if name.startswith(key + "_"):
                out.setdefault(name[len(key) + 1:], e)
    return out


def strip_module(name, module):
    return name[len(module) + 1:] if name.startswith(module + "_") else name


def build(baseline, caddump):
    by_name = index_baseline(baseline)
    out, matched, unmatched = {}, [], []
    for key, entries in caddump.items():
        if not isinstance(entries, list):
            out[key] = entries          # default / enableSpecial* pass through
            continue
        rebuilt = []
        for e in entries:
            bare = strip_module(e.get("medium_name", ""), key)
            src = by_name.get(bare) or by_name.get(e.get("medium_name", ""))
            if src is None:
                unmatched.append(f"{key}/{e.get('medium_name')}")
                continue
            rebuilt.append({
                "local_id": e["local_id"],
                "global_id": e["global_id"],
                "medium_name": e["medium_name"],
                "material_name": e.get("material_name"),
                "cuts": src.get("cuts", {}),
                "processes": src.get("processes", {}),
            })
            matched.append(f"{key}/{e.get('medium_name')} <- {bare}")
        out[key] = rebuilt
    for k in ("default", "enableSpecialCuts", "enableSpecialProcesses"):
        if k in baseline:
            out[k] = baseline[k]
    return out, matched, unmatched


def compare(baseline, caddump):
    """Do the two runs give every medium the same cuts and processes?"""
    by_name = index_baseline(baseline)
    same, differ, missing = 0, [], []
    for key, entries in caddump.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            bare = strip_module(e.get("medium_name", ""), key)
            src = by_name.get(bare) or by_name.get(e.get("medium_name", ""))
            if src is None:
                missing.append(f"{key}/{e.get('medium_name')}")
                continue
            if (src.get("cuts") == e.get("cuts")
                    and src.get("processes") == e.get("processes")):
                same += 1
            else:
                bad = [k for k in set(src.get("cuts", {})) | set(e.get("cuts", {}))
                       if src.get("cuts", {}).get(k) != e.get("cuts", {}).get(k)]
                bad += [f"proc:{k}" for k in
                        set(src.get("processes", {})) | set(e.get("processes", {}))
                        if src.get("processes", {}).get(k) != e.get("processes", {}).get(k)]
                differ.append((f"{key}/{e.get('medium_name')}", bad))
    return same, differ, missing


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True)
    p.add_argument("--cad-dump", required=True)
    p.add_argument("--out")
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()

    baseline = json.load(open(args.baseline))
    caddump = json.load(open(args.cad_dump))

    if args.compare:
        same, differ, missing = compare(baseline, caddump)
        print(f"media compared: {same + len(differ) + len(missing)}")
        print(f"  identical cuts and processes: {same}")
        print(f"  differing: {len(differ)}")
        for name, bad in differ[:8]:
            print(f"    {name}: {bad[:6]}")
        print(f"  no baseline medium of that name: {len(missing)}"
              + (f"  e.g. {missing[:5]}" if missing else ""))
        ok = not differ and not missing
        print("VERDICT:", "both runs give every medium the same cuts and processes"
              if ok else "DIFFERENT -- do not trust a transport comparison")
        return 0 if ok else 1

    if not args.out:
        raise SystemExit("--out is required unless --compare is given")
    out, matched, unmatched = build(baseline, caddump)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}: {len(matched)} medium/media matched by name, "
          f"{len(unmatched)} unmatched")
    if unmatched:
        print(f"  [WARN] no baseline cuts for: {unmatched[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
