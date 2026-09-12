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

"""Check that a round-tripped geometry carries the media of its source.

For every converted volume it finds the source volume of the same name (dropping the writer's
`__body` and `__mirrored` suffixes) and compares, exactly: the medium name, all eight Geant medium
parameters, the material's Z, A, density, radiation and interaction length, and every mixture
element's Z, A and weight. A volume left on the `Default` placeholder is reported separately.

Usage:
  check_media.py --original o2sim_geometry.root --macro conv/geom.C [--json out.json]
"""

import argparse
import json
import os
import sys

PARAMS = ("isvol", "ifield", "fieldm", "tmaxfd", "stemax", "deemax", "epsil", "stmin")


def base_name(name, hollow_rename=None):
    """The source volume name behind a writer-emitted part name.

    `hollow_rename` is an exact map of tagged hall name -> source name, read from the writer report.
    """
    if hollow_rename and name in hollow_rename:
        return hollow_rename[name]
    for suffix in ("__mirrored", "__body"):
        while name.endswith(suffix):
            name = name[: -len(suffix)]
    if hollow_rename and name in hollow_rename:
        return hollow_rename[name]
    # `X#2` is the writer's disambiguation of one TGeo name over two definitions
    return name.split("#", 1)[0]


def describe(vol):
    # An assembly carries ROOT's `dummy` medium; the mother's material lives in its `__body` leaf.
    if vol.IsAssembly():
        return None
    med = vol.GetMedium()
    if med is None:
        return None
    mat = med.GetMaterial()
    d = {
        "medium": str(med.GetName()),
        "params": [float(med.GetParam(i)) for i in range(8)],
        "material": str(mat.GetName()),
        "Z": float(mat.GetZ()), "A": float(mat.GetA()),
        "density": float(mat.GetDensity()),
        "radLen": float(mat.GetRadLen()), "intLen": float(mat.GetIntLen()),
        "isMixture": bool(mat.IsMixture()),
    }
    if mat.IsMixture():
        n = int(mat.GetNelements())
        zs, as_, ws = mat.GetZmixt(), mat.GetAmixt(), mat.GetWmixt()
        d["elements"] = [[float(zs[i]), float(as_[i]), float(ws[i])] for i in range(n)]
    return d


def diff(a, b, rtol):
    """Field names that disagree between two describe() dicts."""
    bad = []
    if a["medium"] != b["medium"]:
        bad.append("mediumName")
    for i, k in enumerate(PARAMS):
        if a["params"][i] != b["params"][i]:
            bad.append(k)
    if a["material"] != b["material"]:
        bad.append("materialName")
    for k in ("Z", "A", "density", "radLen", "intLen"):
        x, y = a[k], b[k]
        if x != y and (abs(x - y) > rtol * max(abs(x), abs(y), 1e-300)):
            bad.append(k)
    if a["isMixture"] != b["isMixture"]:
        bad.append("isMixture")
    elif a["isMixture"]:
        if len(a["elements"]) != len(b["elements"]):
            bad.append("nElements")
        else:
            for (za, aa, wa), (zb, ab, wb) in zip(a["elements"], b["elements"]):
                if za != zb or aa != ab or wa != wb:
                    bad.append("elements")
                    break
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", required=True, help="the source o2sim_geometry.root")
    p.add_argument("--macro", required=True, help="the converted geom.C")
    p.add_argument("--writer-report", default=None,
                   help="the writer's JSON report, read for hollowVolumes/hollowTag "
                        "so a tagged hall volume still finds its source")
    p.add_argument("--rtol", type=float, default=0.0,
                   help="relative tolerance on the scalar material fields "
                        "(default 0: require exact equality)")
    p.add_argument("--json", help="write the full result here")
    args = p.parse_args()

    hollow_rename = {}
    if args.writer_report:
        with open(args.writer_report) as fh:
            rep = json.load(fh)
        tag = rep.get("hollowTag")
        if tag:
            for h in rep.get("hollowVolumes", []):
                hollow_rename[f"{h}_{tag}"] = h

    import ROOT
    ROOT.gROOT.SetBatch(True)

    # The source is read into its own manager and set aside; the macro builds into a second one.
    src_mgr = ROOT.TGeoManager.Import(args.original)
    source = {}
    for vol in src_mgr.GetListOfVolumes():
        d = describe(vol)
        if d is not None:
            source[str(vol.GetName())] = d
    ROOT.gGeoManager = ROOT.nullptr

    # Interpreted, not ACLiC-compiled, as ExternalModule JITs it.
    ROOT.gROOT.ProcessLine(f'.L {os.path.abspath(args.macro)}')
    ROOT.gGeoManager = ROOT.TGeoManager("converted", "converted")
    top = ROOT.build(False)
    ROOT.gGeoManager.SetTopVolume(top)
    ROOT.gGeoManager.CloseGeometry()

    res = {"nConverted": 0, "matched": 0, "default": [], "missingInSource": [],
           "disagreements": [], "fieldCounts": {}, "assembliesSkipped": 0,
           "maxRelDevRadLen": 0.0, "maxRelDevIntLen": 0.0}
    for vol in ROOT.gGeoManager.GetListOfVolumes():
        name = str(vol.GetName())
        if vol.IsAssembly():
            res["assembliesSkipped"] += 1
            continue
        d = describe(vol)
        if d is None:
            continue
        res["nConverted"] += 1
        if d["medium"] == "Default":
            res["default"].append(name)
            continue
        src = source.get(base_name(name, hollow_rename))
        if src is None:
            res["missingInSource"].append(name)
            continue
        for key, slot in (("radLen", "maxRelDevRadLen"), ("intLen", "maxRelDevIntLen")):
            x, y = src[key], d[key]
            if max(abs(x), abs(y)) > 0:
                res[slot] = max(res[slot], abs(x - y) / max(abs(x), abs(y)))
        bad = diff(src, d, args.rtol)
        if bad:
            res["disagreements"].append({"volume": name, "fields": bad,
                                         "source": src, "converted": d})
            for f in bad:
                res["fieldCounts"][f] = res["fieldCounts"].get(f, 0) + 1
        else:
            res["matched"] += 1

    n = res["nConverted"]
    print(f"converted volumes with a medium: {n}")
    print(f"  media identical to the source: {res['matched']}")
    print(f"  left on the Default placeholder (transparent): {len(res['default'])}"
          + (f"  e.g. {res['default'][:5]}" if res["default"] else ""))
    print(f"  no source volume of that name: {len(res['missingInSource'])}"
          + (f"  e.g. {res['missingInSource'][:5]}" if res["missingInSource"] else ""))
    print(f"  assemblies skipped (they hold no material): {res['assembliesSkipped']}")
    print(f"  max relative deviation: radLen {res['maxRelDevRadLen']:.3e}, "
          f"intLen {res['maxRelDevIntLen']:.3e}  (derived by ROOT from the recipe, "
          f"not carried)")
    print(f"  disagreeing with the source: {len(res['disagreements'])}"
          + (f"  fields {res['fieldCounts']}" if res["fieldCounts"] else ""))
    for d in res["disagreements"][:5]:
        print(f"    {d['volume']}: {d['fields']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {args.json}")

    ok = (n > 0 and res["matched"] == n)
    print("VERDICT:", "every volume carries its source medium" if ok else "INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
