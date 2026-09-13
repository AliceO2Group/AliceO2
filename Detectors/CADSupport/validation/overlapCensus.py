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

"""Is this CAD assembly a LEGAL geometry for TGeo / Geant4? Measure it, pair by pair.

For every pair of PLACED solids:

1. **AABB rejection** first, which makes N^2 affordable.
2. `BRepExtrema_DistShapeShape` on the survivors: a positive distance settles the pair as
   **disjoint, by this much**.
3. `BRepAlgoAPI_Common` only where the distance is zero, i.e. where the pair touches or
   interpenetrates. Its **volume** is the discriminator:
   * volume ~ 0  -> **coincident faces**. Touching. This is the NORMAL case for an assembly and
     it is legal for TGeo, which tolerates shared boundaries.
   * volume > 0  -> **real interpenetration**. Illegal. Reported with the fraction of the smaller
     part it eats, the bounding box of the shared region (so it can be found), and a sampled
     maximum penetration depth.
   * volume ~ volume of the smaller part -> **containment**. Legal *if* the hierarchy declares it
     mother/daughter, illegal if both are placed as siblings -- which is what a flat CAD-to-TGeo
     conversion does. Reported separately because the fix is different.

Usage
-----
  overlapCensus.py --self-test
  overlapCensus.py --step Detectors/CADSupport/examples/ExcavatorArm.step --out excavator_arm.json
  overlapCensus.py --step .../CAD_noETA.stp --out alice3.json --max-pairs 4000
"""

import argparse
import itertools
import json
import re
import math
import sys
import time
from pathlib import Path

import cadsupport_path  # noqa: E402,F401  (puts ../tools on sys.path)

from cadsupport.occ_env import ensure_occ

ensure_occ()

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_SOLID, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Pnt

from assemblyOracle import Part, assembly_from_shapes, load_assembly

CENSUS_FORMAT_VERSION = 1


def volume_of(shape) -> float:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return abs(props.Mass())


def bbox_of(shape, tight=False):
    """`Bnd_Box.Add` inflates by the shape's own tolerance, which is right for a rejection test
    and wrong for a measurement -- it made a 0.5 cm shared slab read 0.5000002 cm. `AddOptimal`
    is the measurement version; the rejection stays conservative on purpose."""
    box = Bnd_Box()
    if tight:
        box.SetGap(0.0)
        brepbndlib.AddOptimal(shape, box, True, False)
    else:
        brepbndlib.Add(shape, box)
    return None if box.IsVoid() else box.Get()


def surface_of(shape):
    """The shape's boundary as a compound of faces.

    `BRepExtrema_DistShapeShape(vertex, solid)` is 0 for a point inside the solid, so penetration
    depth must be measured against the boundary.
    """
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import TopoDS_Compound
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        builder.Add(compound, explorer.Current())
        explorer.Next()
    return compound


def boxes_overlap(a, b, pad):
    if a is None or b is None:
        return False, 0.0
    volume = 1.0
    for k in range(3):
        lo = max(a[k], b[k])
        hi = min(a[k + 3], b[k + 3])
        if hi < lo - pad:
            return False, 0.0
        volume *= max(0.0, hi - lo)
    return True, volume


def distance_between(a, b):
    """Minimum separation of two placed shapes. Zero means touching OR interpenetrating; the two
    are told apart by the boolean, never by this number."""
    tool = BRepExtrema_DistShapeShape(a, b)
    if not tool.IsDone():
        tool.Perform()
    if not tool.IsDone():
        return None
    return tool.Value()


def distance_to_surface(point: gp_Pnt, shape):
    vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
    tool = BRepExtrema_DistShapeShape(vertex, shape)
    if not tool.IsDone():
        tool.Perform()
    return tool.Value() if tool.IsDone() else None


def penetration_depth(common, surface_a, surface_b, grid=6, budget=120):
    """Max over sampled interior points of the shared region of min(dist to A's surface,
    dist to B's surface).

    A sampled **lower bound**; the shared region's bounding-box diagonal is returned as the upper.
    """
    box = bbox_of(common, tight=True)
    if box is None:
        return 0.0, 0.0, 0
    diag = math.sqrt(sum((box[k + 3] - box[k]) ** 2 for k in range(3)))

    points = []
    explorer = TopExp_Explorer(common, TopAbs_VERTEX)
    seen = set()
    while explorer.More():
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopoDS import topods
        p = BRep_Tool.Pnt(topods.Vertex(explorer.Current()))
        key = (round(p.X(), 9), round(p.Y(), 9), round(p.Z(), 9))
        if key not in seen:
            seen.add(key)
            points.append(p)
        explorer.Next()

    classifier = BRepClass3d_SolidClassifier(common)
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                p = gp_Pnt(box[0] + (i + 0.5) * (box[3] - box[0]) / grid,
                           box[1] + (j + 0.5) * (box[4] - box[1]) / grid,
                           box[2] + (k + 0.5) * (box[5] - box[2]) / grid)
                classifier.Perform(p, 1e-9)
                if classifier.State() == TopAbs_IN:
                    points.append(p)
    points = points[:budget]

    best = 0.0
    for p in points:
        da = distance_to_surface(p, surface_a)
        db = distance_to_surface(p, surface_b)
        if da is None or db is None:
            continue
        best = max(best, min(da, db))
    return best, diag, len(points)


def census(parts, scale, pad_cm=0.1, zero_distance_cm=1.0e-9, zero_volume_cm3=1.0e-12,
           max_pairs=0, verbose=True, deep=True):
    """The full pairwise census. All reported lengths are cm, volumes cm^3.

    `pad_cm` inflates every bounding box before the rejection test. It does not change which pairs
    overlap; it decides which disjoint pairs get their separation measured.
    """
    pad = pad_cm / scale
    zero_distance = zero_distance_cm / scale
    zero_volume = zero_volume_cm3 / (scale ** 3)

    # Volumes are computed lazily and memoised; only pairs that survive the AABB test need them.
    volumes = {}
    surfaces = {}

    def volume(index):
        if index not in volumes:
            volumes[index] = volume_of(parts[index].shape)
        return volumes[index]

    n = len(parts)
    total_pairs = n * (n - 1) // 2
    aabb_survivors = []
    for i, j in itertools.combinations(range(n), 2):
        hit, box_volume = boxes_overlap(parts[i].bbox, parts[j].bbox, pad)
        if hit:
            aabb_survivors.append((i, j, box_volume))
    # Cheapest first: a small shared box is usually a corner touch and resolves fast.
    aabb_survivors.sort(key=lambda t: t[2])
    if max_pairs:
        aabb_survivors = aabb_survivors[:max_pairs]

    if verbose:
        print(f"  {n} placed solids -> {total_pairs} pairs; {len(aabb_survivors)} survive the "
              f"AABB rejection ({100.0 * len(aabb_survivors) / max(1, total_pairs):.2f} %)",
              flush=True)

    results = []
    counts = {"pairs": total_pairs, "aabb": len(aabb_survivors), "disjoint": 0,
              "coincident": 0, "interpenetrating": 0, "contained": 0, "failed": 0}
    started = time.time()
    for k, (i, j, _) in enumerate(aabb_survivors):
        a, b = parts[i], parts[j]
        record = {"a": a.name, "b": b.name, "aPath": a.path, "bPath": b.path,
                  "sameDefinition": a.definition == b.definition,
                  "volA": volume(i) * scale ** 3, "volB": volume(j) * scale ** 3}
        d = distance_between(a.shape, b.shape)
        if d is None:
            record["class"] = "failed"
            counts["failed"] += 1
            results.append(record)
            continue
        record["distance"] = d * scale
        if d > zero_distance:
            record["class"] = "disjoint"
            counts["disjoint"] += 1
            results.append(record)
            if verbose and (k + 1) % 25 == 0:
                print(f"    {k + 1}/{len(aabb_survivors)} ({time.time() - started:.1f} s)",
                      flush=True)
            continue

        # Touching or interpenetrating: only now is a boolean worth its cost.
        try:
            common = BRepAlgoAPI_Common(a.shape, b.shape)
            common.Build()
            ok = common.IsDone()
            shape = common.Shape() if ok else None
        except Exception as exc:
            record["class"] = "failed"
            record["error"] = str(exc)
            counts["failed"] += 1
            results.append(record)
            continue
        if not ok or shape is None:
            record["class"] = "failed"
            counts["failed"] += 1
            results.append(record)
            continue

        solids = 0
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solids += 1
            explorer.Next()
        raw_volume = volume_of(shape) if solids else 0.0
        record["commonSolids"] = solids
        record["commonVolume"] = raw_volume * scale ** 3
        smaller = min(volume(i), volume(j))
        record["fractionOfSmaller"] = raw_volume / smaller if smaller > 0 else 0.0

        if raw_volume <= zero_volume:
            record["class"] = "coincident"
            counts["coincident"] += 1
        else:
            box = bbox_of(shape, tight=True)
            if box is not None:
                record["commonBBox"] = [c * scale for c in box]
                record["commonExtent"] = sorted((box[k + 3] - box[k]) * scale for k in range(3))
            if record["fractionOfSmaller"] > 1.0 - 1e-6:
                record["class"] = "contained"
                counts["contained"] += 1
            else:
                record["class"] = "interpenetrating"
                counts["interpenetrating"] += 1
            if deep:
                if i not in surfaces:
                    surfaces[i] = surface_of(a.shape)
                if j not in surfaces:
                    surfaces[j] = surface_of(b.shape)
                depth, diag, samples = penetration_depth(shape, surfaces[i], surfaces[j])
                record["penetrationDepthSampled"] = depth * scale
                record["penetrationDepthUpper"] = diag * scale
                record["penetrationSamples"] = samples
        results.append(record)
        if verbose and (k + 1) % 25 == 0:
            print(f"    {k + 1}/{len(aabb_survivors)} ({time.time() - started:.1f} s)"
                  f"  [{counts['disjoint']}d {counts['coincident']}c "
                  f"{counts['interpenetrating']}I {counts['contained']}n]", flush=True)

    return results, counts, time.time() - started


# ---------------------------------------------------------------------------------------------

def self_test() -> int:
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if not ok else ""))
        if not ok:
            failures.append(name)

    def box(x0, y0, z0, x1, y1, z1):
        return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), gp_Pnt(x1, y1, z1)).Shape()

    # DistShapeShape(vertex, SOLID) is 0 for an interior point; penetration depth must be measured
    # against the boundary (surface_of), not the solid.
    inner = box(0, 0, 0, 2, 2, 2)
    d_solid = distance_to_surface(gp_Pnt(1, 1, 1), inner)
    check("TRAP: DistShapeShape from an interior point to the SOLID is 0, not the distance to "
          "its boundary", d_solid is not None and d_solid == 0.0, str(d_solid))
    d_surface = distance_to_surface(gp_Pnt(1, 1, 1), surface_of(inner))
    check("...and against the face compound it is the true 1 cm -- which is what the depth "
          "sampler uses", d_surface is not None and abs(d_surface - 1.0) < 1e-9, str(d_surface))

    parts = assembly_from_shapes([
        ("touchA", box(0, 0, 0, 2, 2, 2)),          # touchA | touchB share the face x=2
        ("touchB", box(2, 0, 0, 4, 2, 2)),
        ("gapA", box(10, 0, 0, 12, 2, 2)),          # 0.25 cm gap to gapB
        ("gapB", box(12.25, 0, 0, 14, 2, 2)),
        ("tinyA", box(20, 0, 0, 22, 2, 2)),         # 1e-7 cm gap: separate, and must be measured
        ("tinyB", box(22 + 1e-7, 0, 0, 24, 2, 2)),
        ("ovA", box(30, 0, 0, 34, 2, 2)),           # 0.5 cm interpenetration over [33.5, 34]
        ("ovB", box(33.5, 0, 0, 37, 2, 2)),
        ("outer", box(40, 40, 40, 46, 46, 46)),     # inner2 wholly contained in outer
        ("inner2", box(42, 42, 42, 44, 44, 44)),
    ])
    results, counts, _ = census(parts, scale=1.0, pad_cm=1.0, zero_distance_cm=1e-12,
                                verbose=False)
    by_pair = {tuple(sorted((r["a"], r["b"]))): r for r in results}

    r = by_pair.get(("touchA", "touchB"))
    check("touching pair: distance 0 and ZERO common volume -> coincident faces, not an overlap",
          r is not None and r["class"] == "coincident" and r["commonVolume"] < 1e-12,
          str(r))

    r = by_pair.get(("gapA", "gapB"))
    check("0.25 cm gap: reported disjoint with the separation measured",
          r is not None and r["class"] == "disjoint" and abs(r["distance"] - 0.25) < 1e-9, str(r))

    r = by_pair.get(("tinyA", "tinyB"))
    check("1e-7 cm gap: STILL reported disjoint, with the separation measured, not rounded to 0",
          r is not None and r["class"] == "disjoint" and abs(r["distance"] - 1e-7) < 1e-12, str(r))

    r = by_pair.get(("ovA", "ovB"))
    check("0.5 cm interpenetration: classified interpenetrating",
          r is not None and r["class"] == "interpenetrating", str(r))
    check("interpenetration: common volume is exactly 0.5 x 2 x 2 = 2 cm^3",
          r is not None and abs(r["commonVolume"] - 2.0) < 1e-9, str(r.get("commonVolume")))
    check("interpenetration: fraction of the smaller part is 2 / 14",
          r is not None and abs(r["fractionOfSmaller"] - 2.0 / 14.0) < 1e-9,
          str(r.get("fractionOfSmaller")))
    check("interpenetration: the shared slab is 0.5 cm thick",
          r is not None and abs(r["commonExtent"][0] - 0.5) < 1e-9, str(r.get("commonExtent")))
    # A 6-sample grid cannot land on the 0.25 cm mid-plane, so the sampled depth is bracketed.
    check("interpenetration: sampled depth is a LOWER bound on the true 0.25 cm, within one "
          "grid half-cell of it, and under the bbox-diagonal upper bound",
          r is not None and 0.25 - 0.5 / 12 - 1e-9 <= r["penetrationDepthSampled"] <= 0.25 + 1e-9
          and r["penetrationDepthUpper"] > r["penetrationDepthSampled"],
          f"{r.get('penetrationDepthSampled')} vs 0.25, upper {r.get('penetrationDepthUpper')}")

    r = by_pair.get(("inner2", "outer"))
    check("containment: classified `contained`, not `interpenetrating`",
          r is not None and r["class"] == "contained", str(r))
    check("containment: common volume equals the inner part's 8 cm^3",
          r is not None and abs(r["commonVolume"] - 8.0) < 1e-9, str(r.get("commonVolume")))

    check("census bookkeeping: 1 coincident, 1 interpenetrating, 1 contained, the rest disjoint",
          counts["coincident"] == 1 and counts["interpenetrating"] == 1
          and counts["contained"] == 1 and counts["failed"] == 0, str(counts))

    # The NEGATIVE control: no overlaps are reported when there are none.
    clean = assembly_from_shapes([("p0", box(0, 0, 0, 1, 1, 1)),
                                  ("p1", box(2, 0, 0, 3, 1, 1)),
                                  ("p2", box(4, 0, 0, 5, 1, 1))])
    _, clean_counts, _ = census(clean, scale=1.0, verbose=False)
    check("negative control: three separated boxes report 0 interpenetrating, 0 contained",
          clean_counts["interpenetrating"] == 0 and clean_counts["contained"] == 0,
          str(clean_counts))

    print(f"\n{'SELF-TEST PASSED' if not failures else 'SELF-TEST FAILED'}: "
          f"{len(failures)} failure(s)")
    return 0 if not failures else 1


def report(results, counts, scale, seconds, top=25):
    print()
    print(f"  AABB survivors {counts['aabb']} of {counts['pairs']} pairs   "
          f"({seconds:.1f} s)")
    print(f"  disjoint            {counts['disjoint']:6d}")
    print(f"  coincident faces    {counts['coincident']:6d}   (touching -- legal)")
    print(f"  INTERPENETRATING    {counts['interpenetrating']:6d}   (illegal for TGeo/Geant4)")
    print(f"  contained           {counts['contained']:6d}   (legal only as mother/daughter)")
    print(f"  failed              {counts['failed']:6d}")

    bad = [r for r in results if r.get("class") in ("interpenetrating", "contained")]
    bad.sort(key=lambda r: -r.get("commonVolume", 0.0))
    if bad:
        print()
        print(f"  {'pair':<52s} {'class':<17s} {'V_common cm^3':>14s} {'frac small':>11s} "
              f"{'depth cm':>10s}")
        for r in bad[:top]:
            print(f"  {r['a'][:24]:<24s} {r['b'][:24]:<25s} {r['class']:<17s} "
                  f"{r.get('commonVolume', 0):14.6g} {r.get('fractionOfSmaller', 0):11.4g} "
                  f"{r.get('penetrationDepthSampled', 0):10.4g}")
        if len(bad) > top:
            print(f"  ... and {len(bad) - top} more")

    gaps = sorted((r["distance"], r["a"], r["b"]) for r in results
                  if r.get("class") == "disjoint")
    if gaps:
        print()
        print(f"  tightest measured gaps between disjoint pairs (cm):")
        for d, a, b in gaps[:10]:
            print(f"    {d:12.6g}   {a} | {b}")
    coincident = [r for r in results if r.get("class") == "coincident"]
    if coincident:
        print()
        print(f"  coincident-face pairs (shared boundary, zero volume): {len(coincident)}")
        for r in coincident[:10]:
            print(f"    {r['a']} | {r['b']}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="cap the pairs examined after AABB rejection (bounded runs)")
    parser.add_argument("--max-parts", type=int, default=0)
    parser.add_argument("--parts", type=str, default="")
    parser.add_argument("--parts-regex", type=str, default="",
                        help="keep instances whose name matches this regex -- the way to select a "
                             "replicated prototype, whose copies are named NAME, NAME#1, NAME#2")
    parser.add_argument("--pad", type=float, default=0.1,
                        help="AABB inflation in cm: decides which DISJOINT pairs get their\n                             separation measured (default 0.1 cm)")
    parser.add_argument("--no-deep", action="store_true",
                        help="skip the penetration-depth sampling")
    parser.add_argument("--inject", type=str, default="",
                        help="NAME:DX,DY,DZ -- translate one part by this many cm before the "
                             "census. The positive control on a real model.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.step:
        parser.error("--step is required (unless --self-test)")

    started = time.time()
    parts, scale = load_assembly(args.step)
    if args.parts:
        wanted = set(args.parts.split(","))
        parts = [p for p in parts if p.name in wanted]
    if args.parts_regex:
        pattern = re.compile(args.parts_regex)
        parts = [p for p in parts if pattern.search(p.name)]
    if args.max_parts:
        parts = parts[:args.max_parts]
    print(f"  {args.step.name}: {len(parts)} placed solids, {scale} cm/unit "
          f"({time.time() - started:.1f} s)", flush=True)

    injected = None
    if args.inject:
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.gp import gp_Trsf, gp_Vec
        name, deltas = args.inject.split(":")
        dx, dy, dz = (float(v) / scale for v in deltas.split(","))
        for k, p in enumerate(parts):
            if p.name == name:
                trsf = gp_Trsf()
                trsf.SetTranslation(gp_Vec(dx, dy, dz))
                moved = p.shape.Moved(TopLoc_Location(trsf))
                parts[k] = Part(p.name + "@INJECTED", p.definition, p.path, moved)
                injected = parts[k].name
                break
        if injected is None:
            raise SystemExit(f"--inject: no part named {name}")
        print(f"  INJECTED: {injected} translated by {args.inject.split(':')[1]} cm", flush=True)

    results, counts, seconds = census(parts, scale, pad_cm=args.pad, max_pairs=args.max_pairs,
                                      deep=not args.no_deep)
    report(results, counts, scale, seconds)

    if args.out:
        args.out.write_text(json.dumps(
            {"version": CENSUS_FORMAT_VERSION, "model": str(args.step), "scaleToCm": scale,
             "nParts": len(parts), "parts": [p.name for p in parts], "injected": injected,
             "counts": counts, "seconds": seconds, "pairs": results}, indent=1))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
