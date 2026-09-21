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

"""Compare the hit positions of two o2-sim runs, hit by hit.

Hits are keyed by (event, track) and compared in the order the track made them, which
SimCutParams.trackSeed=true makes well defined; the detector id is never used for matching.

Usage:
  compare_hits.py A/ B/ --branch-a ITSHit --branch-b ITSHit \
      --file-a o2sim_HitsITS.root --file-b o2sim.root [--json out.json]
"""

import argparse
import json
import math
import os
import sys


def primary_map(rundir):
    """{event: {trackID: primary ordinal}} for tracks that ARE primaries.

    A primary ordinal is shared between two runs; a track index is not.
    """
    import ROOT
    path = os.path.join(rundir, "o2sim.root")
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise SystemExit(f"cannot open {path} (needed for MCTrack)")
    tree = f.Get("o2sim")
    if not tree or not tree.GetBranch("MCTrack"):
        raise SystemExit(f"no MCTrack branch in {path}")
    out = {}
    for iev in range(tree.GetEntries()):
        tree.GetEntry(iev)
        tracks = getattr(tree, "MCTrack")
        m, ordinal = {}, 0
        for i in range(tracks.size()):
            if tracks.at(i).getMotherTrackId() < 0:
                m[i] = ordinal
                ordinal += 1
        out[iev] = m
    f.Close()
    return out


def load_hits(rundir, filename, branch, primaries=None):
    """Return {(event, key): [(x, y, z), ...]} in the order the hits appear.

    `key` is the track index, or -- when `primaries` is given -- the primary
    ordinal, and hits of secondary tracks are dropped.
    """
    import ROOT

    path = os.path.join(rundir, filename)
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise SystemExit(f"cannot open {path}")
    tree = f.Get("o2sim")
    if not tree:
        raise SystemExit(f"no 'o2sim' tree in {path}")
    if not tree.GetBranch(branch):
        have = [b.GetName() for b in tree.GetListOfBranches()]
        raise SystemExit(f"no branch {branch!r} in {path}; have {have}")

    hits = {}
    total = 0
    for iev in range(tree.GetEntries()):
        tree.GetEntry(iev)
        vec = getattr(tree, branch)
        for i in range(vec.size()):
            h = vec.at(i)
            key = h.GetTrackID()
            if primaries is not None:
                key = primaries.get(iev, {}).get(key)
                if key is None:
                    continue          # a secondary: not a shared identity
            hits.setdefault((iev, key), []).append(
                (h.GetX(), h.GetY(), h.GetZ()))
            total += 1
    f.Close()
    return hits, total


def compare_nearest(a, b, tol):
    """For every hit of A, the nearest hit of B on the same track.

    The native ITS and an external detector define hits differently, so n-th hits do not match.
    """
    res = {"hitsMatched": 0, "hitsUnmatched": 0, "withinTolerance": 0,
           "maxDr": 0.0, "sumDr": 0.0, "tracksOnlyInA": 0, "worst": None,
           "drQuantiles": {}}
    drs = []
    for key, ha in sorted(a.items()):
        hb = b.get(key)
        if not hb:
            res["tracksOnlyInA"] += 1
            res["hitsUnmatched"] += len(ha)
            continue
        for (xa, ya, za) in ha:
            best, bestpt = None, None
            for (xb, yb, zb) in hb:
                d = math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)
                if best is None or d < best:
                    best, bestpt = d, (xb, yb, zb)
            res["hitsMatched"] += 1
            res["sumDr"] += best
            drs.append(best)
            if best <= tol:
                res["withinTolerance"] += 1
            if best > res["maxDr"]:
                res["maxDr"] = best
                res["worst"] = {"event": key[0], "trackID": key[1],
                                "a": [xa, ya, za], "b": list(bestpt), "dr": best}
    if drs:
        drs.sort()
        for q in (50, 90, 99):
            res["drQuantiles"][f"p{q}"] = drs[min(len(drs) - 1, (q * len(drs)) // 100)]
        res["meanDr"] = res["sumDr"] / len(drs)
    else:
        res["meanDr"] = 0.0
    return res


def compare(a, b, tol):
    """Compare two keyed hit maps. Returns a result dict."""
    keys_a, keys_b = set(a), set(b)
    common = keys_a & keys_b

    res = {
        "tracksOnlyInA": len(keys_a - keys_b),
        "tracksOnlyInB": len(keys_b - keys_a),
        "tracksCommon": len(common),
        "tracksWithDifferentHitCount": 0,
        "hitsCompared": 0,
        "hitsWithinTolerance": 0,
        "maxDx": 0.0, "maxDy": 0.0, "maxDz": 0.0, "maxDr": 0.0,
        "sumDr": 0.0,
        "worst": None,
    }

    for key in sorted(common):
        ha, hb = a[key], b[key]
        if len(ha) != len(hb):
            res["tracksWithDifferentHitCount"] += 1
        for (xa, ya, za), (xb, yb, zb) in zip(ha, hb):
            dx, dy, dz = abs(xa - xb), abs(ya - yb), abs(za - zb)
            dr = math.sqrt(dx * dx + dy * dy + dz * dz)
            res["hitsCompared"] += 1
            res["sumDr"] += dr
            if dr <= tol:
                res["hitsWithinTolerance"] += 1
            res["maxDx"] = max(res["maxDx"], dx)
            res["maxDy"] = max(res["maxDy"], dy)
            res["maxDz"] = max(res["maxDz"], dz)
            if dr > res["maxDr"]:
                res["maxDr"] = dr
                res["worst"] = {
                    "event": key[0], "trackID": key[1],
                    "a": [xa, ya, za], "b": [xb, yb, zb], "dr": dr,
                }

    n = res["hitsCompared"]
    res["meanDr"] = res["sumDr"] / n if n else 0.0
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--file-a", default="o2sim_HitsITS.root")
    p.add_argument("--file-b", default="o2sim_HitsITS.root")
    p.add_argument("--branch-a", default="ITSHit")
    p.add_argument("--branch-b", default="ITSHit")
    p.add_argument("--tol", type=float, default=0.0,
                   help="position tolerance in cm; 0 means require exact equality")
    p.add_argument("--primaries", action="store_true",
                   help="key hits by the primary's ordinal in the event and drop "
                        "hits of secondaries; the only identity two runs share")
    p.add_argument("--match", choices=("order", "nearest"), default="order",
                   help="'order' compares the n-th hit of each track and is the "
                        "right test between two runs of the same geometry; "
                        "'nearest' asks whether every hit of A has a counterpart "
                        "at the same place in B, which is the geometry question "
                        "when the two sides define a hit differently")
    p.add_argument("--json", help="also write the result as JSON here")
    args = p.parse_args()

    pa = primary_map(args.dir_a) if args.primaries else None
    pb = primary_map(args.dir_b) if args.primaries else None
    a, na = load_hits(args.dir_a, args.file_a, args.branch_a, pa)
    b, nb = load_hits(args.dir_b, args.file_b, args.branch_b, pb)

    if args.match == "nearest":
        res = compare_nearest(a, b, args.tol)
        res["hitsInA"], res["hitsInB"], res["tolerance"] = na, nb, args.tol
        res["match"] = "nearest"
        print(f"A: {args.dir_a}/{args.file_a}:{args.branch_a}  {na} hits, {len(a)} tracks")
        print(f"B: {args.dir_b}/{args.file_b}:{args.branch_b}  {nb} hits, {len(b)} tracks")
        print(f"for each hit of A, the nearest on the same track in B:")
        print(f"  matched {res['hitsMatched']}, "
              f"no such track in B {res['hitsUnmatched']} "
              f"({res['tracksOnlyInA']} track(s))")
        print(f"  within {args.tol} cm: {res['withinTolerance']} "
              f"({100.0 * res['withinTolerance'] / max(1, res['hitsMatched']):.1f} %)")
        print(f"  |dr| mean {res['meanDr']:.6g}, median {res['drQuantiles'].get('p50', 0):.6g}, "
              f"p90 {res['drQuantiles'].get('p90', 0):.6g}, "
              f"p99 {res['drQuantiles'].get('p99', 0):.6g}, max {res['maxDr']:.6g} cm")
        if res["worst"]:
            w = res["worst"]
            print(f"  worst: event {w['event']} track {w['trackID']} dr {w['dr']:.4g} cm")
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(res, fh, indent=2)
            print(f"wrote {args.json}")
        ok = res["hitsMatched"] and res["withinTolerance"] == res["hitsMatched"]
        print("VERDICT:", "every hit has a counterpart within tolerance"
              if ok else "NOT all hits matched within tolerance")
        return 0 if ok else 1

    res = compare(a, b, args.tol)
    res["hitsInA"] = na
    res["hitsInB"] = nb
    res["tolerance"] = args.tol

    print(f"A: {args.dir_a}/{args.file_a}:{args.branch_a}  {na} hits, {len(a)} tracks")
    print(f"B: {args.dir_b}/{args.file_b}:{args.branch_b}  {nb} hits, {len(b)} tracks")
    print(f"tracks: {res['tracksCommon']} common, "
          f"{res['tracksOnlyInA']} only in A, {res['tracksOnlyInB']} only in B, "
          f"{res['tracksWithDifferentHitCount']} with a different hit count")
    print(f"hits compared: {res['hitsCompared']}, "
          f"within {args.tol} cm: {res['hitsWithinTolerance']}")
    print(f"max |dx| {res['maxDx']:.6g}  max |dy| {res['maxDy']:.6g}  "
          f"max |dz| {res['maxDz']:.6g}  cm")
    print(f"max |dr| {res['maxDr']:.6g} cm, mean |dr| {res['meanDr']:.6g} cm")
    if res["worst"]:
        w = res["worst"]
        print(f"worst: event {w['event']} track {w['trackID']} "
              f"A={w['a']} B={w['b']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {args.json}")

    identical = (res["hitsInA"] == res["hitsInB"]
                 and res["tracksOnlyInA"] == 0 and res["tracksOnlyInB"] == 0
                 and res["tracksWithDifferentHitCount"] == 0
                 and res["hitsCompared"] == res["hitsWithinTolerance"])
    print("VERDICT:", "identical within tolerance" if identical else "DIFFERENT")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
