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

"""Compare the material two geometries present to the same rays.

x/X0 and x/lambda are integrated along a fixed set of Fibonacci-sphere rays through both
geometries.

Usage:
  matbudget_diff.py A/o2sim_geometry.root B/o2sim_geometry.root \
      --rays 2000 --rmax 45 [--json out.json]
"""

import argparse
import json
import math
import sys


def directions(n):
    """n roughly-uniform directions on the sphere (Fibonacci)."""
    ga = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(n):
        z = 1.0 - (2.0 * i + 1.0) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = ga * i
        out.append((r * math.cos(phi), r * math.sin(phi), z))
    return out


def integrate(mgr, dirs, rmax, origin=(0.0, 0.0, 0.0)):
    """Per ray: (sum x/X0, sum x/lambda, number of volumes crossed)."""
    import ROOT
    ROOT.gGeoManager = mgr
    out = []
    for (dx, dy, dz) in dirs:
        mgr.InitTrack(origin[0], origin[1], origin[2], dx, dy, dz)
        x0 = lam = 0.0
        ncross = 0
        travelled = 0.0
        while not mgr.IsOutside() and travelled < rmax and ncross < 20000:
            node = mgr.GetCurrentNode()
            if node is None:
                break
            med = node.GetVolume().GetMedium()
            mgr.FindNextBoundary()
            step = mgr.GetStep()
            if travelled + step > rmax:
                step = rmax - travelled
            if med is not None and step > 0:
                mat = med.GetMaterial()
                rl, il = mat.GetRadLen(), mat.GetIntLen()
                if rl > 0:
                    x0 += step / rl
                if il > 0:
                    lam += step / il
            travelled += step
            ncross += 1
            mgr.Step()
            if mgr.GetStep() <= 0 and step <= 0:
                break
        out.append((x0, lam, ncross))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("geometry_a")
    p.add_argument("geometry_b")
    p.add_argument("--rays", type=int, default=2000)
    p.add_argument("--rmax", type=float, default=45.0,
                   help="integrate out to this distance from the origin, in cm")
    p.add_argument("--json")
    p.add_argument("--dump-rays", metavar="CSV",
                   help="per-ray x/X0 and x/lambda for both geometries, so the "
                        "distribution can be plotted rather than summarised away")
    args = p.parse_args()

    import ROOT
    ROOT.gROOT.SetBatch(True)
    dirs = directions(args.rays)

    def load(path):
        f = ROOT.TFile.Open(path)
        key = f.GetListOfKeys().At(0).GetName()
        return f, f.Get(key)

    fa, ma = load(args.geometry_a)
    ra = integrate(ma, dirs, args.rmax)
    fa.Close()
    fb, mb = load(args.geometry_b)
    rb = integrate(mb, dirs, args.rmax)
    fb.Close()

    diffs_x0, diffs_l, rel = [], [], []
    suma = sumb = 0.0
    for (xa, la, na), (xb, lb, nb) in zip(ra, rb):
        suma += xa
        sumb += xb
        diffs_x0.append(abs(xa - xb))
        diffs_l.append(abs(la - lb))
        if max(xa, xb) > 0:
            rel.append(abs(xa - xb) / max(xa, xb))

    n = len(ra)
    diffs_x0.sort(); rel.sort()
    res = {
        "rays": n, "rmax_cm": args.rmax,
        "meanX0_a": suma / n, "meanX0_b": sumb / n,
        "meanAbsDiffX0": sum(diffs_x0) / n,
        "maxAbsDiffX0": diffs_x0[-1],
        "medianRelDiff": rel[len(rel) // 2] if rel else 0.0,
        "p99RelDiff": rel[min(len(rel) - 1, (99 * len(rel)) // 100)] if rel else 0.0,
        "maxRelDiff": rel[-1] if rel else 0.0,
        "raysAbove1pct": sum(1 for r in rel if r > 0.01),
        "raysAbove10pct": sum(1 for r in rel if r > 0.10),
        "meanAbsDiffLambda": sum(diffs_l) / n,
        "meanCrossings_a": sum(x[2] for x in ra) / n,
        "meanCrossings_b": sum(x[2] for x in rb) / n,
    }

    if args.dump_rays:
        with open(args.dump_rays, "w") as fh:
            fh.write("ux,uy,uz,x0_a,x0_b,lambda_a,lambda_b,crossings_a,crossings_b\n")
            for u, (xa, la, na), (xb, lb, nb) in zip(dirs, ra, rb):
                # 17 significant digits, so a double round-trips exactly.
                fh.write(f"{u[0]:.9g},{u[1]:.9g},{u[2]:.9g},{xa:.17g},{xb:.17g},"
                         f"{la:.17g},{lb:.17g},{na},{nb}\n")
        print(f"wrote {args.dump_rays}")

    print(f"{n} Fibonacci rays from the origin, integrated to r = {args.rmax} cm")
    print(f"  mean x/X0   A {res['meanX0_a']:.6f}   B {res['meanX0_b']:.6f}   "
          f"({100.0 * (res['meanX0_b'] - res['meanX0_a']) / max(1e-30, res['meanX0_a']):+.3f} %)")
    print(f"  mean |diff| x/X0  {res['meanAbsDiffX0']:.3e}   max {res['maxAbsDiffX0']:.3e}")
    print(f"  relative per ray: median {res['medianRelDiff']:.3e}, "
          f"p99 {res['p99RelDiff']:.3e}, max {res['maxRelDiff']:.3e}")
    print(f"  rays differing by >1 %: {res['raysAbove1pct']} / {n}; "
          f">10 %: {res['raysAbove10pct']} / {n}")
    print(f"  mean |diff| x/lambda  {res['meanAbsDiffLambda']:.3e}")
    print(f"  mean volumes crossed  A {res['meanCrossings_a']:.1f}   "
          f"B {res['meanCrossings_b']:.1f}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
