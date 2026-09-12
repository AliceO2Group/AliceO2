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
# Since: 2026-09

"""Compare two o2-sim runs as distributions, not hit by hit.

  compare_distributions.py A/ B/ --file-a o2sim_HitsITS.root --branch-a ITSHit \
      --file-b o2sim.root --branch-b CITSHit --json out.json --npz out.npz
"""
import argparse
import json
import math
import numpy as np


def read(folder, fname, branch):
    import ROOT
    f = ROOT.TFile.Open(f"{folder}/{fname}")
    t = f.Get("o2sim")
    r, z, edep, per_event = [], [], [], []
    n = t.GetEntries()
    for i in range(n):
        t.GetEntry(i)
        hits = getattr(t, branch)
        per_event.append(len(hits))
        for h in hits:
            x, y, zz = h.GetX(), h.GetY(), h.GetZ()
            r.append(math.hypot(x, y))
            z.append(zz)
            # both hit classes carry the deposit under this name
            try:
                edep.append(h.GetEnergyLoss())
            except AttributeError:
                edep.append(float("nan"))
    f.Close()
    return (np.array(r), np.array(z), np.array(edep), np.array(per_event, dtype=float))


ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--file-a", default="o2sim_HitsITS.root")
ap.add_argument("--branch-a", default="ITSHit")
ap.add_argument("--file-b", default="o2sim.root")
ap.add_argument("--branch-b", default="CITSHit")
ap.add_argument("--json"); ap.add_argument("--npz")
args = ap.parse_args()

ra, za, ea, na = read(args.a, args.file_a, args.branch_a)
rb, zb, eb, nb = read(args.b, args.file_b, args.branch_b)


def stat(name, x, y):
    """Compare two samples of the same observable."""
    out = {"n_a": int(x.size), "n_b": int(y.size),
           "mean_a": float(np.nanmean(x)), "mean_b": float(np.nanmean(y)),
           "std_a": float(np.nanstd(x)), "std_b": float(np.nanstd(y))}
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    if hi > lo:
        bins = np.linspace(lo, hi, 101)
        ha, _ = np.histogram(x[~np.isnan(x)], bins=bins, density=True)
        hb, _ = np.histogram(y[~np.isnan(y)], bins=bins, density=True)
        w = bins[1] - bins[0]
        # total variation distance: 0 = the same distribution, 1 = disjoint
        out["totalVariation"] = float(0.5 * w * np.abs(ha - hb).sum())
    print(f"  {name:16s} A {out['mean_a']:12.5g} +- {out['std_a']:<11.5g} "
          f"B {out['mean_b']:12.5g} +- {out['std_b']:<11.5g} "
          f"TV {out.get('totalVariation', float('nan')):.4f}")
    return out


print(f"hits: A {ra.size}  B {rb.size}   events: A {na.size}  B {nb.size}")
res = {"radius_cm": stat("radius [cm]", ra, rb),
       "z_cm": stat("z [cm]", za, zb),
       "edep": stat("energy loss", ea, eb),
       "hits_per_event": stat("hits/event", na, nb)}
if args.json:
    json.dump(res, open(args.json, "w"), indent=2)
    print(f"wrote {args.json}")
if args.npz:
    np.savez_compressed(args.npz, ra=ra, rb=rb, za=za, zb=zb, ea=ea, eb=eb, na=na, nb=nb)
    print(f"wrote {args.npz}")
