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

"""How far the tessellation moves a cylindrical surface, measured rather than estimated.

The error is the sagitta of the chord between neighbouring lateral-wall vertices,

    sagitta = R (1 - cos(dphi / 2))

compared with the exact radius of every leaf the cascade recognised as a TGeoTube / TGeoTubeSeg.
"""
import glob
import math
import os
import struct
import sys
import numpy as np
import ROOT

CSG, MESH = sys.argv[1], sys.argv[2]


def read_vertices(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<I", fh.read(4))[0]
        return np.frombuffer(fh.read(n * 36), dtype="<f4").reshape(-1, 3)


rows = []
for shape_file in sorted(glob.glob(f"{CSG}/shape_*.root")):
    key = os.path.basename(shape_file)[len("shape_"):-len(".root")]
    facets = f"{MESH}/facets_{key}.bin"
    if not os.path.exists(facets):
        continue
    tf = ROOT.TFile.Open(shape_file)
    sh = tf.Get("shape")
    if sh is None or not sh.InheritsFrom("TGeoTube") or tf.Get("placement"):
        tf.Close(); continue
    rmax = sh.GetRmax()
    tf.Close()
    v = read_vertices(facets)
    if not len(v):
        continue
    r = np.hypot(v[:, 0], v[:, 1])
    on = np.abs(r - rmax) < 1e-4 * rmax
    if on.sum() < 12:
        continue
    phi = np.unique(np.round(np.arctan2(v[on, 1], v[on, 0]), 6))
    if len(phi) < 4:
        continue
    gaps = np.diff(np.sort(phi))
    gaps = gaps[gaps > 1e-5]
    if not len(gaps):
        continue
    dphi = float(np.median(gaps))
    rows.append((key, rmax, len(phi), dphi, rmax * (1.0 - math.cos(dphi / 2.0))))

rows.sort(key=lambda t: -t[4])
print(f"{'part':<44}{'R (cm)':>9}{'segments':>10}{'sagitta':>12}")
for key, rmax, nphi, dphi, sag in rows[:10]:
    print(f"{key[:44]:<44}{rmax:9.3f}{nphi:10d}{sag * 1e4:9.1f} um")
if rows:
    sag = np.array([r[4] for r in rows]) * 1e4
    R = np.array([r[1] for r in rows])
    print(f"\n{len(rows)} cylindrical parts, R = {R.min():.2f}-{R.max():.2f} cm")
    print(f"  surface displacement: median {np.median(sag):.0f} um, "
          f"p90 {np.percentile(sag, 90):.0f} um, max {sag.max():.0f} um")
