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

# Usage: compareTrackNcl.py FILE1 FILE2, with files of lines "Track <id> NCl <ncl>" (from --DEBUGmergerDumpTrackNCl).
# Prints the ids of tracks with less than half the clusters in FILE1 compared to FILE2 as DEBUG_TRACK="...", to be used as environment variable with GPUCA_CADEBUG 2 in GPUTPCGMTrackParam.cxx, followed by some statistics.

import sys


def readNCl(fileName):
    ncl = {}
    with open(fileName) as f:
        for line in f:
            cols = line.split()
            if len(cols) >= 4 and cols[0] == "Track" and cols[2] == "NCl":
                ncl[int(cols[1])] = int(cols[3])
    return ncl


if len(sys.argv) != 3:
    sys.exit("Usage: " + sys.argv[0] + " FILE1 FILE2")
ncl1, ncl2 = readNCl(sys.argv[1]), readNCl(sys.argv[2])
common = [i for i in sorted(ncl1) if i in ncl2]
ids = [str(i) for i in common if 2 * ncl1[i] < ncl2[i]]
print('DEBUG_TRACK="' + "\n".join(",".join(ids[i:i + 30]) for i in range(0, len(ids), 30)) + '"')
print()  # Statistics as shell comments, so that the output can still be passed to eval
print("# Tracks in total (present in both files):", len(common))
for text, n in (("Tracks with 0 clusters in file 2", sum(1 for i in common if ncl2[i] == 0)),
                ("Tracks with 0 clusters in file 1 and > 0 in file 2", sum(1 for i in common if ncl1[i] == 0 and ncl2[i] > 0)),
                ("Tracks with NCl(file 1) < 0.5 * NCl(file 2)", len(ids))):
    print("# %s: %d (%.2f%%)" % (text, n, 100. * n / max(1, len(common))))
