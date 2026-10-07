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

"""Write the o2-sim external-geometry and detector-list JSON for one representation.

The ExcavatorArm model is anchored to `barrel`, which Cave.cxx places at cave (0,-30,0), and is authored
with its long axis along the CAD *y* axis. TGeoCombiTrans::RotateX(+90) maps local +y -> master +z,
so rotation_deg [90,0,0] puts that axis on the beam.

Usage: make_configs.py <conv-root> <exact|tess|coarse> <output-dir>
"""
import json
import os
import sys

# The barrel-frame placement that puts ExcavatorArm's bounding-box centre at ALICE (100, 0, 0), inside the
# barrel and close enough to the origin that a box generator reaches it.
PLACEMENT = {"translation": [120.928, 102.064, 20.327], "rotation_deg": [90.0, 0.0, 0.0]}


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    conv_root, rep, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    macro = os.path.abspath(os.path.join(conv_root, "conv", f"excavator_arm_{rep}", "geom.C"))
    if not os.path.exists(macro):
        print(f"missing macro {macro}")
        return 1

    ext = {
        "externalDetectors": [
            {
                "name": "BAGR",
                "title": f"Excavator, Bucket sensitive ({rep})",
                "macro": macro,
                "anchor": "barrel",
                "detID": "FOC",
                # Substring match: this selects Bucket, BucketLink1, BucketLink2,
                # BucketCylinderInner and BucketCylinderOuter -- the whole bucket group.
                "sensitiveVolumes": ["Bucket"],
                "placement": PLACEMENT,
            },
        ]
    }
    detlist = {"EXTCAD": ["BAGR"]}

    with open(os.path.join(outdir, "externalGeometry.json"), "w") as f:
        json.dump(ext, f, indent=2)
        f.write("\n")
    with open(os.path.join(outdir, "detectorlist.json"), "w") as f:
        json.dump(detlist, f, indent=2)
        f.write("\n")
    print(f"wrote {outdir}/externalGeometry.json and {outdir}/detectorlist.json ({rep})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
