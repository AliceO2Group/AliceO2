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

"""Map a frozen harness sample set through the same transform that was applied to the shape.

The position/scale sweep needs the transformed run to ask the same questions as the baseline, so
the baseline's samples are pushed through the same `gp_Trsf` and handed over with
`runOracleGate.py --load-samples`.

  * **Ray origins are points, ray directions are directions.** Under a uniform scaling a point is
    multiplied by the factor and a unit direction is not -- transforming both as points would
    denormalise every direction and silently change what `DistFromOutside` was asked. `gp_Pnt` and
    `gp_Dir` are used respectively, so OCCT applies the right one of the two.
  * **The spec is in millimetres**, exactly as `make_boolean_fixtures.py --transform` takes it, and
    is converted here. The STEP fixtures are written in mm; the sidecars, meshes, `.brep` files and
    therefore the sample sets are all in cm (the converter scales by `step_unit_scale_to_cm`).
    Taking the same string on both sides removes the one arithmetic step where the two could
    silently disagree.

Usage
-----
  transformSamples.py --in <base>/oracle --out /tmp/samples_z400 --transform translate:0,0,4000
  transformSamples.py --in <base>/oracle --out /tmp/samples_x10  --transform scale:10

Requires the pythonOCC environment; OCCT's own transform is used, not a reimplementation.
"""

import argparse
import json
import sys
from pathlib import Path

from OCC.Core.gp import gp_Dir, gp_Pnt

from make_boolean_fixtures import parse_transform  # noqa: E402

# mm -> cm as a divisor, which gives back exactly 400 for 4000 where a 0.1 factor does not.
MM_PER_CM = 10.0


def transform_point(trsf, xyz):
    p = gp_Pnt(*xyz)
    p.Transform(trsf)
    return [p.X(), p.Y(), p.Z()]


def transform_direction(trsf, xyz):
    d = gp_Dir(*xyz)
    d.Transform(trsf)
    return [d.X(), d.Y(), d.Z()]


def transform_samples(doc, trsf):
    out = dict(doc)
    out["bboxMin"] = transform_point(trsf, doc["bboxMin"])
    out["bboxMax"] = transform_point(trsf, doc["bboxMax"])
    # Only a negative scale factor could swap min and max, and parse_transform rejects it.
    out["points"] = {category: [transform_point(trsf, p) for p in points]
                     for category, points in doc["points"].items()}
    out["rays"] = {category: [{"o": transform_point(trsf, r["o"]),
                               "d": transform_direction(trsf, r["d"])} for r in rays]
                   for category, rays in doc["rays"].items()}
    return out


def to_cm(trsf_spec: str) -> str:
    """Re-express a millimetre transform spec in centimetres. Scalings are unit-free."""
    if ";" in trsf_spec:
        return ";".join(to_cm(part) for part in trsf_spec.split(";") if part.strip())
    kind, _, rest = trsf_spec.partition(":")
    if kind.strip().lower() == "translate":
        parts = [float(v) / MM_PER_CM for v in rest.split(",")]
        return f"translate:{parts[0]!r},{parts[1]!r},{parts[2]!r}"
    return trsf_spec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True, type=Path,
                    help="directory holding the baseline samples_<part>.json files "
                         "(a gate run's <workdir>/oracle)")
    ap.add_argument("--out", required=True, type=Path,
                    help="directory to write the transformed sample sets into")
    ap.add_argument("--transform", required=True,
                    help="the same spec given to make_boolean_fixtures.py --transform; lengths in "
                         "MILLIMETRES and converted to cm here")
    args = ap.parse_args()

    cm_spec = to_cm(args.transform)
    trsf, _volume_scale, description = parse_transform(cm_spec)
    args.out.mkdir(parents=True, exist_ok=True)

    sources = sorted(args.indir.glob("samples_*.json"))
    if not sources:
        raise SystemExit(f"no samples_*.json in {args.indir}")
    print(f"Transforming {len(sources)} sample set(s) by {description} (cm) "
          f"[spec {args.transform} in mm]")
    for source in sources:
        doc = json.loads(source.read_text())
        (args.out / source.name).write_text(json.dumps(transform_samples(doc, trsf), indent=1))
        print(f"  {source.name}")
    print(f"Wrote {len(sources)} file(s) into {args.out}")


if __name__ == "__main__":
    main()
