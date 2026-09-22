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
# Since: 2026-07

"""OpenCascade reference oracle for the exact-surface solid (O2BVHSurfaceSolid).

It answers the kernel's questions about the *same* BREP the converter read, with explicit tolerance
semantics: far too slow to navigate with (milliseconds per query, not thread-safe), fine for an
oracle.


What it answers
---------------
For a solid loaded from a `.brep` file (in cm; written by `O2_CADtoTGeo.py --dump-brep`) and a
sample set dumped by the C++ harness (`o2-bench-cadsupport-solid-harness --dump-samples`):

  contains          BRepClass3d_SolidClassifier      1 = inside, 0 = outside, -1 = ON (no verdict)
  distFromOutside   IntCurvesFace_ShapeIntersector   nearest positive ray/shell crossing
  distFromInside    IntCurvesFace_ShapeIntersector   same call; the origin is inside instead
  safetyUpperBound  BRepExtrema_DistShapeShape       true distance to the boundary
  capacity          BRepGProp                        exact volume
  tolerance         max BRep_Tool::Tolerance         the model's own declared ambiguity band

`distFromOutside` and `distFromInside` are deliberately the *same* computation: the nearest
positive intersection of the ray with the shell. Entering versus exiting is a property of where
the origin is, not of the intersector, so no face-orientation bookkeeping is needed and the
oracle cannot get it subtly wrong. The origin's own classification is reported alongside each
answer so the consumer can check that assumption rather than trust it.

Tolerance semantics (important when comparing)
----------------------------------------------
OCCT is a *tolerant* modeller: every face, edge and vertex carries its own tolerance, and a point
is ON when it is within that distance of the boundary. Imported CAD routinely carries 1e-5 cm or
worse. So the honest comparison rule is: a disagreement is only meaningful when the query point
is further than the model tolerance from the boundary. The oracle reports `tolerance` (the max
over the shape's sub-shapes) so the consumer can apply exactly that rule instead of inventing a
band.

Usage
-----
  answer a sample set:
    occtOracle.py --brep part.brep --samples samples.json --out answers.json
  check the oracle itself against closed-form geometry (no inputs needed):
    occtOracle.py --self-test

Environment: an interpreter that can import OCC, for example
  alienv setenv pythonOCC/latest -c python3 occtOracle.py ...
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.TopAbs import (TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_OUT,
                             TopAbs_SOLID, TopAbs_VERTEX)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.BRepTools import breptools
from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt

# The sample/answer JSON contract version. Bump together with the C++ harness writer/reader.
ORACLE_FORMAT_VERSION = 1

# A ray parameter this close to the origin is the origin itself, not a crossing. Matches the
# kernel's kRayTolerance so "starts exactly on a face" is treated the same way on both sides.
_RAY_EPS = 1.0e-9

# TGeoShape::Big(); the harness uses it for "no intersection".
_BIG = 1.0e30


# ----------------------------------------------------------------------------------------------
# Shape loading and interrogation
# ----------------------------------------------------------------------------------------------

def load_solid(path: Path):
    """Read a .brep file and return the single TopoDS_Solid it contains.

    A BREP file can hold a compound; the converter writes exactly one leaf solid per file, so
    anything else is a genuine inconsistency and must fail loudly rather than pick a shape.
    """
    shape = TopoDS_Shape_read(path)
    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(topods.Solid(explorer.Current()))
        explorer.Next()
    if len(solids) == 1:
        return solids[0]
    if not solids:
        raise RuntimeError(f"{path}: contains no TopoDS_Solid (a shell or compound of faces?)")
    raise RuntimeError(f"{path}: contains {len(solids)} solids, expected exactly one")


def TopoDS_Shape_read(path: Path):
    from OCC.Core.TopoDS import TopoDS_Shape
    shape = TopoDS_Shape()
    builder = BRep_Builder()
    if not breptools.Read(shape, str(path), builder):
        raise RuntimeError(f"{path}: BRepTools::Read failed")
    if shape.IsNull():
        raise RuntimeError(f"{path}: read a null shape")
    return shape


def shape_tolerance(shape) -> float:
    """Max BRep_Tool tolerance over faces, edges and vertices.

    This is the model's own statement about how well its boundary is defined, and therefore the
    only defensible width for a "no verdict" band when comparing against it.
    """
    worst = 0.0
    for shape_type, getter in ((TopAbs_FACE, lambda s: BRep_Tool.Tolerance(topods.Face(s))),
                               (TopAbs_EDGE, lambda s: BRep_Tool.Tolerance(topods.Edge(s))),
                               (TopAbs_VERTEX, lambda s: BRep_Tool.Tolerance(topods.Vertex(s)))):
        explorer = TopExp_Explorer(shape, shape_type)
        while explorer.More():
            worst = max(worst, getter(explorer.Current()))
            explorer.Next()
    return worst


def shape_bbox(shape):
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return [xmin, ymin, zmin], [xmax, ymax, zmax]


def count_subshapes(shape, shape_type) -> int:
    count = 0
    explorer = TopExp_Explorer(shape, shape_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _shells_of(solid):
    """The solid's boundary as a shape distances can be measured against.

    Returned as a compound so a solid with inner voids keeps all of its shells; measuring against
    the solid itself would report 0 for every interior point.
    """
    from OCC.Core.TopAbs import TopAbs_SHELL
    from OCC.Core.TopoDS import TopoDS_Compound
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    shells = 0
    explorer = TopExp_Explorer(solid, TopAbs_SHELL)
    while explorer.More():
        builder.Add(compound, explorer.Current())
        shells += 1
        explorer.Next()
    if shells == 0:
        raise RuntimeError("solid has no shell; cannot measure boundary distances")
    return compound


def volume_of(shape) -> float:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


# ----------------------------------------------------------------------------------------------
# The four query kernels
# ----------------------------------------------------------------------------------------------

class Oracle:
    """Stateful wrapper around the OCCT algorithms, so the expensive setup happens once.

    Not thread-safe -- OCCT classifiers and intersectors carry mutable state. That is fine here
    (an oracle run is a batch job) but it is exactly why this cannot be a navigation kernel.
    """

    def __init__(self, solid, classifier_tolerance: float = _RAY_EPS):
        self.solid = solid
        self.classifier_tolerance = classifier_tolerance
        self.classifier = BRepClass3d_SolidClassifier(solid)
        self.intersector = IntCurvesFace_ShapeIntersector()
        self.intersector.Load(solid, _RAY_EPS)
        # Distances are measured against the shells: a point inside a solid is 0 away from it.
        self.boundary = _shells_of(solid)

    def contains(self, point) -> int:
        """1 = inside, 0 = outside, -1 = ON the boundary within the classifier tolerance."""
        self.classifier.Perform(gp_Pnt(*point), self.classifier_tolerance)
        state = self.classifier.State()
        if state == TopAbs_IN:
            return 1
        if state == TopAbs_OUT:
            return 0
        if state == TopAbs_ON:
            return -1
        raise RuntimeError(f"unexpected classifier state {state} at {point}")

    def nearest_crossing(self, origin, direction) -> float:
        """Nearest strictly-positive ray/shell crossing, or _BIG when the ray misses.

        This is the answer to *both* DistFromOutside and DistFromInside: whether the crossing is
        an entry or an exit is decided by where the origin lies, not by this computation.
        """
        norm = math.sqrt(sum(component * component for component in direction))
        if norm <= 0.0:
            raise ValueError(f"degenerate ray direction {direction}")
        unit = [component / norm for component in direction]
        line = gp_Lin(gp_Pnt(*origin), gp_Dir(*unit))
        self.intersector.Perform(line, _RAY_EPS, _BIG)
        if not self.intersector.IsDone() or self.intersector.NbPnt() == 0:
            return _BIG
        best = _BIG
        for index in range(1, self.intersector.NbPnt() + 1):
            parameter = self.intersector.WParameter(index)
            if parameter > _RAY_EPS:
                best = min(best, parameter)
        return best

    def distance_to_boundary(self, point) -> float:
        """True distance from a point to the solid's boundary (its shell), always >= 0.

        This is the upper bound a correct Safety() must not exceed, for points inside *and*
        outside: BRepExtrema measures against the faces, not against the solid's interior.
        """
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
        extrema = BRepExtrema_DistShapeShape(vertex, self.boundary)
        if not extrema.IsDone():
            raise RuntimeError(f"BRepExtrema failed at {point}")
        return extrema.Value()


# ----------------------------------------------------------------------------------------------
# Sample-set driving
# ----------------------------------------------------------------------------------------------

def answer_samples(oracle: Oracle, samples: dict, distance_limit: int, verbose: bool) -> dict:
    """Answer every point and ray in `samples`, following the harness's category names."""
    answers = {"contains": {}, "originContains": {}, "distFromOutside": {},
               "distFromInside": {}, "safetyUpperBound": {}}
    timing = {}

    point_categories = samples.get("points", {})
    for category, points in point_categories.items():
        start = time.monotonic()
        answers["contains"][category] = [oracle.contains(p) for p in points]
        timing[f"contains/{category}"] = time.monotonic() - start
        if verbose:
            print(f"  contains/{category}: {len(points)} points "
                  f"({timing[f'contains/{category}']:.1f} s)", flush=True)

        # The exact distance is the costliest query, so it is capped, and the count is reported.
        limited = points if distance_limit <= 0 else points[:distance_limit]
        start = time.monotonic()
        answers["safetyUpperBound"][category] = [oracle.distance_to_boundary(p) for p in limited]
        timing[f"safety/{category}"] = time.monotonic() - start
        if verbose:
            print(f"  safetyUpperBound/{category}: {len(limited)}/{len(points)} points "
                  f"({timing[f'safety/{category}']:.1f} s)", flush=True)

    ray_categories = samples.get("rays", {})
    for category, rays in ray_categories.items():
        start = time.monotonic()
        distances = []
        origin_states = []
        for ray in rays:
            origin, direction = ray["o"], ray["d"]
            origin_states.append(oracle.contains(origin))
            distances.append(oracle.nearest_crossing(origin, direction))
        # One column per category; which TGeo entry point it corresponds to is decided by the
        # origin state, which is reported next to it rather than assumed from the category name.
        target = "distFromInside" if category.startswith("inside") else "distFromOutside"
        answers[target][category] = distances
        answers["originContains"][category] = origin_states
        timing[f"{target}/{category}"] = time.monotonic() - start
        if verbose:
            inside_count = sum(1 for s in origin_states if s == 1)
            print(f"  {target}/{category}: {len(rays)} rays, {inside_count} origins inside "
                  f"({timing[f'{target}/{category}']:.1f} s)", flush=True)

    answers["timingSeconds"] = timing
    return answers


def build_answer_document(brep_path: Path, samples: dict, distance_limit: int,
                          verbose: bool) -> dict:
    solid = load_solid(brep_path)
    analyzer = BRepCheck_Analyzer(solid)
    bbox_min, bbox_max = shape_bbox(solid)
    document = {
        "version": ORACLE_FORMAT_VERSION,
        "oracle": "OpenCascade",
        "brep": str(brep_path),
        "part": samples.get("part"),
        "valid": bool(analyzer.IsValid()),
        "tolerance": shape_tolerance(solid),
        "capacity": volume_of(solid),
        "nFaces": count_subshapes(solid, TopAbs_FACE),
        "nEdges": count_subshapes(solid, TopAbs_EDGE),
        "bboxMin": bbox_min,
        "bboxMax": bbox_max,
        "distanceLimit": distance_limit,
    }
    if verbose:
        print(f"{brep_path.name}: valid={document['valid']} tolerance={document['tolerance']:.3e} "
              f"volume={document['capacity']:.6g} cm^3 faces={document['nFaces']}", flush=True)
    if not document["valid"]:
        # Not fatal, but a broken reference must never pass unnoticed into a comparison.
        print(f"WARNING: {brep_path} is not BRepCheck-valid; its answers are not authoritative",
              file=sys.stderr)
    oracle = Oracle(solid)
    document.update(answer_samples(oracle, samples, distance_limit, verbose))
    return document


# ----------------------------------------------------------------------------------------------
# Self-test: the oracle must be checked before anything is judged by it
# ----------------------------------------------------------------------------------------------

def self_test() -> int:
    """Check every kernel against closed-form answers; needs no input files, so it can gate CI."""
    failures = []

    def check(name, got, expected, tolerance):
        deviation = abs(got - expected)
        ok = deviation <= tolerance
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: got {got:.12g}, expected {expected:.12g} "
              f"(dev {deviation:.3g}, tol {tolerance:g})")
        if not ok:
            failures.append(name)

    def check_int(name, got, expected):
        ok = got == expected
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: got {got}, expected {expected}")
        if not ok:
            failures.append(name)

    print("Self-test 1: axis-aligned box 2 x 4 x 6 at the origin corner")
    box = BRepPrimAPI_MakeBox(2.0, 4.0, 6.0).Solid()
    oracle = Oracle(box)
    check("box volume", volume_of(box), 48.0, 1e-9)
    check_int("box contains centre", oracle.contains([1.0, 2.0, 3.0]), 1)
    check_int("box contains outside point", oracle.contains([5.0, 2.0, 3.0]), 0)
    check_int("box contains face point", oracle.contains([0.0, 2.0, 3.0]), -1)
    # A ray from outside along +x through the centre: enters at x=0, so the distance is 3.
    check("box distance from outside", oracle.nearest_crossing([-3.0, 2.0, 3.0], [1.0, 0.0, 0.0]),
          3.0, 1e-9)
    # The same ray started inside at x=1 exits at x=2.
    check("box distance from inside", oracle.nearest_crossing([1.0, 2.0, 3.0], [1.0, 0.0, 0.0]),
          1.0, 1e-9)
    check("box ray miss", oracle.nearest_crossing([-3.0, 20.0, 3.0], [1.0, 0.0, 0.0]), _BIG, 0.0)
    # Nearest face from an interior point at (1,2,3) is x=0 or x=2, both 1 away.
    check("box distance to boundary (inside)", oracle.distance_to_boundary([1.0, 2.0, 3.0]),
          1.0, 1e-9)
    check("box distance to boundary (outside)", oracle.distance_to_boundary([5.0, 2.0, 3.0]),
          3.0, 1e-9)

    print("Self-test 2: cylinder r=3 h=10 along +z from the origin")
    cylinder = BRepPrimAPI_MakeCylinder(3.0, 10.0).Solid()
    oracle = Oracle(cylinder)
    check("cylinder volume", volume_of(cylinder), math.pi * 9.0 * 10.0, 1e-6)
    check_int("cylinder contains axis point", oracle.contains([0.0, 0.0, 5.0]), 1)
    check_int("cylinder contains outside point", oracle.contains([4.0, 0.0, 5.0]), 0)
    # Radial ray from outside enters the curved wall at r=3.
    check("cylinder radial entry", oracle.nearest_crossing([10.0, 0.0, 5.0], [-1.0, 0.0, 0.0]),
          7.0, 1e-9)
    # From the axis outwards, the exit is the wall at r=3.
    check("cylinder radial exit", oracle.nearest_crossing([0.0, 0.0, 5.0], [1.0, 0.0, 0.0]),
          3.0, 1e-9)
    # A ray exactly tangent to the wall must not be reported as a crossing at a shorter distance
    # than the cap it actually reaches; this is the configuration that breaks naive intersectors.
    tangent = oracle.nearest_crossing([3.0, -10.0, 5.0], [0.0, 1.0, 0.0])
    print(f"  [info] cylinder tangent ray -> {tangent:.12g} "
          f"({'grazes' if tangent < _BIG else 'misses'}; either is defensible)")
    check("cylinder distance to boundary on axis", oracle.distance_to_boundary([0.0, 0.0, 5.0]),
          3.0, 1e-9)

    print("Self-test 3: box with a drilled hole (a boundary that is not convex)")
    plate = BRepPrimAPI_MakeBox(gp_Pnt(-5.0, -5.0, 0.0), 10.0, 10.0, 2.0).Solid()
    drill = BRepPrimAPI_MakeCylinder(2.0, 10.0).Solid()
    drilled = BRepAlgoAPI_Cut(plate, drill).Shape()
    solid = load_solid_from_shape(drilled)
    oracle = Oracle(solid)
    check("drilled plate volume", volume_of(solid), 10.0 * 10.0 * 2.0 - math.pi * 4.0 * 2.0, 1e-6)
    check_int("hole centre is outside the material", oracle.contains([0.0, 0.0, 1.0]), 0)
    check_int("material point is inside", oracle.contains([4.0, 0.0, 1.0]), 1)
    # Crossing the plate through the hole: from x=-10 the first material is the hole wall at
    # x=-5 (the outer face), then the hole starts at x=-2.
    check("drilled plate first crossing", oracle.nearest_crossing([-10.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
          5.0, 1e-9)
    # Starting inside the hole, the nearest boundary going +x is the hole wall at x=2.
    check("crossing out of the hole", oracle.nearest_crossing([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
          2.0, 1e-9)

    print()
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("SELF-TEST PASSED: every kernel matches closed-form geometry")
    return 0


def load_solid_from_shape(shape):
    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(topods.Solid(explorer.Current()))
        explorer.Next()
    if len(solids) != 1:
        raise RuntimeError(f"expected exactly one solid, got {len(solids)}")
    return solids[0]


# ----------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brep", type=Path, help="solid to answer for, in cm")
    parser.add_argument("--samples", type=Path, help="sample set dumped by the C++ harness")
    parser.add_argument("--out", type=Path, help="where to write the answers JSON")
    parser.add_argument("--distance-limit", type=int, default=2000,
                        help="max points per category to compute the exact boundary distance for "
                             "(the most expensive query); 0 means no limit (default: 2000)")
    parser.add_argument("--self-test", action="store_true",
                        help="validate the oracle's own kernels against closed-form geometry")
    parser.add_argument("--quiet", action="store_true", help="suppress per-category progress")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not (args.brep and args.samples and args.out):
        parser.error("--brep, --samples and --out are required unless --self-test is given")

    samples = json.loads(args.samples.read_text())
    version = samples.get("version")
    if version != ORACLE_FORMAT_VERSION:
        raise RuntimeError(f"{args.samples}: sample format version {version}, "
                           f"this oracle speaks {ORACLE_FORMAT_VERSION}")

    document = build_answer_document(args.brep, samples, args.distance_limit, not args.quiet)
    args.out.write_text(json.dumps(document, indent=1))
    if not args.quiet:
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
