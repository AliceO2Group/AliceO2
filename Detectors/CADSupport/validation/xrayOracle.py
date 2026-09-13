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

"""Ground truth for the X-ray transport benchmark: the ORDERED CROSSING LIST along each ray.

Companion to `Detectors/CADSupport/test/runXRayBenchmark.cxx`.
Reads a ray file written by `o2-bench-cadsupport-xray --dump-rays` and answers exactly those
rays from the part's `.brep`, in OpenCascade.

`IntCurvesFace_ShapeIntersector` returns every crossing along a ray in one call, so one OCCT call
answers a whole transport.

How a crossing list is decided
------------------------------
The raw intersections are *face* hits, not *solid* transitions (a shared edge is hit twice, a
tangent cylinder twice without being entered). So they are only CANDIDATE positions: the MIDPOINT
of every interval between consecutive candidates is classified with `BRepClass3d_SolidClassifier`,
and only positions where the classification changes are kept, which alternates enter/exit by
construction.

A midpoint the classifier calls ON is a position where OCCT itself has no answer; the ray is
flagged `amb` and the benchmark excludes it rather than scoring it either way.

The same pass yields the OCCT chord integral (the summed inside-segment length times the raster
cell area), which is the ground-truth column for the benchmark's volume-by-chord-integration.

Usage
-----
  xrayOracle.py --brep <part>.brep --rays <dir>/xrays_<part>.json \\
                --out <dir>/crossings_<part>.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path


from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT
from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt

from occtOracle import load_solid, shape_tolerance, volume_of

# Must match kXRayFormatVersion in runXRayBenchmark.cxx.
XRAY_FORMAT_VERSION = 2

# A ray parameter this close to the origin is the origin itself, not a crossing. Same constant the
# kernel and occtOracle.py use.
_RAY_EPS = 1.0e-9


class CrossingOracle:
    """Stateful OCCT wrapper: the expensive setup happens once per part."""

    def __init__(self, solid, tolerance: float):
        self.solid = solid
        self.tolerance = tolerance
        self.intersector = IntCurvesFace_ShapeIntersector()
        self.intersector.Load(solid, _RAY_EPS)
        self.classifier = BRepClass3d_SolidClassifier(solid)
        # Candidate positions closer together than this are the same crossing seen through two
        # faces (a shared edge) or the same tangency seen twice. Never wider than the model's own
        # statement about how well its boundary is defined.
        self.merge_tolerance = max(tolerance, _RAY_EPS)

    def candidates(self, origin, direction, tmax):
        """Every face intersection parameter in (eps, tmax], sorted and merged."""
        line = gp_Lin(gp_Pnt(*origin), gp_Dir(*direction))
        self.intersector.Perform(line, _RAY_EPS, tmax)
        if not self.intersector.IsDone():
            return None
        raw = []
        for index in range(1, self.intersector.NbPnt() + 1):
            parameter = self.intersector.WParameter(index)
            if _RAY_EPS < parameter <= tmax:
                raw.append(parameter)
        raw.sort()
        merged = []
        for parameter in raw:
            if merged and parameter - merged[-1] <= self.merge_tolerance:
                continue
            merged.append(parameter)
        return merged

    def classify_at(self, origin, direction, t):
        point = gp_Pnt(*(origin[k] + t * direction[k] for k in range(3)))
        self.classifier.Perform(point, _RAY_EPS)
        state = self.classifier.State()
        if state == TopAbs_IN:
            return 1
        if state == TopAbs_OUT:
            return 0
        if state == TopAbs_ON:
            return -1
        raise RuntimeError(f"unexpected classifier state {state}")

    def crossings(self, origin, direction, tmax):
        """The ordered crossing list, the inside length, and whether OCCT declined anywhere.

        Returns (t_list, kind_list, inside_length, ambiguous, origin_state).
        """
        candidates = self.candidates(origin, direction, tmax)
        if candidates is None:
            return [], [], 0.0, True, -1
        edges = [0.0] + candidates + [tmax]
        states = []
        ambiguous = False
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if hi <= lo:
                states.append(states[-1] if states else 0)
                continue
            state = self.classify_at(origin, direction, 0.5 * (lo + hi))
            if state < 0:
                ambiguous = True
                state = states[-1] if states else 0
            states.append(state)
        ts, kinds = [], []
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                ts.append(edges[i])
                kinds.append(1 if states[i] == 1 else -1)
        inside = 0.0
        for i, state in enumerate(states):
            if state == 1:
                inside += edges[i + 1] - edges[i]
        return ts, kinds, inside, ambiguous, states[0] if states else 0


def answer(brep_path: Path, ray_doc: dict, verbose: bool) -> dict:
    if ray_doc.get("version") != XRAY_FORMAT_VERSION:
        raise RuntimeError(f"ray file speaks version {ray_doc.get('version')}, "
                           f"this oracle speaks {XRAY_FORMAT_VERSION}")
    solid = load_solid(brep_path)
    tolerance = shape_tolerance(solid)
    oracle = CrossingOracle(solid, tolerance)

    rays_out = []
    inside_by_beam = {}
    ambiguous_rays = 0
    total_crossings = 0
    started = time.time()
    for index, ray in enumerate(ray_doc["rays"]):
        origin = ray["o"]
        direction = ray["d"]
        norm = math.sqrt(sum(c * c for c in direction))
        unit = [c / norm for c in direction]
        ts, kinds, inside, ambiguous, origin_state = oracle.crossings(origin, unit, ray["tmax"])
        beam = ray["beam"]
        inside_by_beam[beam] = inside_by_beam.get(beam, 0.0) + inside
        ambiguous_rays += bool(ambiguous)
        total_crossings += len(ts)
        rays_out.append({"o": origin, "d": direction, "tmax": ray["tmax"], "beam": beam,
                         "t": ts, "k": kinds, "L": inside, "amb": bool(ambiguous),
                         "s": origin_state})
        if verbose and (index + 1) % 2000 == 0:
            print(f"    {index + 1}/{len(ray_doc['rays'])} rays "
                  f"({time.time() - started:.1f} s)", flush=True)

    # Each beam is an independent estimate of the same volume; the reported number is their mean
    # and the per-beam spread is the honest error bar.
    cell_area = ray_doc["cellArea"]
    labels = [b["label"] for b in ray_doc.get("beams", [])]
    per_beam = {}
    volumes = []
    for beam, length in sorted(inside_by_beam.items()):
        volume = length * cell_area[beam]
        per_beam[labels[beam] if beam < len(labels) else str(beam)] = volume
        volumes.append(volume)
    chord_volume = sum(volumes) / len(volumes) if volumes else 0.0

    document = dict(ray_doc)
    document["rays"] = rays_out
    document["tolerance"] = tolerance
    document["capacity"] = volume_of(solid)
    document["valid"] = bool(BRepCheck_Analyzer(solid).IsValid())
    document["volumeChord"] = chord_volume
    document["volumeChordPerBeam"] = per_beam
    document["ambiguousRays"] = ambiguous_rays
    document["totalCrossings"] = total_crossings
    document["oracleSeconds"] = time.time() - started
    return document


def self_test() -> int:
    """Analytic controls: a box, a hollow cylinder, and a sphere's chord integral.

    Every one of them has a closed-form answer, so this checks the oracle against something that
    is not another implementation of the same idea. Needs no .brep and no benchmark run.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder,
                                      BRepPrimAPI_MakeSphere)
    from occtOracle import load_solid_from_shape

    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if not ok else ""))
        if not ok:
            failures.append(name)

    # A 2 x 3 x 4 box with its corner at the origin: two crossings on a central x ray.
    box = load_solid_from_shape(BRepPrimAPI_MakeBox(2.0, 3.0, 4.0).Shape())
    oracle = CrossingOracle(box, shape_tolerance(box))
    ts, kinds, inside, amb, _ = oracle.crossings([-5.0, 1.5, 2.0], [1.0, 0.0, 0.0], 20.0)
    check("box: two crossings", len(ts) == 2, str(ts))
    check("box: at 5 and 7 cm", len(ts) == 2 and abs(ts[0] - 5.0) < 1e-9 and abs(ts[1] - 7.0) < 1e-9,
          str(ts))
    check("box: enter then exit", kinds == [1, -1], str(kinds))
    check("box: chord = 2 cm", abs(inside - 2.0) < 1e-9, str(inside))

    # A hollow cylinder: FOUR crossings along a diameter.
    outer = BRepPrimAPI_MakeCylinder(1.0, 4.0).Shape()
    inner = BRepPrimAPI_MakeCylinder(0.5, 6.0).Shape()
    tube = load_solid_from_shape(BRepAlgoAPI_Cut(outer, inner).Shape())
    oracle = CrossingOracle(tube, shape_tolerance(tube))
    ts, kinds, inside, amb, _ = oracle.crossings([-5.0, 0.0, 2.0], [1.0, 0.0, 0.0], 20.0)
    check("hollow cylinder: four crossings along a diameter", len(ts) == 4, str(ts))
    check("hollow cylinder: at 4.0 / 4.5 / 5.5 / 6.0",
          len(ts) == 4 and all(abs(a - b) < 1e-7 for a, b in zip(ts, [4.0, 4.5, 5.5, 6.0])), str(ts))
    check("hollow cylinder: in, out, in, out", kinds == [1, -1, 1, -1], str(kinds))

    # A ray grazing the outer wall tangentially must produce ZERO crossings, not two.
    ts, kinds, inside, amb, _ = oracle.crossings([-5.0, 1.0, 2.0], [1.0, 0.0, 0.0], 20.0)
    check("tangent ray: no crossings (the classifier overrules the intersector)",
          len(ts) == 0, f"{ts} {kinds}")

    # A sphere's chord integral against 4/3 pi r^3, on a structured raster: the volume instrument.
    sphere = load_solid_from_shape(BRepPrimAPI_MakeSphere(1.0).Shape())
    oracle = CrossingOracle(sphere, shape_tolerance(sphere))
    exact = 4.0 / 3.0 * math.pi
    for n in (16, 32):
        window = 1.02
        cell = (2 * window / n) ** 2
        total = 0.0
        for i in range(n):
            for j in range(n):
                x = -window + (i + 0.5) * 2 * window / n
                y = -window + (j + 0.5) * 2 * window / n
                _, _, inside, _, _ = oracle.crossings([x, y, -window], [0.0, 0.0, 1.0], 2 * window)
                total += inside
        volume = total * cell
        rel = abs(volume - exact) / exact
        print(f"        sphere r=1: raster {n:3d} x {n:3d} -> V = {volume:.8f}, "
              f"exact {exact:.8f}, relative {rel:.3e}")
        check(f"sphere chord integral converges at N={n}", rel < 5.0e-2 / n, f"rel={rel:.3e}")

    print(f"\n{'SELF-TEST PASSED' if not failures else 'SELF-TEST FAILED'}: "
          f"{len(failures)} failure(s)")
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brep", type=Path, help="the part's .brep (converter --dump-brep)")
    parser.add_argument("--rays", type=Path, help="xrays_<part>.json from --dump-rays")
    parser.add_argument("--out", type=Path, help="where to write crossings_<part>.json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="analytic controls (box, hollow cylinder, tangent ray, sphere "
                             "volume); needs no .brep and no benchmark run")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not (args.brep and args.rays and args.out):
        parser.error("--brep, --rays and --out are required (unless --self-test)")

    ray_doc = json.loads(args.rays.read_text())
    document = answer(args.brep, ray_doc, verbose=not args.quiet)
    args.out.write_text(json.dumps(document))
    if not args.quiet:
        print(f"  {args.out}: {len(document['rays'])} rays, {document['totalCrossings']} crossings, "
              f"{document['ambiguousRays']} ambiguous, chord volume "
              f"{document['volumeChord']:.8g} cm^3 vs OCCT capacity {document['capacity']:.8g} cm^3 "
              f"({document['oracleSeconds']:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
