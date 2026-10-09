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

"""Acceptance test 1 of 2: the OCCT symmetric-difference volume.

    volume(candidate - original) + volume(original - candidate) <= bandFactor * modelTolerance * area

A zero difference is also what a failed build or an empty cut gives, so three guards refuse a false
accept: both cuts must report `IsDone()`, the original's volume and area must be positive, and the
candidate's volume must be positive and within a loose factor of the original's.
"""

import math

_BAND_FACTOR = 1.0
# A candidate whose volume is off by more than this factor is a recogniser bug, not a near-miss.
_SANITY_VOLUME_RATIO = 4.0


def model_tolerance_cm(shape):
    """The largest tolerance over the shape's faces, edges and vertices, in the shape's own unit.

    It lives here because `recognise.py` needs it and must not import the emitter.
    """
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    worst = 0.0
    for kind, getter in ((TopAbs_FACE, lambda s: BRep_Tool.Tolerance(topods.Face(s))),
                         (TopAbs_EDGE, lambda s: BRep_Tool.Tolerance(topods.Edge(s))),
                         (TopAbs_VERTEX, lambda s: BRep_Tool.Tolerance(topods.Vertex(s)))):
        walk = TopExp_Explorer(shape, kind)
        while walk.More():
            worst = max(worst, getter(walk.Current()))
            walk.Next()
    return worst


def contains_disagreements(original, cand_shape, model_tol, n_points=4000, seed=1234):
    """Classify points against both solids; `(disagreements, scored, worst distance)`.

    It tells an empty cut from an equal pair, which the symmetric difference cannot. Points within
    `model_tol` of either boundary are skipped; `worst` is the farthest disagreement from it.
    """
    import random
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
    from OCC.Core.gp import gp_Pnt
    from cadsupport.recognise import _point_to_shape_distance

    # Distances go to the original's faces: against the solid, an interior point would read 0.
    boundary = _faces_of(original)
    box = _bbox(original)
    if box is None:
        return 0, 0, 0.0
    xmin, ymin, zmin, xmax, ymax, zmax = box
    pad = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    tol = max(model_tol, 1.0e-9)
    original_cls = BRepClass3d_SolidClassifier(original)
    candidate_cls = BRepClass3d_SolidClassifier(cand_shape)
    rng = random.Random(seed)
    disagreements = scored = 0
    worst = 0.0
    for _ in range(n_points):
        point = (rng.uniform(xmin - pad, xmax + pad), rng.uniform(ymin - pad, ymax + pad),
                 rng.uniform(zmin - pad, zmax + pad))
        gp = gp_Pnt(*point)
        original_cls.Perform(gp, tol)
        if original_cls.State() == TopAbs_ON:
            continue
        candidate_cls.Perform(gp, tol)
        if candidate_cls.State() == TopAbs_ON:
            continue
        scored += 1
        if (original_cls.State() == TopAbs_IN) != (candidate_cls.State() == TopAbs_IN):
            disagreements += 1
            if disagreements <= _WORST_DISTANCE_SAMPLES:
                distance = _point_to_shape_distance(point, boundary)
                if distance == distance and distance != float("inf"):
                    worst = max(worst, distance)
    return disagreements, scored, worst


# `worst` is a reporting number, so it is measured on the first few disagreements rather than on
# all of them; a part that disagrees hundreds of times has already declined.
_WORST_DISTANCE_SAMPLES = 24


def _faces_of(shape):
    """The shape's faces as one compound: its boundary, as something to measure a distance to."""
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Compound, topods
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    walk = TopExp_Explorer(shape, TopAbs_FACE)
    while walk.More():
        builder.Add(compound, topods.Face(walk.Current()))
        walk.Next()
    return compound


def _bbox(shape):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    box.SetGap(0.0)
    try:
        return box.Get()
    except Exception:                                            # noqa: BLE001
        return None


def _props(shape):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    vol = GProp_GProps()
    brepgprop.VolumeProperties(shape, vol)
    surf = GProp_GProps()
    brepgprop.SurfaceProperties(shape, surf)
    return vol.Mass(), surf.Mass()


def _cut_volume(a, b, what):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    op = BRepAlgoAPI_Cut(a, b)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"BRepAlgoAPI_Cut failed ({what})")
    volume, _area = _props(op.Shape())
    return abs(volume)


def symmetric_difference(original, cand_shape, model_tolerance_cm, band_factor=_BAND_FACTOR,
                         original_props=None):
    """Measure `original` against `cand_shape`; returns a dict, never raises on a mere mismatch.

    `original_props` is `_props(original)` when the caller already has it.
    """
    v_orig, a_orig = _props(original) if original_props is None else original_props
    if not (v_orig > 0.0 and a_orig > 0.0):
        return {"accepted": False,
                "reason": f"original has non-positive volume/area ({v_orig:.6g}/{a_orig:.6g})"}
    v_cand, a_cand = _props(cand_shape)
    if not v_cand > 0.0:
        return {"accepted": False, "volumeOriginal": v_orig, "volumeCandidate": v_cand,
                "reason": f"candidate has non-positive volume ({v_cand:.6g})"}
    if not (1.0 / _SANITY_VOLUME_RATIO <= v_cand / v_orig <= _SANITY_VOLUME_RATIO):
        return {"accepted": False, "volumeOriginal": v_orig, "volumeCandidate": v_cand,
                "reason": f"candidate volume {v_cand:.6g} is not comparable to the original's "
                          f"{v_orig:.6g}"}
    extra = _cut_volume(cand_shape, original, "candidate - original")
    missing = _cut_volume(original, cand_shape, "original - candidate")
    dv = extra + missing
    band = band_factor * model_tolerance_cm * a_orig
    return {
        "accepted": dv <= band,
        "volumeOriginal": v_orig,
        "volumeCandidate": v_cand,
        "areaOriginal": a_orig,
        "extraVolume": extra,
        "missingVolume": missing,
        "symmetricDifference": dv,
        "band": band,
        "modelToleranceCm": model_tolerance_cm,
        "relativeToVolume": dv / v_orig,
        "reason": None if dv <= band else
                  f"symmetric difference {dv:.6g} cm^3 exceeds the band {band:.6g} cm^3 "
                  f"(= {model_tolerance_cm:.3g} cm x area {a_orig:.6g} cm^2); "
                  f"extra {extra:.6g}, missing {missing:.6g}",
    }


# ------------------------------------------------------------------------------------------
# self-test
# ------------------------------------------------------------------------------------------

def self_test(verbose=True):
    """Hand-built pairs whose verdict is known, including candidates that must be rejected."""
    from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder,
                                      BRepPrimAPI_MakeSphere)
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
    from cadsupport import primitives as prim

    checks = []

    def check(name, condition, detail=""):
        checks.append((name, bool(condition), detail))
        if verbose:
            print(f"  [{'ok ' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    tol = 1.0e-7

    # 1. positive control: a shape against itself.
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 3.0, 4.0).Shape()
    r = symmetric_difference(box, box, tol)
    check("a box against itself is accepted", r["accepted"], f"dV={r['symmetricDifference']:.3g}")
    check("a box against itself has zero symmetric difference", r["symmetricDifference"] == 0.0)

    # 2. positive control through the description, which is how the pipeline uses it: an OCCT box
    #    built independently of the description must still match.
    cand = prim.candidate("primitive", [prim.leaf(
        "TGeoBBox", {"dx": 1.0, "dy": 1.5, "dz": 2.0},
        prim.identity_frame((1.0, 1.5, 2.0)))], "self-test")
    r = symmetric_difference(box, prim.build_occ(cand), tol)
    check("an independently built TGeoBBox description matches the box", r["accepted"],
          f"dV={r['symmetricDifference']:.3g} band={r['band']:.3g}")

    # 3. negative control: the same box 1 micron (1e-4 cm) too long.
    long_box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 3.0, 4.0001).Shape()
    r = symmetric_difference(box, long_box, tol)
    check("a box 1e-4 cm too long is rejected", not r["accepted"],
          f"dV={r['symmetricDifference']:.3g} band={r['band']:.3g}")

    # 3b. how fine is the knife? A displacement of exactly one model tolerance must sit at the
    #     band, and ten of them must be outside it.
    for factor, want_accept in ((0.5, True), (10.0, False)):
        nudged = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 3.0, 4.0 + factor * tol).Shape()
        r = symmetric_difference(box, nudged, tol)
        check(f"a box {factor}x the model tolerance too long is "
              f"{'accepted' if want_accept else 'rejected'}",
              r["accepted"] == want_accept,
              f"dV={r['symmetricDifference']:.3g} band={r['band']:.3g}")

    # 4. negative control: right volume, wrong shape (a volume comparison alone would pass it).
    v = 2.0 * 3.0 * 4.0
    rad = (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)
    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(1.0, 1.5, 2.0), rad).Shape()
    r = symmetric_difference(box, sphere, tol)
    check("a sphere of equal volume is rejected", not r["accepted"],
          f"dV={r['symmetricDifference']:.3g}, volumes {r['volumeOriginal']:.4f} vs "
          f"{r['volumeCandidate']:.4f}")

    # 5. a tube, built two ways: OCCT cut versus the description's TGeoTube.
    outer = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -5), gp_Dir(0, 0, 1)), 2.0, 10.0).Shape()
    inner = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -6), gp_Dir(0, 0, 1)), 1.0, 12.0).Shape()
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    op = BRepAlgoAPI_Cut(outer, inner)
    op.Build()
    tube = op.Shape()
    cand = prim.candidate("primitive", [prim.leaf(
        "TGeoTube", {"rmin": 1.0, "rmax": 2.0, "dz": 5.0}, prim.identity_frame())], "self-test")
    r = symmetric_difference(tube, prim.build_occ(cand), tol)
    check("a TGeoTube description matches an OCCT-cut tube", r["accepted"],
          f"dV={r['symmetricDifference']:.3g} band={r['band']:.3g}")

    # 6. negative control on the tube: a solid cylinder must not pass as the tube.
    solid_cand = prim.candidate("primitive", [prim.leaf(
        "TGeoTube", {"rmin": 0.0, "rmax": 2.0, "dz": 5.0}, prim.identity_frame())], "self-test")
    r = symmetric_difference(tube, prim.build_occ(solid_cand), tol)
    check("a solid cylinder is rejected as the tube", not r["accepted"],
          f"dV={r['symmetricDifference']:.3g}")

    # 7. the tube in a rotated, translated frame -- the case the whole frame machinery exists for.
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Vec
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 0)), 0.7)
    shift = gp_Trsf()
    shift.SetTranslation(gp_Vec(3.0, -4.0, 5.0))
    moved = BRepBuilderAPI_Transform(tube, shift.Multiplied(trsf), True).Shape()
    zaxis = _rotated((0.0, 0.0, 1.0), (1.0, 1.0, 0.0), 0.7)
    xaxis = _rotated((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), 0.7)
    frame = prim.frame_from_axis((3.0, -4.0, 5.0), zaxis, xaxis)
    cand = prim.candidate("primitive", [prim.leaf(
        "TGeoTube", {"rmin": 1.0, "rmax": 2.0, "dz": 5.0}, frame)], "self-test")
    r = symmetric_difference(moved, prim.build_occ(cand), tol)
    check("a rotated, translated tube matches its placed description", r["accepted"],
          f"dV={r['symmetricDifference']:.3g} band={r['band']:.3g}")

    # 8. negative control on the frame: the *unrotated* description must be rejected against it.
    flat = prim.candidate("primitive", [prim.leaf(
        "TGeoTube", {"rmin": 1.0, "rmax": 2.0, "dz": 5.0},
        prim.identity_frame((3.0, -4.0, 5.0)))], "self-test")
    r = symmetric_difference(moved, prim.build_occ(flat), tol)
    check("the same tube without the rotation is rejected", not r["accepted"],
          f"dV={r['symmetricDifference']:.3g}")

    # --- the containment corroboration, checked as an instrument before it is trusted ---
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 4.0, 3.0, 2.0).Shape()
    same, scored, worst = contains_disagreements(box, BRepPrimAPI_MakeBox(
        gp_Pnt(0, 0, 0), 4.0, 3.0, 2.0).Shape(), 1.0e-7)
    check("the containment corroboration reports no disagreement for an identical pair",
          same == 0 and scored > 3000, f"{same} of {scored} scored")
    for grow, want_worst in ((0.02, 0.02), (0.2, 0.2)):
        bigger = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 4.0 + grow, 3.0, 2.0).Shape()
        n_bad, n_scored, far = contains_disagreements(box, bigger, 1.0e-7)
        check(f"the containment corroboration sees a face displaced by {grow} cm",
              n_bad > 0 and abs(far - want_worst) <= 0.1 * want_worst,
              f"{n_bad} of {n_scored} scored, the farthest {far:.4g} cm from the boundary "
              f"(the slab is {want_worst} cm thick)")
    # The mirror case: a missing slab is measured to the original's faces, not to its solid.
    smaller = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 4.0 - 0.2, 3.0, 2.0).Shape()
    n_bad, n_scored, far = contains_disagreements(box, smaller, 1.0e-7)
    check("the containment corroboration measures a MISSING slab at its true size, not at zero",
          n_bad > 0 and 0.5 * 0.2 <= far <= 1.05 * 0.2,
          f"{n_bad} of {n_scored} scored, the farthest {far:.4g} cm from the boundary "
          f"(the missing slab is 0.2 cm thick; measured against the solid instead of its faces "
          f"this number would be 0)")

    # A sub-tolerance difference must not be reported.
    hair = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 4.0 + 1.0e-9, 3.0, 2.0).Shape()
    n_bad, n_scored, _far = contains_disagreements(box, hair, 1.0e-7)
    check("the containment corroboration does not cry wolf on a sub-tolerance difference",
          n_bad == 0, f"{n_bad} of {n_scored} scored")

    n_ok = sum(1 for _n, ok, _d in checks if ok)
    if verbose:
        print(f"  {n_ok}/{len(checks)} acceptance self-checks passed")
    return n_ok, len(checks)


def _rotated(vec, axis, angle):
    """Rodrigues; used only by the self-test, to state the expected frame independently."""
    from cadsupport.primitives import _cross, _dot, _scale, _unit, _add
    k = _unit(axis)
    return _add(_add(_scale(vec, math.cos(angle)), _scale(_cross(k, vec), math.sin(angle))),
                _scale(k, _dot(k, vec) * (1.0 - math.cos(angle))))
