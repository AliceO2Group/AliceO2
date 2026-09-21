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

"""The emitter self-test behind `python3 -m cadsupport.emit --self-test`: its fixtures and recorded candidates."""

import functools
import json
import math
from pathlib import Path

from cadsupport import accept, primitives as prim, recognise  # noqa: E402
from cadsupport.emit import (crosscheck_bbox, crosscheck_contains, process_solid,  # noqa: E402
                             write_shape_root)

# The recorded candidates: structure exactly, floats within a tolerance, on every platform.
_RECORDED_CANDIDATES = Path(__file__).with_name("emit_selftest_candidates.json")
# The emitted description (leaf parameters and frames, cm) differs between platforms by ~1e-15.
_DESCRIPTION_TOLERANCE = 1.0e-12
# The measured residues in `notes` (gaps and drifts near zero) differ by up to ~1e-9.
_NOTES_TOLERANCE = 1.0e-8

# The whole-part fixtures, recorded before the revolved matcher and the acceptance retry existed.
_WHOLE_PART_FIXTURES = (
    'box',
    'solid cylinder',
    'tube',
    'tube segment',
    'cone',
    'sphere',
    'placed tube',
    'rod-and-eye (two-cluster union)',
)

# The torus carrier and the elliptic cylinder; the two bellows shapes are what PIPE's plies reduce to.
_TORUS_ELTU_FIXTURES = (
    'solid torus',
    'torus shell (a bellows ply)',
    'hollow torus wedge',
    'half a bellows ply',
    'elliptic cylinder, a > b',
    'elliptic cylinder, a < b',
    'elliptic cylinder with equal semi-axes',
)

# The single cell.
_CELL_FIXTURES = (
    'Steinmetz solid (two cylinders intersected)',
    'tube with a transverse window',
    'cylinder cut by an oblique plane',
    'cube with an axial through-hole',
    'cylinder with a milled flat',
)

# The two-level DNF: the cells, their order and every leaf in them.
_UNION_OF_CELLS_FIXTURES = (
    'a cylinder with a hexagonal collar',
    'two rods sharing no edge',
    'a torus with a cylinder through it',
    'three disjoint boxes',
)

# Whole parts whose every carrier arrives Tier-0 canonicalised from a stored B-spline.
_TIER0_FIXTURES = (
    'NURBS-encoded box',
    'NURBS-encoded solid cylinder',
    'NURBS-encoded tube segment',
    'NURBS-encoded cone',
    'NURBS-encoded sphere',
    'NURBS-encoded solid torus',
    'NURBS-encoded hollow torus wedge',
    'NURBS-encoded cube with an axial through-hole',
)

# The prism family.
_PRISM_FIXTURES = (
    'L-shaped plate',
    'hollow 8-edge polygon (TGeoPgon)',
    'hollow 48-edge polygon (TGeoPgon)',
    'Trd1 (slanted x faces)',
    'Trd1 (taper reversed)',
    "Trd1 (TPC_IRB1's 0.5 % slant)",
    'Trd2 (both half-widths vary)',
    'Trd2 (isotropic taper, also a legal Xtru)',
    'Arb8 (parallelepiped)',
    "Arb8 (TPC_IHSTR's trapezoidal prism)",
    'Arb8 (sheared in x only)',
    "Arb8 (a TGeoTrap's eight corners)",
    'Xtru (non-convex L section)',
    "Xtru (ITS ConeARibVol0's eight-corner section)",
    'Xtru (a triangular section)',
    'Xtru (three sections, offset and scaled)',
    'Pgon (solid hexagonal prism)',
    'Pgon (tapered eight-edge prism)',
    'Pgon (hollow 8-edge prism)',
    'Pgon (hollow 48-edge prism)',
    "Pgon (TPC_Strip's thin 18-edge shell)",
    'Pgon (three hollow sections)',
    'Pgon (a 90 deg wedge closing on the axis)',
    'Pgon (a wedge across phi = 0)',
    'placed Trd1',
    'placed Xtru (non-convex L section)',
)


def _count_trusted_concave(solid):
    """Trusted concave or mixed edges of a solid, counted as `recognise._match_single_cell` does."""
    from cadsupport import census
    counts = census.edge_census(solid)
    return (counts["concave"] + counts["mixed"]
            - counts["concaveNearTangential"] - counts["mixedNearTangential"])


@functools.lru_cache(maxsize=None)
def _recorded_candidates():
    with open(_RECORDED_CANDIDATES) as f:
        return json.load(f)


def _candidate_differences(want, got, path=""):
    """Where `got` differs from the recorded `want`: structure exactly, floats within tolerance."""
    if isinstance(want, (bool, str)) or want is None or isinstance(got, (bool, str)) or got is None:
        return [] if type(want) is type(got) and want == got else [f"{path}: {got!r} != {want!r}"]
    if isinstance(want, dict) or isinstance(got, dict):
        if not (isinstance(want, dict) and isinstance(got, dict)) or sorted(want) != sorted(got):
            return [f"{path}: keys differ"]
        return [d for key in sorted(want) for d in _candidate_differences(want[key], got[key], f"{path}/{key}")]
    if isinstance(want, list) or isinstance(got, list):
        if not (isinstance(want, list) and isinstance(got, list)) or len(want) != len(got):
            return [f"{path}: lengths differ"]
        return [d for i, (w, g) in enumerate(zip(want, got)) for d in _candidate_differences(w, g, f"{path}[{i}]")]
    if type(want) is not type(got):
        return [f"{path}: {type(got).__name__} {got!r} != {type(want).__name__} {want!r}"]
    if isinstance(want, int) and isinstance(got, int):
        return [] if want == got else [f"{path}: {got} != {want}"]
    tolerance = _NOTES_TOLERANCE if path.startswith("/notes") else _DESCRIPTION_TOLERANCE
    if abs(got - want) <= tolerance * max(1.0, abs(want), abs(got)):
        return []
    return [f"{path}: {got!r} != {want!r}"]


def _recorded_match(fixtures, seen):
    """(ok, detail) for `fixtures` against their recorded candidates."""
    problems = []
    for name in fixtures:
        if name not in seen:
            problems.append(f"{name}: not converted")
            continue
        diffs = _candidate_differences(_recorded_candidates()[name], seen[name])
        if diffs:
            more = f" (+{len(diffs) - 3} more)" if len(diffs) > 3 else ""
            problems.append(f"{name}: " + "; ".join(diffs[:3]) + more)
    return not problems, "; ".join(problems) or f"{len(fixtures)} candidates unchanged"


def self_test(verbose=True, with_root=True):  # noqa: C901
    """Synthetic solids whose recognition and emission are known in closed form.

    Every positive case has a negative one; the ROOT half checks the emitted `TGeoShape` against
    the closed form.
    """
    import math
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                         BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeSolid,
                                         BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing,
                                         BRepBuilderAPI_Transform)
    from OCC.Core.BRepFill import brepfill
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone,
                                      BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism,
                                      BRepPrimAPI_MakeRevol, BRepPrimAPI_MakeSphere,
                                      BRepPrimAPI_MakeTorus)
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopoDS import topods
    from OCC.Core.GeomAPI import GeomAPI_Interpolate
    from OCC.Core.TColgp import TColgp_HArray1OfPnt
    from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Elips, gp_Pnt, gp_Trsf, gp_Vec

    checks = []

    def check(name, condition, detail=""):
        checks.append((name, bool(condition), detail))
        if verbose:
            print(f"  [{'ok ' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    seen_candidates = {}
    seen_recognisers = {}

    def expect(name, solid, want_recogniser, want_leaves=1):
        record = process_solid(solid, name)
        seen_recognisers[name] = record["recogniser"] if record["accepted"] else None
        if record["accepted"]:
            seen_candidates[name] = json.loads(json.dumps(record["candidate"], sort_keys=True))
        ok = record["accepted"] and record["recogniser"] == want_recogniser and \
            len(record["candidate"]["leaves"]) == want_leaves
        detail = (f"{record['recogniser']}: {record['description']}"
                  if record["recognised"] else f"declined: {record['reason']}")
        if record["recognised"] and not record["accepted"]:
            detail += f" -- rejected: {record['reason']}"
        check(f"{name} recognised as {want_recogniser} and accepted", ok, detail)
        return record

    def expect_single_cell_declined(name, solid, needle):
        """The one-cell read's verdict, asserted against the matcher that makes it."""
        _cand, why = recognise.recognise_single_cell(solid)
        ok = _cand is None and needle in (why or "")
        check(f"{name} is refused by the one-cell read", ok, f"reason: {why}")
        return why

    def expect_declined(name, solid, needle=""):
        record = process_solid(solid, name)
        ok = not record["accepted"] and (needle in (record["reason"] or ""))
        check(f"{name} is not converted as CSG", ok, f"reason: {record['reason']}")
        return record

    ax = gp_Ax2(gp_Pnt(0, 0, -5), gp_Dir(0, 0, 1))

    # --- Tier 1, one per primitive the brief scopes ---
    expect("box", BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 3.0, 4.0).Shape(), "tier1-box")
    cyl = BRepPrimAPI_MakeCylinder(ax, 2.0, 10.0).Shape()
    expect("solid cylinder", cyl, "tier1-tube")
    bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -6), gp_Dir(0, 0, 1)), 1.0, 12.0).Shape()
    tube = BRepAlgoAPI_Cut(cyl, bore).Shape()
    expect("tube", tube, "tier1-tube")
    wedge = BRepPrimAPI_MakeCylinder(ax, 2.0, 10.0, math.radians(75.0)).Shape()
    seg = BRepAlgoAPI_Cut(wedge, bore).Shape()
    expect("tube segment", seg, "tier1-tubeseg")
    expect("cone", BRepPrimAPI_MakeCone(ax, 3.0, 1.0, 10.0).Shape(), "tier1-cone")
    expect("sphere", BRepPrimAPI_MakeSphere(gp_Pnt(1, 2, 3), 2.5).Shape(), "tier1-sphere")

    # A rotated, translated tube: the frame machinery, end to end.
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 0)), 0.7)
    shift = gp_Trsf()
    shift.SetTranslation(gp_Vec(3.0, -4.0, 5.0))
    moved = BRepBuilderAPI_Transform(tube, shift.Multiplied(trsf), True).Shape()
    moved_record = expect("placed tube", moved, "tier1-tube")

    # --- Tier 2, the ExcavatorArm ram in miniature: a rod through the wall of an eye ---
    eye = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-0.75, 0, 0), gp_Dir(1, 0, 0)), 1.2, 1.5).Shape(),
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-1.0, 0, 0), gp_Dir(1, 0, 0)), 0.7, 2.0).Shape()
    ).Shape()
    rod_full = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.6, 8.0).Shape()
    rod = BRepAlgoAPI_Cut(rod_full, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(-0.75, 0, 0), gp_Dir(1, 0, 0)), 1.2, 1.5).Shape()).Shape()
    ram = BRepAlgoAPI_Fuse(eye, rod).Shape()
    ram_record = expect("rod-and-eye (two-cluster union)", ram, "tier2-tube-union", want_leaves=2)

    # --- the revolved profile: the shapes O2_TGeoToCAD.conv_pcon writes, read back ---
    # The fixture states its own (r, z) ring, independent of `primitives.pcon_profile_rz`.
    def revolved(z, rmin, rmax, phi1=0.0, dphi=360.0):
        nz = len(z)
        ring = [(rmax[i], z[i]) for i in range(nz)]
        if all(r <= 0.0 for r in rmin):
            ring += [(0.0, z[nz - 1]), (0.0, z[0])]
        else:
            ring += [(rmin[i], z[i]) for i in range(nz - 1, -1, -1)]
        deduped = []
        for pt in ring:
            if deduped and abs(pt[0] - deduped[-1][0]) < 1e-12 \
                    and abs(pt[1] - deduped[-1][1]) < 1e-12:
                continue
            deduped.append(pt)
        poly = BRepBuilderAPI_MakePolygon()
        for (r, zz) in deduped:
            poly.Add(gp_Pnt(float(r), 0.0, float(zz)))
        poly.Close()
        rev = BRepPrimAPI_MakeRevol(BRepBuilderAPI_MakeFace(poly.Wire()).Face(),
                                    gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                    math.radians(dphi))
        rev.Build()
        shape = rev.Shape()
        if abs(phi1) > 1e-12:
            spin = gp_Trsf()
            spin.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(phi1))
            shape = BRepBuilderAPI_Transform(shape, spin, True).Shape()
        return shape

    def expect_pcon(name, solid, z, rmin, rmax, phi1=0.0, dphi=360.0):
        record = expect(name, solid, "revolved-pcon")
        if not record["accepted"]:
            return record
        p = record["candidate"]["leaves"][0]["params"]
        worst = max([abs(a - b) for a, b in zip(p["z"], z)]
                    + [abs(a - b) for a, b in zip(p["rmin"], rmin)]
                    + [abs(a - b) for a, b in zip(p["rmax"], rmax)]
                    + [abs(p["phi1"] - phi1), abs(p["dphi"] - dphi)]) \
            if len(p["z"]) == len(z) else float("inf")
        check(f"{name} reconstructs the source TGeoPcon parameters",
              len(p["z"]) == len(z) and worst < 1.0e-9,
              f"nz {len(p['z'])} vs {len(z)}, worst parameter deviation {worst:.3g}")
        return record

    # z-steps: duplicate z planes on both rmin and rmax, which the writer emits as cap annuli.
    step_z, step_rmin, step_rmax = [-5, 0, 0, 5], [1, 1, 2, 2], [3, 3, 4, 4]
    stepped = revolved(step_z, step_rmin, step_rmax)
    expect_pcon("stepped polycone (duplicate z planes)", stepped, step_z, step_rmin, step_rmax)
    # mixed cone and cylinder laterals on one axis -- the IBCYSSCone case, which the whole-part
    # matcher declines with "mixed lateral surface kinds".
    expect_pcon("cone and cylinder laterals on one axis",
                revolved([-5, 0, 5], [1, 1, 2], [2, 3, 3]), [-5, 0, 5], [1, 1, 2], [2, 3, 3])
    # rmin stepping through 0: the inner lateral is a cone that reaches the axis.
    expect_pcon("polycone whose rmin steps through 0",
                revolved([0, 5, 10], [0, 0, 2], [4, 4, 4]), [0, 5, 10], [0, 0, 2], [4, 4, 4])
    # a half turn, and a partial-phi wedge stated in absolute phi on an identity frame.
    expect_pcon("half-turn polycone", revolved([-5, 0, 5], [1, 1, 2], [2, 3, 3], 0.0, 180.0),
                [-5, 0, 5], [1, 1, 2], [2, 3, 3], 0.0, 180.0)
    expect_pcon("partial-phi stepped polycone",
                revolved(step_z, step_rmin, step_rmax, 10.0, 120.0),
                step_z, step_rmin, step_rmax, 10.0, 120.0)
    # a rotated, translated polycone: the frame machinery on a multi-section leaf.
    pcon_trsf = gp_Trsf()
    pcon_trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 0)), 0.7)
    pcon_shift = gp_Trsf()
    pcon_shift.SetTranslation(gp_Vec(3.0, -4.0, 5.0))
    pcon_place = pcon_shift.Multiplied(pcon_trsf)
    moved_pcon = BRepBuilderAPI_Transform(stepped, pcon_place, True).Shape()
    moved_pcon_record = expect("placed stepped polycone", moved_pcon, "revolved-pcon")
    check("a placed polycone travels as one leaf plus a rigid placement",
          moved_pcon_record["accepted"]
          and prim.placement_for_candidate(moved_pcon_record["candidate"]) is not None,
          "placement present" if moved_pcon_record["accepted"] else "not accepted")

    # --- negative controls: each must decline or be rejected ---
    # 1. a blind bore, which is a polycone.
    blind = BRepAlgoAPI_Cut(cyl, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, -6), gp_Dir(0, 0, 1)), 1.0, 9.0).Shape()).Shape()
    expect_pcon("cylinder with a blind bore", blind, [-5, 3, 3, 5], [1, 1, 0, 0], [2, 2, 2, 2])
    # 2. an L-shape, which is a TGeoXtru.
    ell = BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 4.0, 4.0, 1.0).Shape(),
                          BRepPrimAPI_MakeBox(gp_Pnt(2, 2, -1), 4.0, 4.0, 3.0).Shape()).Shape()
    expect("L-shaped plate", ell, "rung2-xtru")
    # 3. a torus.
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeTorus
    expect("torus", BRepPrimAPI_MakeTorus(5.0, 1.0).Shape(), "tier1-torus")
    # 4. a cylinder with a flat milled off it.
    flatted = BRepAlgoAPI_Cut(cyl, BRepPrimAPI_MakeBox(
        gp_Pnt(1.5, -3, -6), 3.0, 6.0, 12.0).Shape()).Shape()
    # A milled flat is one cell of four halfspaces.
    flat_record = expect("cylinder with a milled flat", flatted, "cell-intersection",
                         want_leaves=2)

    # --- negative controls for the revolved matcher ---
    # 5. a TGeoPgon, whose planar laterals must never be read as a polycone.
    def prism_ring(apothem, nedges, phi1=0.0, dphi=360.0):
        dseg = math.radians(dphi) / nedges
        radius = apothem / math.cos(dseg / 2.0)
        n = nedges if abs(dphi - 360.0) < 1e-9 else nedges + 1
        return [(radius * math.cos(math.radians(phi1) + k * dseg),
                 radius * math.sin(math.radians(phi1) + k * dseg)) for k in range(n)]

    def swept_polygon(apothem, nedges, z0, z1):
        poly = BRepBuilderAPI_MakePolygon()
        for (x, y) in prism_ring(apothem, nedges):
            poly.Add(gp_Pnt(x, y, z0))
        poly.Close()
        pr = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(poly.Wire()).Face(),
                                   gp_Vec(0, 0, z1 - z0))
        pr.Build()
        return pr.Shape()

    #    They convert as TGeoPgon, not as a polycone.
    for nedges in (8, 48):
        pgon = BRepAlgoAPI_Cut(swept_polygon(3.0, nedges, -5.0, 5.0),
                               swept_polygon(1.5, nedges, -6.0, 6.0)).Shape()
        expect(f"hollow {nedges}-edge polygon (TGeoPgon)", pgon, "rung2-pgon")
    # 6. polygonal laterals sharing an axis with a real cylinder.
    hybrid = BRepAlgoAPI_Fuse(
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -5), gp_Dir(0, 0, 1)), 3.0, 5.0).Shape(),
        swept_polygon(3.0, 6, 0.0, 5.0)).Shape()
    # It converts as two cells; no whole-part matcher may take it.
    hybrid_single = recognise.recognise_single_cell(hybrid)[1]
    check("a cylinder with a coaxial hexagonal section is no whole-part primitive",
          recognise.recognise(hybrid)[0]["recogniser"] == "cells-union"
          and "neither a cap nor a wedge" in (recognise.recognise_revolved(hybrid)[1] or ""),
          f"one-cell read: {(hybrid_single or '')[:90]}")
    # 7. a bore displaced off the axis, which the symmetric difference refuses.
    for displacement in (1.0e-6, 1.0e-5):
        off = BRepAlgoAPI_Cut(
            revolved(step_z, [0, 0, 0, 0], step_rmax),
            BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(displacement, 0, -6), gp_Dir(0, 0, 1)),
                                     1.0, 12.0).Shape()).Shape()
        expect_declined(f"stepped polycone with the bore {displacement:g} cm off axis", off)
    # 8. a cap plane tilted off perpendicular.
    tilt = gp_Trsf()
    tilt.SetRotation(gp_Ax1(gp_Pnt(0, 0, 5), gp_Dir(1, 0, 0)), 1.0e-4)
    knife = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 4.9), gp_Dir(0, 0, 1)),
                                     10.0, 5.0).Shape()
    expect_declined("stepped polycone with a tilted top cap",
                    BRepAlgoAPI_Cut(stepped,
                                    BRepBuilderAPI_Transform(knife, tilt, True).Shape()).Shape(),
                    "neither a cap nor a wedge")

    # --- the instrument that scores the revolved candidate must be able to say "no" ---
    true_profile = prim.pcon_profile_rz({"z": [float(v) for v in step_z],
                                         "rmin": [float(v) for v in step_rmin],
                                         "rmax": [float(v) for v in step_rmax]})
    samples = [(3.0, -2.5), (4.0, 2.5), (1.0, -5.0), (2.0, 5.0), (3.5, 0.0)]
    check("the profile gap is zero on the profile's own boundary",
          recognise._profile_gap(true_profile, samples) < 1.0e-12,
          f"gap {recognise._profile_gap(true_profile, samples):.3g} cm")
    nudged_profile = [(r + (1.0e-6 if abs(r - 3.0) < 1e-12 else 0.0), z)
                      for (r, z) in true_profile]
    nudged_gap = recognise._profile_gap(nudged_profile, samples)
    check("the profile gap reports a radius displaced by ten model tolerances",
          abs(nudged_gap - 1.0e-6) < 1.0e-12, f"gap {nudged_gap:.3g} cm, expected 1e-06 cm")

    # --- the description must refuse an illegal TGeoPcon before either builder sees it ---
    for name, params in (
            ("unequal array lengths",
             {"phi1": 0.0, "dphi": 360.0, "z": [0.0, 1.0], "rmin": [0.0], "rmax": [1.0, 1.0]}),
            ("rmin above rmax",
             {"phi1": 0.0, "dphi": 360.0, "z": [0.0, 1.0], "rmin": [2.0, 2.0],
              "rmax": [1.0, 1.0]}),
            ("a single section",
             {"phi1": 0.0, "dphi": 360.0, "z": [0.0], "rmin": [0.0], "rmax": [1.0]}),
            ("z running backwards",
             {"phi1": 0.0, "dphi": 360.0, "z": [1.0, 0.0], "rmin": [0.0, 0.0],
              "rmax": [1.0, 1.0]})):
        try:
            prim.leaf("TGeoPcon", params, prim.identity_frame())
            refused = False
        except ValueError:
            refused = True
        check(f"a TGeoPcon description with {name} is refused", refused)

    # --- an all-cone stack, retried after the acceptance test refuses tier 1 ---
    stack_record = expect_pcon("all-cone stack (two cones and two caps)",
                               revolved([-3, 0, 3], [0, 0, 0], [2, 3, 1]),
                               [-3, 0, 3], [0, 0, 0], [2, 3, 1])
    check("the all-cone stack was retried after tier 1 was rejected, not merely declined",
          (stack_record.get("retriedAfter") or {}).get("recogniser") == "tier1-cone",
          f"retried after {(stack_record.get('retriedAfter') or {}).get('recogniser')}: "
          f"{(stack_record.get('retriedAfter') or {}).get('reason')}")
    # An hourglass pinches to the axis (rmax = 0), a legal polycone.
    expect("hourglass (two cones meeting on the axis)",
           revolved([-5, 0, 5], [0, 0, 0], [2, 0, 2]), "revolved-pcon")

    # --- a two-section full-turn profile is said in its native class ---
    def expect_native(name, solid, want_recogniser, want_type, want_params):
        record = expect(name, solid, want_recogniser)
        if not record["accepted"]:
            return record
        lf = record["candidate"]["leaves"][0]
        worst = max(abs(lf["params"][k] - v) for k, v in want_params.items()) \
            if lf["type"] == want_type else float("inf")
        check(f"{name} emits a native {want_type} with the source's parameters",
              lf["type"] == want_type and worst < 1.0e-9,
              f"{lf['type']}, worst parameter deviation {worst:.3g}")
        return record

    # A TGeoCone with one radius constant must come back as a TGeoCone.
    expect_native("cone with a cylindrical bore (constant rmin)",
                  revolved([-25, 25], [4.5, 4.5], [16.22, 25.04]), "revolved-cone", "TGeoCone",
                  {"dz": 25.0, "rmin1": 4.5, "rmax1": 16.22, "rmin2": 4.5, "rmax2": 25.04})
    expect_native("cylinder with a conical bore (constant rmax)",
                  revolved([-3, 3], [6.99, 7.374], [26.02, 26.02]), "revolved-cone", "TGeoCone",
                  {"dz": 3.0, "rmin1": 6.99, "rmax1": 26.02, "rmin2": 7.374, "rmax2": 26.02})
    # A wedge and a step must stay polycones.
    expect_pcon("two-section wedge stays a polycone", revolved([-5, 5], [1, 1], [2, 3], 0.0,
                                                               120.0),
                [-5, 5], [1, 1], [2, 3], 0.0, 120.0)
    expect_pcon("a stepped profile stays a polycone", stepped, step_z, step_rmin, step_rmax)
    # The TGeoTube branch is unreachable from CAD, so it is exercised on the description.
    tube_leaf, tube_tag = recognise._canonical_revolved_leaf(
        prim.leaf("TGeoPcon", {"phi1": 0.0, "dphi": 360.0, "z": [-4.0, 6.0],
                               "rmin": [1.0, 1.0], "rmax": [2.0, 2.0]}, prim.identity_frame()),
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0e-9)
    check("a two-section profile with constant radii canonicalises to a TGeoTube",
          tube_tag == "revolved-tube" and tube_leaf["type"] == "TGeoTube"
          and abs(tube_leaf["params"]["dz"] - 5.0) < 1e-12
          and abs(tube_leaf["frame"]["origin"][2] - 1.0) < 1e-12,
          f"{tube_tag}, {tube_leaf['type']}, dz {tube_leaf['params']['dz']}, origin "
          f"{tube_leaf['frame']['origin']}")

    # --- rung 2: the prism family, the shapes `_prism_from_rings` writes, read back ---
    # The fixture sews its own faces from its own ring coordinates.
    def prism(rings, inner=None):
        stacks = [[[tuple(float(c) for c in q) for q in ring] for ring in rings]]
        if inner is not None:
            stacks.append([[tuple(float(c) for c in q) for q in ring] for ring in inner])
        faces = []
        for stack in stacks:
            nv = len(stack[0])
            for k in range(len(stack) - 1):
                lo, hi = stack[k], stack[k + 1]
                for i in range(nv):
                    j = (i + 1) % nv
                    poly = BRepBuilderAPI_MakePolygon()
                    for q in (lo[i], lo[j], hi[j], hi[i]):
                        poly.Add(gp_Pnt(*q))
                    poly.Close()
                    made = BRepBuilderAPI_MakeFace(poly.Wire())
                    if made.IsDone():
                        faces.append(made.Face())
        for idx in (0, -1):
            poly = BRepBuilderAPI_MakePolygon()
            for q in stacks[0][idx]:
                poly.Add(gp_Pnt(*q))
            poly.Close()
            made = BRepBuilderAPI_MakeFace(poly.Wire())
            if len(stacks) == 2:
                hole = BRepBuilderAPI_MakePolygon()
                for q in stacks[1][idx]:
                    hole.Add(gp_Pnt(*q))
                hole.Close()
                made.Add(topods.Wire(hole.Wire().Reversed()))
            faces.append(made.Face())
        extent = max(abs(c) for stack in stacks for r in stack for q in r for c in q) or 1.0
        sew = BRepBuilderAPI_Sewing(1.0e-7 * extent)
        for face in faces:
            sew.Add(face)
        sew.Perform()
        ms = BRepBuilderAPI_MakeSolid(topods.Shell(sew.SewedShape()))
        ms.Build()
        solid = ms.Solid()
        props = GProp_GProps()
        brepgprop.VolumeProperties(solid, props)
        if props.Mass() < 0.0:
            solid = topods.Solid(solid.Reversed())
        return solid

    def polygon_ring(corners, z):
        return [(x, y, z) for (x, y) in corners]

    def regular_ring(apothem, nedges, z, phi1=0.0, dphi=360.0):
        dseg = math.radians(dphi) / nedges
        radius = apothem / math.cos(dseg / 2.0)
        n = nedges if abs(dphi - 360.0) < 1e-9 else nedges + 1
        return [(radius * math.cos(math.radians(phi1) + k * dseg),
                 radius * math.sin(math.radians(phi1) + k * dseg), z) for k in range(n)]

    def expect_prism(name, solid, want_recogniser, want_type, want_params):
        record = expect(name, solid, want_recogniser)
        if not record["accepted"]:
            return record
        lf = record["candidate"]["leaves"][0]
        worst = 0.0
        if lf["type"] != want_type:
            worst = float("inf")
        else:
            for key, want in want_params.items():
                got = lf["params"][key]
                if isinstance(want, (list, tuple)):
                    worst = (float("inf") if len(got) != len(want)
                             else max([worst] + [abs(a - b) for a, b in zip(got, want)]))
                else:
                    worst = max(worst, abs(got - want))
        check(f"{name} emits a native {want_type} with the source's parameters",
              lf["type"] == want_type and worst < 1.0e-9,
              f"{lf['type']}, worst parameter deviation {worst:.3g}")
        return record

    def trd_rings(dx1, dx2, dy1, dy2, dz):
        return [[(-dx1, -dy1, -dz), (dx1, -dy1, -dz), (dx1, dy1, -dz), (-dx1, dy1, -dz)],
                [(-dx2, -dy2, dz), (dx2, -dy2, dz), (dx2, dy2, dz), (-dx2, dy2, dz)]]

    # TGeoTrd1: the slanted prism behind TPC's 44 "a box face has no opposite partner" declines.
    expect_prism("Trd1 (slanted x faces)", prism(trd_rings(3, 1, 2, 2, 5)), "rung2-trd1",
                 "TGeoTrd1", {"dx1": 3.0, "dx2": 1.0, "dy": 2.0, "dz": 5.0})
    # The taper reversed, and TPC_IRB1's 0.076 cm slant on 14.2 cm.
    expect_prism("Trd1 (taper reversed)", prism(trd_rings(1, 3, 2, 2, 4)), "rung2-trd1",
                 "TGeoTrd1", {"dx1": 1.0, "dx2": 3.0, "dy": 2.0, "dz": 4.0})
    expect_prism("Trd1 (TPC_IRB1's 0.5 % slant)",
                 prism(trd_rings(14.205637404580152, 14.281551908396947, 2.06, 2.06, 0.2)),
                 "rung2-trd1", "TGeoTrd1",
                 {"dx1": 14.205637404580152, "dx2": 14.281551908396947, "dy": 2.06, "dz": 0.2})
    # TGeoTrd2: both half-widths vary; the more specific class wins over a legal Xtru.
    expect_prism("Trd2 (both half-widths vary)", prism(trd_rings(3, 1, 2, 4, 5)), "rung2-trd2",
                 "TGeoTrd2", {"dx1": 3.0, "dx2": 1.0, "dy1": 2.0, "dy2": 4.0, "dz": 5.0})
    expect_prism("Trd2 (isotropic taper, also a legal Xtru)", prism(trd_rings(3, 1.5, 2, 1, 5)),
                 "rung2-trd2", "TGeoTrd2",
                 {"dx1": 3.0, "dx2": 1.5, "dy1": 2.0, "dy2": 1.0, "dz": 5.0})

    # TGeoArb8: a sheared hexahedron, and TPC_IHSTR's trapezoidal prism stated corner for corner.
    para = prism([[(-2, -2, -3), (2, -2, -3), (2, 2, -3), (-2, 2, -3)],
                  [(-1, -1.5, 3), (3, -1.5, 3), (3, 2.5, 3), (-1, 2.5, 3)]])
    expect_prism("Arb8 (parallelepiped)", para, "rung2-arb8", "TGeoArb8",
                 {"dz": 3.0, "vertices": [-2, -2, 2, -2, 2, 2, -2, 2,
                                          -1, -1.5, 3, -1.5, 3, 2.5, -1, 2.5]})
    ihstr = [(0.0, 0.0), (0.0, 1.08), (2.3, 1.08), (3.38, 0.0)]
    expect_prism("Arb8 (TPC_IHSTR's trapezoidal prism)",
                 prism([polygon_ring(ihstr, -0.6), polygon_ring(ihstr, 0.6)]),
                 "rung2-arb8", "TGeoArb8",
                 {"dz": 0.6, "vertices": [0, 0, 3.38, 0, 2.3, 1.08, 0, 1.08,
                                          0, 0, 3.38, 0, 2.3, 1.08, 0, 1.08]})
    # A hexahedron sheared in x only: neither a Trd nor an Xtru.
    expect("Arb8 (sheared in x only)",
           prism([[(-2, -1, -2), (2, -1, -2), (2, 1, -2), (-2, 1, -2)],
                  [(-2, -1, 2), (4, -1, 2), (4, 1, 2), (-2, 1, 2)]]), "rung2-arb8")
    # A TGeoTrap's corners, from `TGeoTrap(5, 10, 20, 2, 3, 4, 5, 2, 3, 4, 5).GetVertices()`.
    trap_bottom = [(-4.003443140137866, -2.301536896070458),
                   (-4.653488486034171, 1.698463103929542),
                   (3.3465115139658295, 1.698463103929542),
                   (1.996556859862133, -2.301536896070458)]
    trap_top = [(-2.346511513965829, -1.698463103929542),
                (-2.996556859862133, 2.301536896070458),
                (5.003443140137866, 2.301536896070458),
                (3.653488486034171, -1.698463103929542)]
    expect("Arb8 (a TGeoTrap's eight corners)",
           prism([polygon_ring(trap_bottom, -5.0), polygon_ring(trap_top, 5.0)]), "rung2-arb8")

    # TGeoXtru: ITS's 23 Xtru volumes are all right prisms on a general, often non-convex polygon.
    ell_poly = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    expect_prism("Xtru (non-convex L section)",
                 prism([polygon_ring(ell_poly, -2), polygon_ring(ell_poly, 2)]),
                 "rung2-xtru", "TGeoXtru",
                 {"x": [0, 3, 3, 1, 1, 0], "y": [0, 0, 1, 1, 3, 3], "z": [-2, 2],
                  "xoff": [0, 0], "yoff": [0, 0], "scale": [1, 1]})
    rib = [(0, 0), (4.2, 0), (4.2, 0.1), (5.05, 0.1), (9.803, 1.83), (5.9, 1.83), (5.0, 2.73),
           (0, 2.73)]
    expect("Xtru (ITS ConeARibVol0's eight-corner section)",
           prism([polygon_ring(rib, -0.045), polygon_ring(rib, 0.045)]), "rung2-xtru")
    expect("Xtru (a triangular section)",
           prism([polygon_ring([(0, 0), (0.05, 0), (0, 0.074)], -14.5),
                  polygon_ring([(0, 0), (0.05, 0), (0, 0.074)], 14.5)]), "rung2-xtru")
    # Three sections with a per-section offset and an isotropic scale.
    scaled_poly = [(0, 0), (2, 0), (2, 1), (1, 2), (0, 2)]
    scaled = prism([[(0.0 + 1.0 * x, 0.0 + 1.0 * y, -3.0) for x, y in scaled_poly],
                    [(0.5 + 1.4 * x, -0.25 + 1.4 * y, 0.0) for x, y in scaled_poly],
                    [(1.0 + 0.6 * x, 0.0 + 0.6 * y, 3.0) for x, y in scaled_poly]])
    expect_prism("Xtru (three sections, offset and scaled)", scaled, "rung2-xtru", "TGeoXtru",
                 {"z": [-3, 0, 3], "xoff": [0, 0.5, 1.0], "yoff": [0, -0.25, 0],
                  "scale": [1.0, 1.4, 0.6]})

    # TGeoPgon: the laterals are planes at the apothem radius, corners at `r / cos(dseg/2)`.
    expect_prism("Pgon (solid hexagonal prism)",
                 prism([regular_ring(3, 6, -5), regular_ring(3, 6, 5)]), "rung2-pgon",
                 "TGeoPgon", {"nedges": 6, "phi1": 0.0, "dphi": 360.0, "z": [-5, 5],
                              "rmin": [0, 0], "rmax": [3, 3]})
    expect_prism("Pgon (tapered eight-edge prism)",
                 prism([regular_ring(3, 8, -5), regular_ring(1.5, 8, 5)]), "rung2-pgon",
                 "TGeoPgon", {"nedges": 8, "phi1": 0.0, "dphi": 360.0, "z": [-5, 5],
                              "rmin": [0, 0], "rmax": [3, 1.5]})
    for nedges in (8, 48):
        hollow = prism([regular_ring(3, nedges, -5), regular_ring(3, nedges, 5)],
                       inner=[regular_ring(1.5, nedges, -5), regular_ring(1.5, nedges, 5)])
        expect_prism(f"Pgon (hollow {nedges}-edge prism)", hollow, "rung2-pgon", "TGeoPgon",
                     {"nedges": nedges, "phi1": 0.0, "dphi": 360.0, "z": [-5, 5],
                      "rmin": [1.5, 1.5], "rmax": [3, 3]})
    # TPC_Strip: 18 edges, a 1 mm wall on an 85 cm radius, 250 cm long.
    expect_prism("Pgon (TPC_Strip's thin 18-edge shell)",
                 prism([regular_ring(85.235, 18, -124.8), regular_ring(85.235, 18, 124.8)],
                       inner=[regular_ring(85.225, 18, -124.8), regular_ring(85.225, 18, 124.8)]),
                 "rung2-pgon", "TGeoPgon",
                 {"nedges": 18, "phi1": 0.0, "dphi": 360.0, "z": [-124.8, 124.8],
                  "rmin": [85.225, 85.225], "rmax": [85.235, 85.235]})
    # Three hollow sections with the radii stepping.
    expect_prism("Pgon (three hollow sections)",
                 prism([regular_ring(3, 6, -5), regular_ring(3, 6, 0), regular_ring(4, 6, 5)],
                       inner=[regular_ring(1, 6, -5), regular_ring(1, 6, 0),
                              regular_ring(2, 6, 5)]),
                 "rung2-pgon", "TGeoPgon",
                 {"nedges": 6, "phi1": 0.0, "dphi": 360.0, "z": [-5, 0, 5],
                  "rmin": [1, 1, 2], "rmax": [3, 3, 4]})
    # A phi wedge closing on its axis, and one across phi = 0.
    expect_prism("Pgon (a 90 deg wedge closing on the axis)",
                 prism([regular_ring(4, 3, -2, 10.0, 90.0) + [(0.0, 0.0, -2.0)],
                        regular_ring(4, 3, 2, 10.0, 90.0) + [(0.0, 0.0, 2.0)]]),
                 "rung2-pgon", "TGeoPgon",
                 {"nedges": 3, "phi1": 10.0, "dphi": 90.0, "z": [-2, 2], "rmin": [0, 0],
                  "rmax": [4, 4]})
    expect_prism("Pgon (a wedge across phi = 0)",
                 prism([regular_ring(4, 2, -2, 350.0, 20.0) + [(0.0, 0.0, -2.0)],
                        regular_ring(4, 2, 2, 350.0, 20.0) + [(0.0, 0.0, 2.0)]]),
                 "rung2-pgon", "TGeoPgon",
                 {"nedges": 2, "phi1": 350.0, "dphi": 20.0, "z": [-2, 2], "rmin": [0, 0],
                  "rmax": [4, 4]})

    # A placed prism: the frame machinery on a leaf that has no origin of its own.
    prism_trsf = gp_Trsf()
    prism_trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 0)), 0.7)
    prism_shift = gp_Trsf()
    prism_shift.SetTranslation(gp_Vec(3.0, -4.0, 5.0))
    prism_place = prism_shift.Multiplied(prism_trsf)
    moved_trd = BRepBuilderAPI_Transform(prism(trd_rings(3, 1, 2, 2, 5)), prism_place,
                                         True).Shape()
    moved_trd_record = expect("placed Trd1", moved_trd, "rung2-trd1")
    check("a placed Trd1 travels as one leaf plus a rigid placement",
          moved_trd_record["accepted"]
          and prim.placement_for_candidate(moved_trd_record["candidate"]) is not None,
          "placement present" if moved_trd_record["accepted"] else "not accepted")
    moved_xtru = BRepBuilderAPI_Transform(
        prism([polygon_ring(ell_poly, -2), polygon_ring(ell_poly, 2)]), prism_place,
        True).Shape()
    expect("placed Xtru (non-convex L section)", moved_xtru, "rung2-xtru")

    # --- rung 2 negative controls ---
    # 1. A twisted TGeoArb8, whose ruled B-spline laterals are declined as free-form.
    twisted_faces = []
    twist_bottom = [(-2, -2, -2), (-2, 2, -2), (2, 2, -2), (2, -2, -2)]
    twist_top = [(-1.41, -2.73, 2), (-2.73, 1.41, 2), (1.41, 2.73, 2), (2.73, -1.41, 2)]
    for i in range(4):
        j = (i + 1) % 4
        e1 = BRepBuilderAPI_MakeEdge(gp_Pnt(*twist_bottom[i]), gp_Pnt(*twist_bottom[j])).Edge()
        e2 = BRepBuilderAPI_MakeEdge(gp_Pnt(*twist_top[i]), gp_Pnt(*twist_top[j])).Edge()
        twisted_faces.append(brepfill.Face(e1, e2))
    for ring in (twist_bottom, twist_top):
        poly = BRepBuilderAPI_MakePolygon()
        for q in ring:
            poly.Add(gp_Pnt(*q))
        poly.Close()
        twisted_faces.append(BRepBuilderAPI_MakeFace(poly.Wire()).Face())
    sew_twist = BRepBuilderAPI_Sewing(1.0e-6)
    for face in twisted_faces:
        sew_twist.Add(face)
    sew_twist.Perform()
    twist_solid = BRepBuilderAPI_MakeSolid(topods.Shell(sew_twist.SewedShape()))
    twist_solid.Build()
    expect_declined("twisted hexahedron (a ruled TGeoArb8 side)", twist_solid.Solid(),
                    "free-form faces")

    # 2. A middle section stretched in y only: the volume refuses 1e-06 cm, the gap 1e-05 cm.
    for displacement in (1.0e-6, 1.0e-5):
        near = prism([[(-2, -1, -2), (2, -1, -2), (2, 1, -2), (-2, 1, -2)],
                      [(-2, -1 - displacement, 0), (2, -1 - displacement, 0),
                       (2, 1 + displacement, 0), (-2, 1 + displacement, 0)],
                      [(-2, -1, 2), (2, -1, 2), (2, 1, 2), (-2, 1, 2)]])
        expect_declined(f"prism with one section {displacement:g} cm out of similarity", near)

    # 3. A polycone must not be taken by a prism template.
    check("a polycone reaches the revolved matcher, not the prism one",
          process_solid(stepped, "pcon-vs-prism")["recogniser"] == "revolved-pcon",
          f"{process_solid(stepped, 'pcon-vs-prism')['recogniser']}")

    # --- the instrument that scores a prism candidate must be able to say "no" ---
    exact_ring = [(-2.0, -1.0, -2.0), (2.0, -1.0, -2.0), (2.0, 1.0, -2.0), (-2.0, 1.0, -2.0)]
    nudged = [(x + (1.0e-6 if i == 0 else 0.0), y, z)
              for i, (x, y, z) in enumerate(exact_ring)]
    check("the point-set gap is zero on the point set itself",
          recognise._point_set_gap(exact_ring, exact_ring) == 0.0,
          f"gap {recognise._point_set_gap(exact_ring, exact_ring):.3g} cm")
    nudged_gap = recognise._point_set_gap(exact_ring, nudged)
    check("the point-set gap reports a corner displaced by ten model tolerances",
          abs(nudged_gap - 1.0e-6) < 1.0e-15, f"gap {nudged_gap:.3g} cm, expected 1e-06 cm")
    # ... and a hexahedron read in the wrong corner order, which only the edge midpoints catch.
    good_arb8 = prim.leaf("TGeoArb8", {"dz": 3.0,
                                       "vertices": [-2, -2, 2, -2, 2, 2, -2, 2,
                                                    -1, -1.5, 3, -1.5, 3, 2.5, -1, 2.5]},
                          prim.identity_frame())
    swapped = list(good_arb8["params"]["vertices"])
    swapped[2:4], swapped[4:6] = swapped[4:6], swapped[2:4]
    bad_arb8 = prim.leaf("TGeoArb8", {"dz": 3.0, "vertices": swapped}, prim.identity_frame())
    corner_only_gap = recognise._point_set_gap(
        [tuple(q) for q in prim.prism_samples(good_arb8)[0::3]],
        [tuple(q) for q in prim.prism_samples(bad_arb8)[0::3]])
    order_gap = recognise._point_set_gap(prim.prism_samples(good_arb8),
                                         prim.prism_samples(bad_arb8))
    check("the edge midpoints are what catch a hexahedron read in the wrong corner order",
          corner_only_gap == 0.0 and order_gap > 0.1,
          f"corners alone {corner_only_gap:.3g} cm, corners and edge midpoints "
          f"{order_gap:.3g} cm")

    # --- the description must refuse an illegal prism before either builder sees it ---
    for name, kind, params in (
            ("a TGeoXtru whose z runs backwards", "TGeoXtru",
             {"x": [0, 1, 0], "y": [0, 0, 1], "z": [1.0, 0.0], "xoff": [0, 0], "yoff": [0, 0],
              "scale": [1, 1]}),
            ("a TGeoXtru with a repeated corner", "TGeoXtru",
             {"x": [0, 1, 1], "y": [0, 0, 0], "z": [0.0, 1.0], "xoff": [0, 0], "yoff": [0, 0],
              "scale": [1, 1]}),
            ("a TGeoXtru with two corners", "TGeoXtru",
             {"x": [0, 1], "y": [0, 0], "z": [0.0, 1.0], "xoff": [0, 0], "yoff": [0, 0],
              "scale": [1, 1]}),
            ("a TGeoXtru with a zero scale", "TGeoXtru",
             {"x": [0, 1, 0], "y": [0, 0, 1], "z": [0.0, 1.0], "xoff": [0, 0], "yoff": [0, 0],
              "scale": [1, 0]}),
            ("a TGeoArb8 with fifteen coordinates", "TGeoArb8",
             {"dz": 1.0, "vertices": [0.0] * 15}),
            ("a TGeoArb8 with a collapsed face", "TGeoArb8",
             {"dz": 1.0, "vertices": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]}),
            ("a TGeoTrd1 with both half-widths zero", "TGeoTrd1",
             {"dx1": 0.0, "dx2": 0.0, "dy": 1.0, "dz": 1.0}),
            ("a TGeoTrd2 with a negative half-width", "TGeoTrd2",
             {"dx1": -1.0, "dx2": 1.0, "dy1": 1.0, "dy2": 1.0, "dz": 1.0}),
            ("a TGeoPgon with no edges", "TGeoPgon",
             {"phi1": 0.0, "dphi": 360.0, "nedges": 0, "z": [0.0, 1.0], "rmin": [0.0, 0.0],
              "rmax": [1.0, 1.0]})):
        try:
            prim.leaf(kind, params, prim.identity_frame())
            refused = False
        except ValueError:
            refused = True
        check(f"{name} is refused", refused)

    # A TGeoXtru's two array-length groups are each checked.
    try:
        prim.leaf("TGeoXtru", {"x": [0, 1, 0], "y": [0, 0], "z": [0.0, 1.0], "xoff": [0, 0],
                               "yoff": [0, 0], "scale": [1, 1]}, prim.identity_frame())
        refused = False
    except ValueError:
        refused = True
    check("a TGeoXtru whose x and y differ in length is refused", refused)
    xtru_two_lengths = prim.leaf(
        "TGeoXtru", {"x": [0, 2, 2, 0], "y": [0, 0, 1, 1], "z": [-1.0, 0.0, 1.0],
                     "xoff": [0, 0, 0], "yoff": [0, 0, 0], "scale": [1, 1, 1]},
        prim.identity_frame())
    check("a TGeoXtru carries four corners and three sections in one description",
          len(xtru_two_lengths["params"]["x"]) == 4 and len(xtru_two_lengths["params"]["z"]) == 3,
          f"{len(xtru_two_lengths['params']['x'])} corners, "
          f"{len(xtru_two_lengths['params']['z'])} sections")

    # --- the floor for rung 2's own emissions ---
    check("every prism-family candidate matches its recorded candidate within tolerance",
          *_recorded_match(_PRISM_FIXTURES, seen_candidates))

    # --- the floor: nothing that converted before this matcher existed converts differently ---
    check("every whole-part candidate matches its recorded candidate within tolerance",
          *_recorded_match(_WHOLE_PART_FIXTURES, seen_candidates))

    # --- rung 3: the single cell ---
    # The same constructions as `make_boolean_fixtures.py`, in cm.
    def cyl_along(radius, length, origin, direction):
        return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*origin), gp_Dir(*direction)),
                                        radius, length).Shape()

    # Two orthogonal r = 1 cylinders intersected: the Steinmetz solid, with no planar face.
    steinmetz = BRepAlgoAPI_Common(cyl_along(1.0, 6.0, (0, 0, -3), (0, 0, 1)),
                                   cyl_along(1.0, 6.0, (-3, 0, 0), (1, 0, 0))).Shape()
    steinmetz_record = expect("Steinmetz solid (two cylinders intersected)", steinmetz,
                              "cell-intersection", want_leaves=2)
    check("the Steinmetz solid reaches the cell emitter only after a rejection",
          (steinmetz_record.get("retriedAfter") or {}).get("recogniser") == "tier2-tube-union",
          f"retried after {(steinmetz_record.get('retriedAfter') or {}).get('recogniser')}")
    # A tube with a transverse hole, whose wall enters as a subtraction.
    window = BRepAlgoAPI_Cut(cyl_along(1.5, 6.0, (0, 0, -3), (0, 0, 1)),
                             cyl_along(0.8, 6.0, (-3, 0, 0), (1, 0, 0))).Shape()
    window_record = expect("tube with a transverse window", window, "cell-intersection",
                           want_leaves=2)
    check("the window's hole wall is a complemented leaf and its barrel is not",
          window_record["accepted"]
          and not window_record["candidate"]["leaves"][0].get("outside")
          and window_record["candidate"]["leaves"][1].get("outside") is True,
          window_record["description"])
    check("the barrel and its two caps folded into one TGeoTube",
          window_record["accepted"]
          and window_record["candidate"]["leaves"][0]["type"] == "TGeoTube"
          and abs(window_record["candidate"]["leaves"][0]["params"]["dz"] - 3.0) < 1e-12
          and window_record["candidate"]["notes"]["nCarriers"] == 4,
          f"{window_record['candidate']['notes'] if window_record['accepted'] else 'n/a'}")
    # A cylinder cut by an oblique plane, which stays a halfspace.
    oblique_knife = BRepPrimAPI_MakeBox(gp_Pnt(-20, -20, 0), 40.0, 40.0, 40.0).Shape()
    oblique_spin = gp_Trsf()
    oblique_spin.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.radians(60.0))
    oblique_lift = gp_Trsf()
    oblique_lift.SetTranslation(gp_Vec(0.0, 0.0, 2.5))
    oblique = BRepAlgoAPI_Cut(
        cyl_along(1.2, 5.0, (0, 0, 0), (0, 0, 1)),
        BRepBuilderAPI_Transform(oblique_knife, oblique_lift.Multiplied(oblique_spin),
                                 True).Shape()).Shape()
    expect("cylinder cut by an oblique plane", oblique, "cell-intersection", want_leaves=2)
    # A cube with an axial through-hole: six planes that are a TGeoBBox, and a hole wall.
    drilled = BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(gp_Pnt(-2, -2, -2), 4.0, 4.0, 4.0).Shape(),
                              cyl_along(0.8, 6.0, (0, 0, -3), (0, 0, 1))).Shape()
    drilled_record = expect("cube with an axial through-hole", drilled, "cell-intersection",
                            want_leaves=2)
    check("the cube's six plane carriers folded into one TGeoBBox",
          drilled_record["accepted"]
          and drilled_record["candidate"]["leaves"][0]["type"] == "TGeoBBox"
          and drilled_record["candidate"]["notes"]["nCarriers"] == 7,
          drilled_record["description"])

    # --- rung 3 negative controls: a V notch ladder; each rung refuses, the last accepts ---
    def notched_cylinder(angle):
        def knife(sign):
            slab = BRepPrimAPI_MakeBox(gp_Pnt(1.5, -10.0, -10.0), 20.0, 20.0, 20.0).Shape()
            spin = gp_Trsf()
            spin.SetRotation(gp_Ax1(gp_Pnt(1.5, 0.0, 0.0), gp_Dir(0, 0, 1)), sign * angle)
            return BRepBuilderAPI_Transform(slab, spin, True).Shape()
        return BRepAlgoAPI_Cut(cyl, BRepAlgoAPI_Common(knife(1.0), knife(-1.0)).Shape()).Shape()

    notch_trusted = expect_single_cell_declined(
        "a cylinder with a 2e-03 rad notch (a trusted concave edge)",
        notched_cylinder(2.0e-3), "trusted concave edge")
    check("the concave decline names how many edges it counted",
          "1 trusted concave edge(s) of 9" in (notch_trusted or ""),
          (notch_trusted or "")[:120])
    # The notch above the trust filter converts as two cells.
    notch_converted = process_solid(notched_cylinder(2.0e-3), "notched cylinder (trusted)")
    check("the notch above the trust filter converts as two cells, exactly",
          notch_converted["accepted"] and notch_converted["recogniser"] == "cells-union"
          and notch_converted["candidate"]["notes"]["nCells"] == 2
          and notch_converted["acceptance"]["symmetricDifference"] == 0.0,
          f"{notch_converted['recogniser']}: {notch_converted['description']}, "
          f"dV_sym={notch_converted['acceptance']['symmetricDifference'] if notch_converted['accepted'] else 'n/a'}")
    notch_gap = expect_declined("cylinder with a 1e-05 rad notch (below the trust filter)",
                                notched_cylinder(1.0e-5), "the cell's boundary is")
    check("the gap is what refuses the notch the trust filter let through",
          "the cell's boundary is" in (notch_gap["reason"] or "")
          and notch_gap["recogniser"] is None,
          (notch_gap["reason"] or "")[-140:])
    notch_volume = expect_declined("cylinder with a 1e-06 rad notch (ten model tolerances deep)",
                                   notched_cylinder(1.0e-6), "symmetric difference")
    check("the volume is what refuses a notch too shallow for the gap to see",
          notch_volume["recogniser"] == "cell-intersection"
          and not notch_volume["accepted"],
          (notch_volume["reason"] or "")[:140])
    # One model tolerance deep must be accepted.
    expect("cylinder with a 1e-07 rad notch (one model tolerance deep)",
           notched_cylinder(1.0e-7), "cell-intersection", want_leaves=2)

    # A genuine two-cell body, asked of the cell emitter directly.
    crossed = BRepAlgoAPI_Fuse(cyl_along(1.0, 6.0, (0, 0, -3), (0, 0, 1)),
                               cyl_along(1.0, 6.0, (-3, 0, 0), (1, 0, 0))).Shape()
    crossed_cand, crossed_why = recognise.recognise_single_cell(crossed)
    check("two fused cylinders are refused by the cell emitter, naming the concave edges",
          crossed_cand is None and "trusted concave edge(s)" in (crossed_why or ""),
          (crossed_why or "")[:120])
    # An all-planar body is the prism family's, even a convex chamfered box.
    chamfer = BRepPrimAPI_MakeBox(gp_Pnt(1.2, -9.0, -9.0), 20.0, 20.0, 20.0).Shape()
    chamfer_spin = gp_Trsf()
    chamfer_spin.SetRotation(gp_Ax1(gp_Pnt(1.2, 0.0, 0.0), gp_Dir(0, 1, 0)),
                             math.radians(35.0))
    chamfered = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeBox(gp_Pnt(-2, -2, -2), 4.0, 4.0, 4.0).Shape(),
        BRepBuilderAPI_Transform(chamfer, chamfer_spin, True).Shape()).Shape()
    planar_cand, planar_why = recognise.recognise_single_cell(chamfered)
    check("an all-planar solid is handed to the prism family, not read as halfspaces",
          planar_cand is None and "belongs to the prism family" in (planar_why or ""),
          (planar_why or "")[:120])
    # The L-plate has a concave edge, so the cell emitter refuses it on that count instead.
    ell_cand, ell_why = recognise.recognise_single_cell(ell)
    check("the L-plate is refused by the cell emitter on its concave edge",
          ell_cand is None and "trusted concave edge(s)" in (ell_why or ""),
          (ell_why or "")[:120])

    # --- the floor: the parts the earlier rungs own are not intercepted ---
    for name, want in (("L-shaped plate", "rung2-xtru"),
                       ("placed Xtru (non-convex L section)", "rung2-xtru"),
                       ("stepped polycone (duplicate z planes)", "revolved-pcon"),
                       ("box", "tier1-box"), ("tube", "tier1-tube"),
                       ("rod-and-eye (two-cluster union)", "tier2-tube-union")):
        check(f"{name} is still recognised as {want}", seen_recognisers.get(name) == want,
              f"{seen_recognisers.get(name)}")

    check("every single-cell candidate matches its recorded candidate within tolerance",
          *_recorded_match(_CELL_FIXTURES, seen_candidates))

    # --- the description must refuse an ill-formed intersection ---
    unit_box = prim.leaf("TGeoBBox", {"dx": 1.0, "dy": 1.0, "dz": 1.0}, prim.identity_frame())
    hole = prim.leaf("TGeoTube", {"rmin": 0.0, "rmax": 0.5, "dz": 2.0},
                     prim.identity_frame(), True)
    for label, op, leaves in (("a single leaf", "intersection", [unit_box]),
                              ("a complement first", "intersection", [hole, unit_box]),
                              ("a complement in a union", "union", [unit_box, hole])):
        try:
            prim.candidate(op, leaves, "self-test")
            refused = False
        except ValueError:
            refused = True
        check(f"a candidate with {label} is refused", refused)

    # --- flat-CSG R1: the torus carrier ---
    def torus_at(major, minor, angle=None, origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
                 ref=(1.0, 0.0, 0.0)):
        axis = gp_Ax2(gp_Pnt(*origin), gp_Dir(*direction), gp_Dir(*ref))
        maker = (BRepPrimAPI_MakeTorus(axis, major, minor) if angle is None
                 else BRepPrimAPI_MakeTorus(axis, major, minor, angle))
        maker.Build()
        return maker.Shape()

    def expect_torus(name, solid, r, rmin, rmax, phi1=0.0, dphi=360.0):
        record = expect(name, solid, "tier1-torus")
        if not record["accepted"]:
            return record
        p = record["candidate"]["leaves"][0]["params"]
        want = {"r": r, "rmin": rmin, "rmax": rmax, "phi1": phi1, "dphi": dphi}
        worst = max(abs(p[k] - v) for k, v in want.items())
        check(f"{name} reconstructs the source TGeoTorus parameters", worst < 1.0e-9,
              f"worst parameter deviation {worst:.3g}")
        return record

    solid_torus = torus_at(4.0, 1.0)
    solid_torus_record = expect_torus("solid torus", solid_torus, 4.0, 0.0, 1.0)
    # A shell: two concentric tori of the same major radius, which is a bellows ply's section.
    ply = BRepAlgoAPI_Cut(torus_at(5.0, 0.30), torus_at(5.0, 0.28)).Shape()
    ply_record = expect_torus("torus shell (a bellows ply)", ply, 5.0, 0.28, 0.30)
    # A phi wedge, hollow, whose two cut planes pass through the axis.
    wedge_torus = BRepAlgoAPI_Cut(
        torus_at(4.0, 1.0, math.radians(120.0), ref=(math.cos(math.radians(20.0)),
                                                     math.sin(math.radians(20.0)), 0.0)),
        torus_at(4.0, 0.8, math.radians(120.0) + 1.0e-4,
                 ref=(math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0))).Shape()
    wedge_torus_record = expect_torus("hollow torus wedge", wedge_torus,
                                      4.0, 0.8, 1.0, 20.0, 120.0)
    # Placed, so the frame machinery is exercised on a torus too.
    torus_spin = gp_Trsf()
    torus_spin.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 0)), 0.7)
    torus_shift = gp_Trsf()
    torus_shift.SetTranslation(gp_Vec(3.0, -4.0, 5.0))
    placed_torus = BRepBuilderAPI_Transform(solid_torus,
                                            torus_shift.Multiplied(torus_spin), True).Shape()
    placed_torus_record = expect("placed torus", placed_torus, "tier1-torus")
    check("a placed torus travels as one leaf plus a rigid placement",
          placed_torus_record["accepted"]
          and prim.placement_for_candidate(placed_torus_record["candidate"]) is not None,
          "placement present" if placed_torus_record["accepted"] else "not accepted")

    # The torus as a cell-emitter carrier: a ply cut by a plane is a cell of two toroidal
    # halfspaces, the bore's one complemented, and one box.
    half_ply = BRepAlgoAPI_Common(
        ply, BRepPrimAPI_MakeBox(gp_Pnt(-10, -10, 0), 20.0, 20.0, 20.0).Shape()).Shape()
    half_ply_record = expect("half a bellows ply", half_ply, "cell-intersection", want_leaves=3)
    check("the ply's bore enters the cell as a complemented TGeoTorus",
          half_ply_record["accepted"]
          and sum(1 for lf in half_ply_record["candidate"]["leaves"]
                  if lf["type"] == "TGeoTorus") == 2
          and any(lf.get("outside") and lf["type"] == "TGeoTorus"
                  for lf in half_ply_record["candidate"]["leaves"]),
          half_ply_record["description"])

    # --- R1 negative controls ---
    # `torus_union_cyl` from the fixture ladder, in cm: two cells, concave on both circles.
    torus_cyl = BRepAlgoAPI_Fuse(
        torus_at(2.5, 0.8),
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -2.0), gp_Dir(0, 0, 1)),
                                 2.0, 4.0).Shape()).Shape()
    torus_cyl_why = expect_single_cell_declined("a torus fused with a coaxial cylinder through it",
                                               torus_cyl, "trusted concave edge")
    # The torus template's own verdict, asked of the template directly.
    torus_cyl_records, _why = recognise._face_records(torus_cyl)
    torus_cyl_diag = recognise._bbox_diagonal(torus_cyl)
    try:
        recognise._match_torus(torus_cyl, torus_cyl_records,
                               recognise.REL_TOL * max(torus_cyl_diag, 1.0), torus_cyl_diag)
        torus_template_why = "the template accepted it"
    except recognise.Declined as declined:
        torus_template_why = str(declined)
    check("the torus template says what it found before the cell test refuses it",
          "is not a whole torus" in torus_template_why, torus_template_why[:120])
    # A shell whose bore is displaced off axis: below tol (1.6e-05 cm) one torus, above it two cells.
    for displacement, want in ((1.0e-6, "tier1-torus"), (1.0e-5, "tier1-torus"),
                               (3.0e-5, "cell-intersection"), (1.0e-3, "cell-intersection")):
        skewed = BRepAlgoAPI_Cut(
            torus_at(5.0, 0.30),
            torus_at(5.0, 0.28, origin=(displacement, 0.0, 0.0))).Shape()
        skewed_record = process_solid(skewed, f"shell, bore {displacement:g} cm off axis")
        acceptance = skewed_record.get("acceptance") or {}
        check(f"a shell whose bore is {displacement:g} cm off the axis converts as {want}, "
              "within the band",
              skewed_record["accepted"] and skewed_record["recogniser"] == want
              and acceptance.get("symmetricDifference", 1.0) <= acceptance.get("band", 0.0),
              f"{skewed_record['recogniser']}: dV="
              f"{acceptance.get('symmetricDifference')} band={acceptance.get('band')}")
        if want == "cell-intersection":
            # And it is the concentricity test that hands it over, not an accident further on.
            records, _reason = recognise._face_records(skewed)
            skewed_diag = recognise._bbox_diagonal(skewed)
            try:
                recognise._match_torus(skewed, records,
                                       recognise.REL_TOL * max(skewed_diag, 1.0), skewed_diag)
                refused = None
            except recognise.Declined as declined:
                refused = str(declined)
            check(f"and the torus template is what refuses it at {displacement:g} cm",
                  refused is not None and "concentric" in refused,
                  refused or "IT PROPOSED ONE")

    # --- a self-intersecting fillet torus declines instead of raising ---
    blend_lobe = BRepAlgoAPI_Common(
        torus_at(0.0428825434729, 0.1),
        BRepPrimAPI_MakeBox(gp_Pnt(0.06, -1.0, -1.0), 2.0, 2.0, 2.0).Shape()).Shape()
    blend_record = expect_declined("a lobe of a self-intersecting fillet torus", blend_lobe,
                                   "self-intersecting torus")
    check("the fillet blend reaches the cell path and declines there, naming the blend",
          "as a single cell: TGeoTorus: rmax" in (blend_record["reason"] or "")
          and "fillet blend" in (blend_record["reason"] or ""),
          (blend_record["reason"] or "")[:150])
    # ... and the description layer does refuse those numbers.
    try:
        prim.leaf("TGeoTorus", {"r": 0.0428825434729, "rmin": 0.0, "rmax": 0.1,
                                "phi1": 0.0, "dphi": 360.0}, prim.identity_frame())
        refused_kind = None
    except prim.InvalidDescription:
        refused_kind = "InvalidDescription"
    except ValueError:
        refused_kind = "ValueError"
    check("the description layer refuses those numbers as an illegal solid",
          refused_kind == "InvalidDescription", f"raised {refused_kind}")
    # An illegal solid declines; a matcher bug still raises.
    for label, kind, params in (("a missing parameter", "TGeoTorus", {"r": 1.0}),
                                ("an unknown leaf type", "TGeoNotAShape", {})):
        try:
            recognise._leaf(kind, params, prim.identity_frame())
            outcome = "returned a leaf"
        except recognise.Declined:
            outcome = "declined"
        except ValueError:
            outcome = "raised"
        check(f"{label} still raises rather than declining", outcome == "raised", outcome)

    # --- flat-CSG R2: the elliptic cylinder ---
    def elliptic_cylinder(a, b, dz, ref=None):
        axis = gp_Ax2(gp_Pnt(0, 0, -dz), gp_Dir(0, 0, 1),
                      gp_Dir(*(ref if ref is not None else (1.0, 0.0, 0.0))))
        major, minor = max(a, b), min(a, b)
        edge = BRepBuilderAPI_MakeEdge(gp_Elips(axis, major, minor)).Edge()
        face = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(edge).Wire()).Face()
        prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, 2 * dz))
        prism.Build()
        return prism.Shape()

    def expect_eltu(name, solid, a, b, dz):
        record = expect(name, solid, "tier1-eltu")
        if not record["accepted"]:
            return record
        p = record["candidate"]["leaves"][0]["params"]
        worst = max(abs(p["a"] - a), abs(p["b"] - b), abs(p["dz"] - dz))
        check(f"{name} reconstructs the source TGeoEltu parameters", worst < 1.0e-9,
              f"a={p['a']:.6g} b={p['b']:.6g} dz={p['dz']:.6g}, worst {worst:.3g}")
        return record

    # Both semi-axis orders, built the way `conv_eltu` writes them.
    eltu_solid = elliptic_cylinder(3.0, 1.5, 5.0)
    eltu_record = expect_eltu("elliptic cylinder, a > b", eltu_solid, 3.0, 1.5, 5.0)
    expect_eltu("elliptic cylinder, a < b",
                elliptic_cylinder(1.5, 3.0, 5.0, ref=(0.0, 1.0, 0.0)), 1.5, 3.0, 5.0)
    # a == b is a circle, and it is still a TGeoEltu: the carrier is an extrusion, never a
    # cylinder, so nothing can confuse the two. Asserted rather than left to chance.
    circle_eltu = expect_eltu("elliptic cylinder with equal semi-axes",
                              elliptic_cylinder(2.0, 2.0, 5.0), 2.0, 2.0, 5.0)
    check("an ellipse with equal semi-axes stays a TGeoEltu and is not read as a tube",
          circle_eltu["accepted"]
          and circle_eltu["candidate"]["leaves"][0]["type"] == "TGeoEltu",
          circle_eltu["description"])
    placed_eltu = BRepBuilderAPI_Transform(elliptic_cylinder(3.0, 1.5, 5.0),
                                           torus_shift.Multiplied(torus_spin), True).Shape()
    placed_eltu_record = expect("placed elliptic cylinder", placed_eltu, "tier1-eltu")
    check("a placed elliptic cylinder travels as one leaf plus a rigid placement",
          placed_eltu_record["accepted"]
          and prim.placement_for_candidate(placed_eltu_record["candidate"]) is not None,
          "placement present" if placed_eltu_record["accepted"] else "not accepted")

    # --- R2 negative controls ---
    # An extruded B-spline racetrack, which is not an ellipse.
    racetrack = []
    for i in range(24):
        ang = 2.0 * math.pi * i / 24.0
        racetrack.append(gp_Pnt(3.0 * math.cos(ang),
                                1.5 * math.sin(ang) * (1.0 + 0.15 * math.cos(2 * ang)), -5.0))
    spline_pts = TColgp_HArray1OfPnt(1, len(racetrack))
    for i, pnt in enumerate(racetrack, start=1):
        spline_pts.SetValue(i, pnt)
    interp = GeomAPI_Interpolate(spline_pts, True, 1.0e-7)
    interp.Perform()
    oval_edge = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()
    oval_face = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(oval_edge).Wire()).Face()
    oval_prism = BRepPrimAPI_MakePrism(oval_face, gp_Vec(0, 0, 10.0))
    oval_prism.Build()
    expect_declined("extruded B-spline racetrack (not an ellipse)", oval_prism.Shape(),
                    "free-form faces")

    # --- both new templates' instruments must be able to say "no" ---
    true_eltu = prim.candidate("primitive", [prim.leaf(
        "TGeoEltu", {"a": 3.0, "b": 1.5, "dz": 5.0}, prim.identity_frame())], "self-test")
    nudged_eltu = prim.candidate("primitive", [prim.leaf(
        "TGeoEltu", {"a": 3.0 + 1.0e-6, "b": 1.5, "dz": 5.0},
        prim.identity_frame())], "self-test")
    eltu_gap = recognise._boundary_gap(prim.build_occ(true_eltu), prim.build_occ(nudged_eltu))
    check("the gap reports a semi-axis displaced by ten model tolerances",
          abs(eltu_gap - 1.0e-6) < 1.0e-9, f"gap {eltu_gap:.3g} cm, expected 1e-06 cm")
    true_torus = prim.candidate("primitive", [prim.leaf(
        "TGeoTorus", {"r": 4.0, "rmin": 0.0, "rmax": 1.0, "phi1": 0.0, "dphi": 360.0},
        prim.identity_frame())], "self-test")
    nudged_torus = prim.candidate("primitive", [prim.leaf(
        "TGeoTorus", {"r": 4.0, "rmin": 0.0, "rmax": 1.0 + 1.0e-6, "phi1": 0.0, "dphi": 360.0},
        prim.identity_frame())], "self-test")
    torus_gap = recognise._boundary_gap(prim.build_occ(true_torus), prim.build_occ(nudged_torus))
    check("the gap reports a tube radius displaced by ten model tolerances",
          abs(torus_gap - 1.0e-6) < 1.0e-9, f"gap {torus_gap:.3g} cm, expected 1e-06 cm")

    # --- the descriptions must refuse illegal parameters ---
    for label, kind, params in (
            ("a torus whose tube is wider than its major radius", "TGeoTorus",
             {"r": 1.0, "rmin": 0.0, "rmax": 2.0, "phi1": 0.0, "dphi": 360.0}),
            ("a torus with rmin above rmax", "TGeoTorus",
             {"r": 4.0, "rmin": 1.0, "rmax": 0.5, "phi1": 0.0, "dphi": 360.0}),
            ("a torus with dphi zero", "TGeoTorus",
             {"r": 4.0, "rmin": 0.0, "rmax": 1.0, "phi1": 0.0, "dphi": 0.0}),
            ("an elliptic cylinder with a zero semi-axis", "TGeoEltu",
             {"a": 0.0, "b": 1.5, "dz": 5.0})):
        try:
            prim.leaf(kind, params, prim.identity_frame())
            refused = False
        except ValueError:
            refused = True
        check(f"a description of {label} is refused", refused)

    check("every torus and elliptic-cylinder candidate matches its recorded candidate within tolerance",
          *_recorded_match(_TORUS_ELTU_FIXTURES, seen_candidates))

    # --- Tier 0: the quadric a stored B-spline face already is ---
    from cadsupport import tier0
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_GTransform, BRepBuilderAPI_NurbsConvert
    from OCC.Core.BRepTools import breptools
    from OCC.Core.GeomAbs import GeomAbs_Cylinder
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.gp import gp_Ax3, gp_Cylinder, gp_GTrsf, gp_Mat

    check("the canonicaliser's band is the cascade's own",
          tier0.REL_TOL == recognise.REL_TOL,
          f"tier0 {tier0.REL_TOL:.0e} vs recognise {recognise.REL_TOL:.0e}")

    def nurbs(shape):
        return BRepBuilderAPI_NurbsConvert(shape, True).Shape()

    def faces_of(shape):
        found = []
        walk = TopExp_Explorer(shape, TopAbs_FACE)
        while walk.More():
            found.append(topods.Face(walk.Current()))
            walk.Next()
        return found

    def samples_of(face, n):
        adaptor = BRepAdaptor_Surface(face, True)
        from cadsupport import analytic as converter
        return converter._sample_surface_for_recognition(adaptor, *breptools.UVBounds(face), n=n)

    # (c) the instrument: a model displaced by a known amount must be reported at that size.
    probe_cylinder = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                              5.0, 8.0).Shape()
    probe_points, _probe_normals = samples_of(
        [f for f in faces_of(probe_cylinder)
         if BRepAdaptor_Surface(f, True).GetType() == GeomAbs_Cylinder][0], 17)
    probe_sphere_points, _ = samples_of(faces_of(BRepPrimAPI_MakeSphere(5.0).Shape())[0], 17)
    probe_torus_points, _ = samples_of(faces_of(BRepPrimAPI_MakeTorus(6.0, 1.5).Shape())[0], 17)
    for displacement in (1.0e-3, 1.0e-6, 1.0e-9):
        for label, kind, model, points in (
                ("cylinder radius", "cylinder",
                 {"axis": [0.0, 0.0, 1.0], "origin": [0.0, 0.0, 0.0],
                  "radius": 5.0 + displacement}, probe_points),
                ("sphere radius", "sphere",
                 {"centre": [0.0, 0.0, 0.0], "radius": 5.0 + displacement},
                 probe_sphere_points),
                ("torus tube radius", "torus",
                 {"axis": [0.0, 0.0, 1.0], "centre": [0.0, 0.0, 0.0], "major": 6.0,
                  "minor": 1.5 + displacement}, probe_torus_points)):
            measured = tier0.surface_gap(kind, model, points)
            check(f"the gap reports a {label} displaced by {displacement:.0e} cm at its true size",
                  abs(measured - displacement) <= 1.0e-9 * displacement + 1.0e-13,
                  f"measured {measured:.6g} cm, displaced {displacement:.0e} cm")

    # The same solid, written as NURBS, must convert to the same body.
    tier0_pairs = (
        ("box", BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 3.0, 4.0).Shape(), "tier1-box", 1),
        ("solid cylinder", BRepPrimAPI_MakeCylinder(ax, 2.0, 10.0).Shape(), "tier1-tube", 1),
        ("tube segment", BRepPrimAPI_MakeCylinder(ax, 2.0, 10.0, math.radians(72.0)).Shape(),
         "tier1-tubeseg", 1),
        ("cone", BRepPrimAPI_MakeCone(ax, 3.0, 1.0, 6.0).Shape(), "tier1-cone", 1),
        ("sphere", BRepPrimAPI_MakeSphere(3.0).Shape(), "tier1-sphere", 1),
        ("solid torus", BRepPrimAPI_MakeTorus(6.0, 1.5).Shape(), "tier1-torus", 1),
        ("hollow torus wedge",
         BRepAlgoAPI_Cut(BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                               6.0, 1.5, math.radians(140.0)).Shape(),
                         BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                               6.0, 0.7, math.radians(140.0)).Shape()).Shape(),
         "tier1-torus", 1),
        ("cube with an axial through-hole",
         BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(gp_Pnt(-3, -3, -3), 6.0, 6.0, 6.0).Shape(),
                         BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -5), gp_Dir(0, 0, 1)),
                                                  1.5, 10.0).Shape()).Shape(),
         "cell-intersection", 2),
    )

    def realisation_gap(one, other):
        """The largest distance between the two candidates' realised boundaries, in cm."""
        if one is None or other is None:
            return float("inf")
        if (one["op"], one["recogniser"], len(one["leaves"])) != \
                (other["op"], other["recogniser"], len(other["leaves"])):
            return float("inf")
        if [lf["type"] for lf in one["leaves"]] != [lf["type"] for lf in other["leaves"]]:
            return float("inf")
        return recognise._boundary_gap(prim.build_occ(one), prim.build_occ(other))

    for label, solid, want_recogniser, want_leaves in tier0_pairs:
        native_record = process_solid(solid, f"tier0 native {label}")
        encoded = expect(f"NURBS-encoded {label}", nurbs(solid), want_recogniser, want_leaves)
        deviation = realisation_gap(native_record["candidate"], encoded["candidate"])
        notes = (encoded["candidate"] or {}).get("notes", {})
        check(f"the NURBS-encoded {label} realises the analytic one's solid",
              deviation <= 1.0e-9,
              f"{notes.get('tier0Faces', 0)} canonicalised carrier(s) at a worst gap of "
              f"{notes.get('tier0WorstGapRelative', float('nan')):.3g} of the part; the two "
              f"realisations are {deviation:.3g} cm apart")

    check("every Tier-0 candidate matches its recorded candidate within tolerance",
          *_recorded_match(_TIER0_FIXTURES, seen_candidates))

    # (a) a free-form face must not canonicalise; its decline carries the best proposal's gap.
    from cadsupport import analytic as converter
    for label, face in (
            ("free-form saddle", converter._self_test_bezier_patch(
                lambda s, t: (10 * s - 5, 10 * t - 5, (10 * s - 5) * (10 * t - 5) / 10.0), 6, 6)),
            ("narrow free-form ridge", converter._self_test_bezier_patch(
                lambda s, t: (20 * s - 10, 0.5 * t,
                              0.02 * (20 * s - 10) ** 2 + 0.3 * (20 * s - 10) * t), 6, 6)),
            ("swept non-circular profile (bulge 1e-2)",
             converter._self_test_tapered_near_circle(1.0e-2, 1.0e-4))):
        adaptor = BRepAdaptor_Surface(face, True)
        carrier, gap = tier0.canonicalise(face, adaptor, 20.0)
        check(f"a {label} is not canonicalised, and the gap says how far off it is",
              carrier is None and gap is not None and gap > 10.0 * tier0.REL_TOL * 20.0,
              f"{'declined' if carrier is None else 'ACCEPTED as ' + carrier['kind']}, best "
              f"proposal {gap:.4g} cm away, {gap / 20.0:.3g} of the part against "
              f"{tier0.REL_TOL:.0e}")

    # (b) a cylinder squashed by a `gp_GTrsf`: refused at ten tolerances, accepted at a tenth.
    squash_radius, squash_scale = 5.0, 20.0
    squash_base = nurbs(BRepBuilderAPI_MakeFace(
        gp_Cylinder(gp_Ax3(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)),
                    squash_radius), 0.0, 2.0 * math.pi, 1.0, 9.0).Shape())
    squash_measured = {}
    for multiple in (0.1, 1.0, 10.0):
        intended = multiple * tier0.REL_TOL * squash_scale
        transform = gp_GTrsf()
        transform.SetVectorialPart(gp_Mat(1.0 + 2.0 * intended / squash_radius, 0.0, 0.0,
                                          0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        squashed = faces_of(BRepBuilderAPI_GTransform(squash_base, transform, True).Shape())[0]
        carrier, gap = tier0.canonicalise(squashed, BRepAdaptor_Surface(squashed, True),
                                          squash_scale)
        squash_measured[multiple] = gap
        want_accepted = multiple < 1.0
        check(f"a disguised cylinder displaced by {multiple:g} model tolerance(s) is "
              f"{'accepted' if want_accepted else 'refused by the gap'}",
              (carrier is not None) == want_accepted,
              f"{'accepted as ' + carrier['kind'] if carrier else 'declined'}, "
              f"measured gap {gap:.4g} cm, {gap / squash_scale:.3g} of the part against "
              f"{tier0.REL_TOL:.0e}")
    ratios = [squash_measured[m] / (m * tier0.REL_TOL * squash_scale) for m in (0.1, 1.0, 10.0)]
    check("the measured gap is proportional to the displacement that caused it",
          max(ratios) - min(ratios) <= 1.0e-3 * max(ratios),
          f"gap / displacement = {', '.join(f'{r:.4f}' for r in ratios)}")

    # An empty proposal must decline as empty.
    empty_common = BRepAlgoAPI_Common(
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(9, 9, 9), 1.0, 1.0, 1.0).Shape()).Shape()
    try:
        recognise._boundary_gap(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape(),
                                empty_common)
        empty_reason = "no decline"
    except recognise.Declined as declined:
        empty_reason = str(declined)
    check("an empty proposal declines as empty, not as an OCCT measurement failure",
          "the proposal is empty" in empty_reason, empty_reason)

    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound

    # --- Rung 4: the union of cells ---
    #
    # Every body here is one whose cell count is known in closed form.
    from cadsupport import decompose as decomp

    def expect_cells(name, solid, want_cells, want_leaves=None, **kwargs):
        record = process_solid(solid, name, **kwargs)
        seen_recognisers[name] = record["recogniser"] if record["accepted"] else None
        if record["accepted"]:
            seen_candidates[name] = json.loads(json.dumps(record["candidate"], sort_keys=True))
        notes = (record["candidate"] or {}).get("notes", {})
        ok = (record["accepted"] and record["recogniser"] == "cells-union"
              and notes.get("nCells") == want_cells
              and (want_leaves is None or notes.get("nLeaves") == want_leaves))
        detail = (f"{notes.get('nCells')} cell(s) of {notes.get('cellLeaves')} leaves, "
                  f"{notes.get('nSplits')} split(s), volume drift "
                  f"{notes.get('volumeDriftRelative', float('nan')):.3g}, gap "
                  f"{notes.get('cellGapCm', float('nan')):.3g} cm"
                  if record["accepted"] else f"declined: {record['reason']}")
        check(f"{name} converts as {want_cells} cells", ok, detail)
        return record

    l_plate = BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 4.0, 4.0, 1.0).Shape(),
                              BRepPrimAPI_MakeBox(gp_Pnt(2, 2, -1), 4.0, 4.0, 3.0).Shape()).Shape()
    grooved = BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 6.0, 4.0, 3.0).Shape(),
                              BRepPrimAPI_MakeBox(gp_Pnt(2, -1, 1), 2.0, 6.0, 3.0).Shape()).Shape()
    # Prism-family parts, driven through `recognise_union_of_cells` directly for their counts.
    for label, solid, want in (("an L-plate", l_plate, 2), ("a grooved block", grooved, 3)):
        cand, why = recognise.recognise_union_of_cells(solid)
        gap = (None if cand is None else
               recognise._boundary_gap(prim.build_occ(cand), solid))
        check(f"{label} decomposes into {want} cells and realises the solid",
              cand is not None and cand["notes"]["nCells"] == want and gap <= 1.0e-9,
              (f"{cand['notes']['nCells']} cells of {cand['notes']['cellLeaves']} leaves, "
               f"{gap:.3g} cm from the part" if cand else f"declined: {why}"))

    # A hexagonal collar on a cylinder: two cells, one eight halfspaces wide.
    hex_collar = BRepAlgoAPI_Fuse(
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -5), gp_Dir(0, 0, 1)), 3.0, 5.0).Shape(),
        swept_polygon(3.0, 6, 0.0, 5.0)).Shape()
    expect_cells("a cylinder with a hexagonal collar", hex_collar, 2, want_leaves=9)

    # Two rods sharing no edge: only the connectivity split finds the two cells.
    disjoint = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(disjoint)
    builder.Add(disjoint, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1.0, 5.0).Shape())
    builder.Add(disjoint, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(6, 0, 0), gp_Dir(0, 0, 1)), 1.0, 5.0).Shape())
    disjoint_record = expect_cells("two rods sharing no edge", disjoint, 2, want_leaves=2)
    check("the disjoint pair is found by connectivity and needs no split at all",
          disjoint_record["accepted"]
          and disjoint_record["candidate"]["notes"]["nComponents"] == 2
          and disjoint_record["candidate"]["notes"]["nSplits"] == 0
          and _count_trusted_concave(disjoint) == 0,
          f"{_count_trusted_concave(disjoint)} trusted concave edge(s), "
          f"{(disjoint_record['candidate'] or {}).get('notes', {}).get('nSplits')} split(s)")

    # A torus with a cylinder through it, whose cells are not all planar.
    torus_through = BRepAlgoAPI_Fuse(
        torus_at(2.5, 0.8),
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -2.0), gp_Dir(0, 0, 1)),
                                 2.0, 4.0).Shape()).Shape()
    expect_cells("a torus with a cylinder through it", torus_through, 2, want_leaves=3)

    # (a) the volume guard: a component walk that loses one of three boxes must be refused.
    three_boxes = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(three_boxes)
    for x in (0.0, 4.0, 8.0):
        builder.Add(three_boxes, BRepPrimAPI_MakeBox(gp_Pnt(x, 0, 0), 2.0, 2.0, 2.0).Shape())
    expect_cells("three disjoint boxes", three_boxes, 3, want_leaves=3)
    intact_components = decomp.solid_components
    try:
        decomp.solid_components = lambda shape: intact_components(shape)[:-1]
        _lost, lost_why = recognise.recognise_union_of_cells(three_boxes)
    finally:
        decomp.solid_components = intact_components
    check("a decomposition that loses a cell is refused by the volume guard",
          _lost is None and "volume" in (lost_why or ""), lost_why or "ACCEPTED")
    check("the volume guard reports the drift it measured, at its true size",
          _lost is None and "0.333" in (lost_why or ""),
          f"one box of three is 1/3 of the part; the decline says: "
          f"{(lost_why or '')[:120]}")

    # (b) the budgets, each declining by name.
    _over_cells, cells_why = recognise.recognise_union_of_cells(grooved, max_cells=2)
    check("a part over the cell budget declines naming the bound",
          _over_cells is None and "cell budget of 2" in (cells_why or ""), cells_why or "ACCEPTED")
    _over_leaves, leaves_why = recognise.recognise_union_of_cells(grooved, max_leaves=2)
    check("a part over the leaf budget declines naming the bound",
          _over_leaves is None and "part budget of 2" in (leaves_why or ""),
          leaves_why or "ACCEPTED")

    # (c) the DNF is two levels and the emitter refuses a third.
    flat_cell = prim.cell("primitive", [prim.leaf("TGeoBBox", {"dx": 1.0, "dy": 1.0, "dz": 1.0},
                                                 prim.identity_frame())])
    for label, cells_in in (
            ("a cell that is itself a union",
             [flat_cell, {"op": "union", "leaves": [flat_cell["leaves"][0]] * 2}]),
            ("a cell carrying a recogniser of its own",
             [flat_cell, {"op": "primitive", "leaves": flat_cell["leaves"],
                          "recogniser": "nested"}]),
            ("a single cell called a union", [flat_cell])):
        try:
            prim.union_of_cells(cells_in, "self-test")
            refused = False
        except (ValueError, prim.InvalidDescription):
            refused = True
        check(f"a description with {label} is refused", refused)

    # N cells must give a union tree of depth ceil(log2 N).
    if with_root:
        import ROOT as _ROOT

        def union_depth(shape):
            if shape.ClassName() != "TGeoCompositeShape":
                return 0
            node = shape.GetBoolNode()
            return 1 + max(union_depth(node.GetLeftShape()), union_depth(node.GetRightShape()))

        ladder = []
        for n_cells in (2, 3, 5, 8):
            comp = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(comp)
            for i in range(n_cells):
                builder.Add(comp, BRepPrimAPI_MakeBox(gp_Pnt(4.0 * i, 0, 0),
                                                      2.0, 2.0, 2.0).Shape())
            cand, why = recognise.recognise_union_of_cells(comp)
            shape, _placement = prim.build_root(cand, f"balanced{n_cells}") if cand else (None, None)
            want = math.ceil(math.log2(n_cells))
            got = union_depth(shape) if shape is not None else -1
            ladder.append((n_cells, got, want))
            check(f"{n_cells} cells emit a balanced union tree of depth {want}", got == want,
                  f"depth {got}" if cand else f"declined: {why}")
        check("the union tree's depth is logarithmic in the cell count, not linear",
              all(got == want for _n, got, want in ladder),
              ", ".join(f"{n}->{got}" for n, got, _w in ladder))

    # A two-level description must survive the round trip through `csg_<part>.json`.
    if with_root:
        round_trip = json.loads(json.dumps(disjoint_record["candidate"]))
        rebuilt, rebuilt_placement = prim.build_root(round_trip, "roundtrip")
        direct, _direct_placement = prim.build_root(disjoint_record["candidate"], "direct")
        check("a two-level description survives the JSON round trip byte for byte",
              json.dumps(round_trip, sort_keys=True)
              == json.dumps(disjoint_record["candidate"], sort_keys=True)
              and rebuilt.ClassName() == direct.ClassName() and rebuilt_placement is None,
              f"{rebuilt.ClassName()}, placement "
              f"{'present' if rebuilt_placement else 'absent'}")
        gap = recognise._boundary_gap(prim.build_occ(round_trip),
                                      prim.build_occ(disjoint_record["candidate"]))
        check("the round-tripped description realises the same solid", gap <= 1.0e-12,
              f"{gap:.3g} cm apart")

    check("every union-of-cells candidate matches its recorded candidate within tolerance",
          *_recorded_match(_UNION_OF_CELLS_FIXTURES, seen_candidates))

    # --- the flat emitter's sign convention, measured against `recognise._cell_leaf` ----------
    import struct
    from cadsupport import decompose, flat as flatmod

    def _flat_gradient(block, point):
        """`|grad f|` at a point, for turning a quadric value into a first-order distance."""
        c = block["c"]
        x, y, z = point
        if block["kind"] == "torus":
            return 1.0                    # the torus block already IS a signed distance
        gx = 2.0 * (c[0] * x + c[1] * y + c[2] * z + c[6])
        gy = 2.0 * (c[1] * x + c[3] * y + c[4] * z + c[7])
        gz = 2.0 * (c[2] * x + c[4] * y + c[5] * z + c[8])
        return math.sqrt(gx * gx + gy * gy + gz * gz)

    def _flat_oracle(name, solid, seed=20260824, samples=4000):
        """The flat blocks of a one-cell solid, and `_cell_leaf`'s verdict on sampled points.

        Points within `REL_TOL x max(diag, 1)` of a carrier surface, or ON it, are not scored.
        """
        import random
        from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
        from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
        from OCC.Core.gp import gp_Pnt
        diag = decompose.bbox_diagonal(solid)
        tol = recognise.REL_TOL * max(diag, 1.0)
        carriers = recognise._halfspace_carriers(solid, tol)
        box = recognise._CellBox(solid, diag)
        blocks = flatmod.blocks_from_carriers(carriers)
        leaves = [recognise._cell_leaf(c, box) for c in carriers]
        cand = prim.cell("intersection" if len(leaves) > 1 else "primitive", leaves)
        classifier = BRepClass3d_SolidClassifier(prim.build_occ(cand))
        rng = random.Random(seed)
        (xlo, ylo, zlo, xhi, yhi, zhi) = recognise._bbox_of(solid)
        scored = []
        for _ in range(samples):
            point = (rng.uniform(xlo, xhi), rng.uniform(ylo, yhi), rng.uniform(zlo, zhi))
            near = min(abs(flatmod.eval_block(b, point))
                       / max(_flat_gradient(b, point), 1.0e-300) for b in blocks)
            if near <= tol:
                continue
            classifier.Perform(gp_Pnt(*point), tol)
            state = classifier.State()
            if state == TopAbs_ON:
                continue
            scored.append((point, state == TopAbs_IN))
        worst_plane = max((flatmod.plane_scaling_error(b) for b in blocks
                           if flatmod.plane_scaling_error(b) is not None), default=None)
        return {"name": name, "solid": solid, "carriers": carriers, "blocks": blocks,
                "points": scored, "kinds": sorted({c["kind"] for c in carriers}),
                "sides": sorted({c["side"] for c in carriers}), "worstPlane": worst_plane}

    def _flat_disagreements(blocks, points):
        return sum(1 for point, occ_inside in points
                   if flatmod.flat_contains(blocks, point) != occ_inside)

    flat_axis = gp_Ax2(gp_Pnt(0, 0, -5), gp_Dir(0, 0, 1))
    flat_tube = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeCylinder(flat_axis, 2.0, 10.0).Shape(),
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -6), gp_Dir(0, 0, 1)), 1.0, 12.0).Shape()
    ).Shape()
    # a box with a spherical scoop taken out of one corner: six planes and an EXTERIOR sphere
    flat_scooped = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeBox(gp_Pnt(-3, -3, -3), 6.0, 6.0, 6.0).Shape(),
        BRepPrimAPI_MakeSphere(gp_Pnt(3, 3, 3), 2.5).Shape()).Shape()
    # A cylinder about (1, 1, 1): the only fixture with off-diagonal quadric coefficients.
    flat_tilted = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(-1, -1, -1), gp_Dir(1, 1, 1)), 1.5, 6.0).Shape()
    flat_cases = (
        ("a box", BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 3.0, 4.0).Shape()),
        ("a tube, whose bore is an exterior cylinder", flat_tube),
        ("a cone frustum", BRepPrimAPI_MakeCone(flat_axis, 3.0, 1.0, 10.0).Shape()),
        ("a hemisphere", BRepAlgoAPI_Common(
            BRepPrimAPI_MakeSphere(gp_Pnt(1, 2, 3), 2.5).Shape(),
            BRepPrimAPI_MakeBox(gp_Pnt(-3, -1, 3), 9.0, 9.0, 9.0).Shape()).Shape()),
        ("a box with a spherical scoop, an exterior sphere", flat_scooped),
        ("a torus ply", BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                              4.0, 1.0).Shape()),
        ("a cylinder tilted about (1,1,1), whose quadric is dense", flat_tilted),
    )
    flat_results = [_flat_oracle(name, solid) for name, solid in flat_cases]
    for result in flat_results:
        bad = _flat_disagreements(result["blocks"], result["points"])
        check(f"the flat halfspaces of {result['name']} classify exactly as _cell_leaf's "
              "primitives",
              bad == 0 and len(result["points"]) > 0.5 * 4000,
              f"{bad} of {len(result['points'])} scored points disagree; carriers "
              f"{'+'.join(result['kinds'])} ({'+'.join(result['sides'])})")

    # All five carrier kinds must be covered.
    flat_kinds_seen = sorted({k for r in flat_results for k in r["kinds"]})
    check("the flat oracle comparison covers all five carrier kinds",
          flat_kinds_seen == ["cone", "cylinder", "plane", "sphere", "torus"],
          f"covered {flat_kinds_seen}")
    check("the flat oracle comparison exercises a complemented (exterior) carrier",
          any("exterior" in r["sides"] for r in flat_results),
          "; ".join(f"{r['name']}: {'+'.join(r['sides'])}" for r in flat_results))
    # and a quadric with genuinely non-zero off-diagonal terms, per the note on `flat_tilted`
    flat_dense = [r["name"] for r in flat_results
                  if any(b["kind"] == "quadric" and max(abs(b["c"][1]), abs(b["c"][2]),
                                                        abs(b["c"][4])) > 1.0e-3
                         for b in r["blocks"])]
    check("the flat oracle comparison exercises off-diagonal quadric coefficients",
          bool(flat_dense), f"dense-quadric fixtures: {flat_dense}")

    # The negative control: inverting any one halfspace of any fixture must be caught.
    flat_missed = []
    for result in flat_results:
        for index in range(len(result["blocks"])):
            flipped = [dict(b, sign=-b["sign"]) if i == index else b
                       for i, b in enumerate(result["blocks"])]
            if _flat_disagreements(flipped, result["points"]) == 0:
                flat_missed.append(f"{result['name']}[{index}]")
    flat_flips = sum(len(r["blocks"]) for r in flat_results)
    check("inverting any one halfspace of any fixture is caught by the same comparison",
          flat_flips > 0 and not flat_missed,
          f"{flat_flips} inversion(s) over {len(flat_results)} fixtures, missed {flat_missed}")

    # --- the cone's mirror nappe beyond the apex, outside the sampled box --------------------
    flat_cone_result = next(r for r in flat_results if r["name"] == "a cone frustum")
    flat_cone_carrier = next(c for c in flat_cone_result["carriers"] if c["kind"] == "cone")
    flat_apex, flat_k = flatmod.cone_apex(flat_cone_carrier)
    flat_axis_d = flat_cone_carrier["d"]
    flat_ref = flat_cone_carrier["x"]

    def _flat_along_apex(steps, radial=0.0):
        """A point `steps` along the axis from the apex, positive being the material side."""
        walk = math.copysign(1.0, flat_k) * steps
        return tuple(flat_apex[i] + walk * flat_axis_d[i] + radial * flat_ref[i]
                     for i in range(3))

    # a point strictly inside the mirror nappe: |r + k u| = |k| * 5 there, and the radius is half
    flat_mirror = _flat_along_apex(-5.0, radial=0.5 * abs(flat_k) * 5.0)
    flat_real = _flat_along_apex(5.0, radial=0.5 * abs(flat_k) * 5.0)
    # the emitter's contract for ONE cone carrier, isolated from the fixture's caps
    flat_cone_blocks = flatmod.blocks_from_carriers([flat_cone_carrier])
    flat_cone_quadric = [b for b in flat_cone_blocks
                         if not (b["kind"] == "quadric" and all(b["c"][i] == 0.0
                                                                for i in range(6)))]
    check("the cone quadric alone would admit a point on the mirror nappe",
          len(flat_cone_quadric) == 1 and flatmod.flat_contains(flat_cone_quadric, flat_mirror),
          f"the point {tuple(round(v, 6) for v in flat_mirror)} beyond the apex "
          f"{tuple(round(v, 6) for v in flat_apex)}")
    check("the emitted interior cone excludes the mirror nappe beyond its apex",
          len(flat_cone_blocks) == 2
          and not flatmod.flat_contains(flat_cone_blocks, flat_mirror),
          f"{len(flat_cone_blocks)} block(s) for one carrier, apex plane included")
    check("the apex plane cuts nothing on the cone's real nappe",
          flatmod.flat_contains(flat_cone_blocks, flat_real),
          f"the mirrored point {tuple(round(v, 6) for v in flat_real)} is still material")
    flat_apex_plane = flatmod.cone_apex_plane(flat_cone_carrier)
    check("the apex plane obeys the 2b = n convention like any other plane",
          flatmod.plane_scaling_error(flat_apex_plane) < 1.0e-15,
          f"residual {flatmod.plane_scaling_error(flat_apex_plane)}")

    # An exterior cone gets no apex plane; `check_cell_box` declines it past the apex.
    flat_exterior_cone = dict(flat_cone_carrier, side="exterior")
    check("an exterior cone is not silently given an apex plane",
          flatmod.cone_apex_plane(flat_exterior_cone) is None,
          "cone_apex_plane declines to repair a complemented cone")

    def _flat_cube_at(centre, half=0.5):
        return ([centre[i] - half for i in range(3)], [centre[i] + half for i in range(3)])

    flat_past_lo, flat_past_hi = _flat_cube_at(_flat_along_apex(-5.0))
    try:
        flatmod.check_cell_box([flat_exterior_cone], flat_past_lo, flat_past_hi)
        flat_box_reason = ""
    except recognise.Declined as why:
        flat_box_reason = str(why)
    check("an exterior cone whose cell box reaches past its apex is declined",
          "mirror nappe" in flat_box_reason, f"reason: {flat_box_reason or 'nothing raised'}")
    flat_short_lo, flat_short_hi = _flat_cube_at(_flat_along_apex(5.0))
    try:
        flatmod.check_cell_box([flat_exterior_cone], flat_short_lo, flat_short_hi)
        flat_stay_ok = True
    except recognise.Declined:
        flat_stay_ok = False
    check("an exterior cone whose cell box stays short of its apex is not declined",
          flat_stay_ok, f"box {tuple(round(v, 3) for v in flat_short_lo)} .. "
                        f"{tuple(round(v, 3) for v in flat_short_hi)}")
    # and an INTERIOR cone is never refused by that check, since its apex plane already fixed it
    try:
        flatmod.check_cell_box([flat_cone_carrier], flat_past_lo, flat_past_hi)
        flat_interior_ok = True
    except recognise.Declined:
        flat_interior_ok = False
    check("an interior cone is not refused for reaching past its apex",
          flat_interior_ok, "the apex plane already removed the mirror nappe")

    # The plane convention |2b| = 1, asserted where the planes are created.
    flat_plane_worst = max((r["worstPlane"] for r in flat_results
                            if r["worstPlane"] is not None), default=None)
    check("every emitted plane block stores 2b = n for a unit normal",
          flat_plane_worst is not None and flat_plane_worst < 1.0e-15,
          f"worst | |2b| - 1 | over the fixtures: "
          f"{'no plane blocks' if flat_plane_worst is None else f'{flat_plane_worst:.3g}'}")
    # negative control on that check itself: a plane rescaled by 3 must be caught
    flat_tripled = {"kind": "quadric", "sign": 1.0,
                    "c": [0.0] * 6 + [1.5, 0.0, 0.0, -3.0, 0.0]}
    check("a plane block rescaled by three is refused by the convention check",
          abs(flatmod.plane_scaling_error(flat_tripled) - 2.0) < 1.0e-15,
          f"residual {flatmod.plane_scaling_error(flat_tripled)}")

    # A carrier kind with no quadric form declines rather than emitting a wrong halfspace.
    try:
        flatmod.quadric_from_carrier({"kind": "torus", "side": "interior"})
        flat_declined = ""
    except recognise.Declined as why:
        flat_declined = str(why)
    check("a carrier with no quadric form is declined, not guessed at",
          "no quadric form" in flat_declined, f"reason: {flat_declined or 'nothing raised'}")

    # A torus axis is normalised on the way into a block, as `AddTorus` does on load.
    flat_long_axis = flatmod.blocks_from_carriers(
        [{"kind": "torus", "side": "interior", "p": (0.0, 0.0, 0.0), "d": (0.0, 0.0, 3.0),
          "r": 4.0, "rt": 1.0}])[0]
    check("a torus block's axis is a unit vector whatever the carrier carried",
          abs(math.sqrt(sum(flat_long_axis["c"][3 + i] ** 2 for i in range(3))) - 1.0) < 1.0e-15,
          f"axis {tuple(flat_long_axis['c'][3:6])}")

    # Sidecar record sizes: 20-byte header, 100-byte halfspace, 64-byte cell, little-endian.
    flat_sidecar = Path("/tmp/csg_selftest_flatcsg.bin")
    flat_probe_blocks = flat_results[1]["blocks"]
    flat_probe_cells = [{"first": 0, "count": len(flat_probe_blocks), "volume": 1.5,
                         "lo": [-2.0, -2.0, -5.0], "hi": [2.0, 2.0, 5.0]}]
    flatmod.write_sidecar(flat_sidecar, flat_probe_blocks, flat_probe_cells)
    flat_bytes = flat_sidecar.read_bytes()
    check("the sidecar is magic + version + two counts + fixed-length records",
          len(flat_bytes) == 20 + 100 * len(flat_probe_blocks) + 64 * len(flat_probe_cells)
          and flat_bytes[:8] == flatmod.SIDECAR_MAGIC
          and struct.unpack("<III", flat_bytes[8:20]) == (flatmod.SIDECAR_VERSION,
                                                          len(flat_probe_blocks),
                                                          len(flat_probe_cells)),
          f"{len(flat_bytes)} bytes for {len(flat_probe_blocks)} halfspace(s) and "
          f"{len(flat_probe_cells)} cell(s)")
    # and it round-trips through the format's own reader, field by field
    flat_read_back = []
    for index in range(len(flat_probe_blocks)):
        at = 20 + 100 * index
        kind = struct.unpack("<i", flat_bytes[at:at + 4])[0]
        sign = struct.unpack("<d", flat_bytes[at + 4:at + 12])[0]
        coeff = struct.unpack("<11d", flat_bytes[at + 12:at + 100])
        flat_read_back.append((kind, sign, coeff))
    check("every halfspace block reads back from the sidecar bit for bit",
          all(rb[0] == (1 if b["kind"] == "torus" else 0) and rb[1] == b["sign"]
              and list(rb[2]) == list(b["c"]) + [0.0] * (11 - len(b["c"]))
              for rb, b in zip(flat_read_back, flat_probe_blocks)),
          f"{len(flat_read_back)} block(s)")
    flat_cell_at = 20 + 100 * len(flat_probe_blocks)
    flat_cell_read = struct.unpack("<iid3d3d", flat_bytes[flat_cell_at:flat_cell_at + 64])
    check("the cell record reads back from the sidecar bit for bit",
          flat_cell_read == (0, len(flat_probe_blocks), 1.5,
                             -2.0, -2.0, -5.0, 2.0, 2.0, 5.0),
          f"{flat_cell_read}")

    # --- R5: the flat path takes what the tree budget refuses, and nothing else ---------------
    import itertools
    import random as flat_random
    tree_declined, tree_why = recognise.recognise_union_of_cells(hex_collar, max_leaves=8)
    check("a part over the tree's leaf budget still declines on the tree path",
          tree_declined is None and "part budget of 8" in (tree_why or ""),
          tree_why or "ACCEPTED")
    flat_record, flat_why = recognise.recognise_flat_cells(hex_collar)
    check("the same part is accepted on the flat path",
          flat_record is not None and flat_record["recogniser"] == "flat-cells", flat_why)
    check("the flat record carries a bounding box per cell",
          flat_record is not None
          and all(len(c["lo"]) == 3 and len(c["hi"]) == 3 for c in flat_record["cells"]),
          "a cell is missing its box")
    check("the flat description carries no leaves and no op on a cell",
          flat_record is not None and "leaves" not in flat_record
          and all(set(c) == set(prim.FLAT_CELL_KEYS) for c in flat_record["cells"]),
          f"{sorted(flat_record) if flat_record else None}")

    over_cells, over_why = recognise.recognise_flat_cells(hex_collar, max_cells=1)
    check("a part over the FLAT cell budget declines naming the bound",
          over_cells is None and "flat part budget of 1 cells" in (over_why or ""),
          over_why or "ACCEPTED")
    over_halfspaces, over_hs_why = recognise.recognise_flat_cells(hex_collar, max_halfspaces=3)
    check("a part over the FLAT halfspace budget declines naming the bound",
          over_halfspaces is None and "flat part budget of 3 halfspaces" in (over_hs_why or ""),
          over_hs_why or "ACCEPTED")

    # the containment corroboration must run on the flat path too
    check("the flat path runs the containment corroboration",
          "containsScored" in (flat_record or {}).get("notes", {})
          and (flat_record or {})["notes"].get("containsScored", 0) > 0,
          "no containment corroboration on the flat record")

    # Routing: the flat path runs only after the union path declines.
    routed, _routed_why = recognise.recognise(hex_collar)
    check("a part the tree path accepts is NOT intercepted by the flat path",
          routed is not None and routed["recogniser"] == "cells-union",
          routed["recogniser"] if routed else "declined")

    # --- R5: an exterior cone judged against its cell box, not the part box -----------------
    l_cone_solid = BRepAlgoAPI_Cut(
        BRepAlgoAPI_Fuse(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 10.0, 4.0, 4.0).Shape(),
                         BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 4), 4.0, 4.0, 6.0).Shape()).Shape(),
        BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(7, 2, 0), gp_Dir(0, 0, 1)),
                             1.5, 0.0, 8.0).Shape()).Shape()
    cone_record, cone_why = recognise.recognise_flat_cells(l_cone_solid)
    check("a multi-cell part with an exterior cone converts on the flat path",
          cone_record is not None and cone_record["notes"]["nCells"] == 2
          and cone_record["notes"]["cellGapCm"] <= 1.0e-9,
          cone_why or f"{cone_record['notes']['nCells']} cells, gap "
                      f"{cone_record['notes']['cellGapCm']:.3g} cm")

    # the cell's own box accepts, the part's box refuses
    l_cone_diag = recognise._bbox_diagonal(l_cone_solid)
    l_cone_tol = recognise.REL_TOL * max(l_cone_diag, 1.0)
    l_cone_report = decomp.split_into_cells(l_cone_solid, scale=max(l_cone_diag, 1.0))
    l_cone_part_box = recognise._bbox_of(l_cone_solid)
    cone_own, cone_part = [], []
    for piece in l_cone_report["pieces"]:
        _lv, piece_carriers, _out = recognise._cell_leaves(
            piece, l_cone_tol, decomp.bbox_diagonal(piece), whole_part=False)
        if not any(c["kind"] == "cone" and c["side"] == "exterior" for c in piece_carriers):
            continue
        piece_lo, piece_hi = recognise._flat_cell_box(
            piece, recognise._FLAT_BOX_MARGIN * max(l_cone_diag, 1.0))
        for label, box_lo, box_hi in (("own", piece_lo, piece_hi),
                                      ("part", list(l_cone_part_box[:3]),
                                       list(l_cone_part_box[3:]))):
            try:
                flatmod.check_cell_box(piece_carriers, box_lo, box_hi)
                (cone_own if label == "own" else cone_part).append("accepted")
            except recognise.Declined:
                (cone_own if label == "own" else cone_part).append("declined")
    check("the exterior cone is judged against its CELL's box, which the PART's box would fail",
          cone_own == ["accepted"] and cone_part == ["declined"],
          f"own box {cone_own}, part box {cone_part}")

    # and the call site really does hand `check_cell_box` the boxes it writes, per cell
    seen_boxes = []
    intact_check = flatmod.check_cell_box
    try:
        def _recording_check(carriers, lo, hi):
            seen_boxes.append(([float(v) for v in lo], [float(v) for v in hi]))
            return intact_check(carriers, lo, hi)
        flatmod.check_cell_box = _recording_check
        boxed_record, _boxed_why = recognise.recognise_flat_cells(l_cone_solid)
    finally:
        flatmod.check_cell_box = intact_check
    check("check_cell_box is called once per cell with exactly the box the sidecar carries",
          boxed_record is not None
          and len(seen_boxes) == len(boxed_record["cells"])
          and all(seen == ([float(v) for v in c["lo"]], [float(v) for v in c["hi"]])
                  for seen, c in zip(seen_boxes, boxed_record["cells"])),
          f"{len(seen_boxes)} call(s) for "
          f"{len(boxed_record['cells']) if boxed_record else '?'} cell(s)")

    # a `check_cell_box` refusal becomes a decline naming the cell
    try:
        def _refusing_check(carriers, lo, hi):
            raise recognise.Declined("a self-test refusal from check_cell_box")
        flatmod.check_cell_box = _refusing_check
        refused, refused_why = recognise.recognise_flat_cells(l_cone_solid)
    finally:
        flatmod.check_cell_box = intact_check
    check("a check_cell_box refusal becomes a decline naming the cell it came from",
          refused is None and "a self-test refusal from check_cell_box" in (refused_why or "")
          and "cell 1 of 2" in (refused_why or ""), refused_why or "ACCEPTED")

    # --- R5: the cell bounding box is an outer bound, checked rather than assumed -------------
    box_escapes = []
    for record_label, record_cand in (("hex collar", flat_record), ("L with a cone", cone_record)):
        for index, c in enumerate(record_cand["cells"]):
            span = [c["hi"][i] - c["lo"][i] for i in range(3)]
            rng = flat_random.Random(90210 + index)
            for _ in range(3000):
                point = tuple(c["lo"][i] - span[i] + rng.random() * 3.0 * span[i]
                              for i in range(3))
                inside_box = all(c["lo"][i] <= point[i] <= c["hi"][i] for i in range(3))
                if not inside_box and flatmod.flat_contains(c["blocks"], point):
                    box_escapes.append(f"{record_label} cell {index}")
                    break
    check("no cell reaches outside the bounding box its record declares",
          not box_escapes, "; ".join(box_escapes) or "2 records, 4 cells, 12000 points sampled")
    escaped = None
    try:
        # one plane, `x <= 0`: an unbounded cell, and the box cannot hold it
        recognise._flat_box_holds_cell(
            [{"kind": "quadric", "sign": 1.0, "c": [0.0] * 6 + [0.5, 0.0, 0.0, 0.0] + [0.0]}],
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    except recognise.Declined as declined:
        escaped = str(declined)
    check("the outward probe catches a cell that is not closed up by its own halfspaces",
          escaped is not None and "do not close the cell up" in escaped
          and "Widening the declared box" in escaped,
          escaped or "ACCEPTED an unbounded cell")

    # a corroboration that scored no point is a decline
    intact_disagreements = accept.contains_disagreements
    try:
        accept.contains_disagreements = lambda *args, **kwargs: (0, 0, 0.0)
        empty_scored, empty_why = recognise.recognise_flat_cells(l_cone_solid)
    finally:
        accept.contains_disagreements = intact_disagreements
    check("a containment corroboration that scored no point is a decline, not a pass",
          empty_scored is None and "scored no point" in (empty_why or ""),
          empty_why or "ACCEPTED on an empty measurement")

    # --- R5: the twin-parity gate, in the live and the deferred --from-json emission paths ----
    if with_root:
        import copy
        import tempfile
        import ROOT
        from cadsupport import emit as emit_mod, hook as hook_mod
        ROOT.gROOT.SetBatch(True)

        # A cell whose declared box does not contain it: the lower arm's box cut off at x = 5.
        out_of_box = copy.deepcopy(cone_record)
        wide_cell = max(range(len(out_of_box["cells"])),
                        key=lambda i: out_of_box["cells"][i]["hi"][0])
        out_of_box["cells"][wide_cell]["hi"][0] = 5.0

        # A debug build's `CloseShape` aborts on that cell, detected by its message in the library.
        marker = b"a cell reaches past the bounding box SetCellBBox was"
        library = Path(f"{ROOT.gSystem.Getenv('O2_ROOT')}/lib/libO2CADSupport.so")
        asserts_compiled = library.exists() and marker in library.read_bytes()

        def _gate_probe(folder, candidate, patched_parity=None):
            """Run both emission paths over one candidate; returns their two verdicts."""
            folder = Path(folder)
            live = folder / "live"
            deferred = folder / "deferred"
            live.mkdir(parents=True, exist_ok=True)
            deferred.mkdir(parents=True, exist_ok=True)
            intact_process = emit_mod.process_solid
            intact_parity = emit_mod.twin_parity
            try:
                emit_mod.process_solid = lambda solid, name, **kw: {
                    "part": name, "recognised": True, "accepted": True, "candidate": candidate,
                    "reason": None, "recogniser": "flat-cells",
                    "description": prim.describe(candidate),
                    "acceptance": {"accepted": True, "symmetricDifference": 0.0, "band": 1.0,
                                   "relativeToVolume": 0.0}}
                if patched_parity is not None:
                    emit_mod.twin_parity = lambda shape, **kw: patched_parity
                csg_files, flat_files, records = hook_mod.recognise_and_emit(
                    {"probe": l_cone_solid}, {"probe": "probe"}, 1.0, live,
                    lambda name: str(name), verbose=False)
                (deferred / "csg_probe.json").write_text(json.dumps(
                    {"part": "probe", "lid": "probe", "candidate": candidate,
                     "acceptance": {}, "recogniser": "flat-cells", "placement": None}))
                written, refused = emit_mod.from_json(deferred, quiet=True)
            finally:
                emit_mod.process_solid = intact_process
                emit_mod.twin_parity = intact_parity
            return {"record": records[0], "csgFiles": csg_files, "flatFiles": flat_files,
                    "liveArtifacts": sorted(p.name for p in live.glob("*")
                                            if p.suffix in (".root", ".bin")),
                    "written": written, "refused": refused,
                    "deferredArtifacts": sorted(p.name for p in deferred.glob("*")
                                                if p.suffix in (".root", ".bin"))}

        # (a) the sound candidate must still pass both paths
        with tempfile.TemporaryDirectory() as folder:
            good = _gate_probe(folder, cone_record)
        check("a sound flat candidate is emitted by both paths",
              good["record"]["accepted"] and good["record"].get("flatSidecar")
              and good["flatFiles"] and not good["csgFiles"]
              and len(good["written"]) == 1 and not good["refused"]
              and good["record"]["twinParity"]["disagreements"] == 0
              and "flatcsg_probe.bin" in good["deferredArtifacts"],
              f"live {good['liveArtifacts']}, deferred {good['deferredArtifacts']}")

        # (b) the gate's REJECT branch, driven by a parity count, in both paths
        with tempfile.TemporaryDirectory() as folder:
            forced = _gate_probe(folder, cone_record,
                                 patched_parity={"points": 20000, "disagreements": 37,
                                                 "insideAccelerated": 4000, "growFactor": 1.0})
        record = forced["record"]
        check("a twin disagreement drops the part a tier on the live path",
              not record["accepted"] and record["shape"] is None
              and record.get("flatSidecar") is None
              and "_Loop twin" in (record["reason"] or "") and "37 of 20000" in (record["reason"] or "")
              and not forced["flatFiles"] and not forced["csgFiles"]
              and forced["liveArtifacts"] == [],
              f"accepted={record['accepted']}, sidecar={record.get('flatSidecar')}, "
              f"artifacts {forced['liveArtifacts']}, reason {(record['reason'] or '')[:80]}")
        check("a twin disagreement refuses the part on the deferred --from-json path",
              not forced["written"] and len(forced["refused"]) == 1
              and forced["deferredArtifacts"] == [],
              f"written {forced['written']}, refused {len(forced['refused'])}, "
              f"artifacts {forced['deferredArtifacts']}")

        # (c) and the gate detects the geometric condition itself
        if asserts_compiled:
            check("a cell outside its declared box is caught before it can ship",
                  True,
                  "not exercised here: this build compiles O2FlatCSG::CloseShape's own "
                  "debug-build sampler for the same condition, which aborts the process rather "
                  "than returning, so the shape cannot be built to be measured")
            check("the out-of-box candidate is refused by both emission paths", True,
                  "not exercised here: same reason")
        else:
            broken_shape, _broken_placement = prim.build_root(out_of_box, "probe_out_of_box")
            # Pinned, not defaulted: this is a negative control and its sensitivity must not
            # move with `_TWIN_PARITY_PER_CELL` or with the fixture's cell count.
            broken_parity = emit_mod.twin_parity(broken_shape, n_points=20000)
            check("a cell outside its declared box is caught before it can ship",
                  broken_parity["disagreements"] > 0
                  and broken_parity["insideAccelerated"] > 0,
                  f"{broken_parity['disagreements']} of {broken_parity['points']} points "
                  f"disagree ({broken_parity['insideAccelerated']} inside the accelerated shape)")
            with tempfile.TemporaryDirectory() as folder:
                real = _gate_probe(folder, out_of_box)
            check("the out-of-box candidate is refused by both emission paths",
                  not real["record"]["accepted"] and real["record"]["shape"] is None
                  and real["record"].get("flatSidecar") is None
                  and real["liveArtifacts"] == [] and not real["written"]
                  and len(real["refused"]) == 1 and real["deferredArtifacts"] == [],
                  f"live {real['liveArtifacts']}, deferred {real['deferredArtifacts']}, "
                  f"reason {(real['record']['reason'] or '')[:90]}")

    # --- R5: the sidecar the macro loads, and the shape the gate scores, are one solid ---------
    flat_blocks, flat_sidecar_cells = prim.flat_sidecar_records(cone_record)
    check("the sidecar's cell table indexes its concatenated halfspace blocks",
          len(flat_blocks) == cone_record["notes"]["nHalfspaces"]
          and [c["count"] for c in flat_sidecar_cells]
              == [len(c["blocks"]) for c in cone_record["cells"]]
          and [c["first"] for c in flat_sidecar_cells]
              == list(itertools.accumulate([0] + [len(c["blocks"])
                                                  for c in cone_record["cells"]][:-1]))
          and all(c["volume"] > 0.0 for c in flat_sidecar_cells),
          f"{len(flat_blocks)} block(s), {len(flat_sidecar_cells)} cell(s)")

    # --- the ROOT half: the emitted TGeoShape must answer like the closed form ---
    if with_root:
        import ROOT
        ROOT.gROOT.SetBatch(True)
        from array import array
        import random
        shape, placement = prim.build_root(moved_record["candidate"], "probe_moved")
        # A placed primitive is the bare primitive plus a transform, not a composite.
        check("a rotated, translated tube emits a bare TGeoTube, not a TGeoCompositeShape",
              shape.ClassName() == "TGeoTube" and placement is not None,
              f"{shape.ClassName()}, placement {'present' if placement else 'absent'}")
        # closed form for the placed tube: 1 <= r <= 2, |z| <= 5 in the tube's frame.
        frame = moved_record["candidate"]["leaves"][0]["frame"]
        bad = 0
        random.seed(11)
        for _ in range(20000):
            p = (random.uniform(-2, 8), random.uniform(-9, 1), random.uniform(0, 10))
            rel = prim._sub(p, tuple(frame["origin"]))
            zc = prim._dot(rel, tuple(frame["z"]))
            rc = math.sqrt(max(prim._dot(rel, rel) - zc * zc, 0.0))
            want = (1.0 <= rc <= 2.0) and abs(zc) <= 5.0
            got = bool(shape.Contains(array("d", list(prim.placement_to_local(placement, p)))))
            if want != got and min(abs(rc - 1.0), abs(rc - 2.0), abs(abs(zc) - 5.0)) > 1e-9:
                bad += 1
        check("the emitted placed tube answers Contains like the closed form",
              bad == 0, f"{bad} disagreement(s) over 20000 points")
        # An analytic Capacity(): pi (rmax^2 - rmin^2) 2 dz, invariant under the placement.
        want_capacity = math.pi * (2.0 ** 2 - 1.0 ** 2) * 10.0
        rel_capacity = abs(shape.Capacity() - want_capacity) / want_capacity
        check("the placed tube's Capacity() is analytic", rel_capacity < 1.0e-14,
              f"{shape.Capacity():.12f} vs {want_capacity:.12f}, rel {rel_capacity:.2e}")
        # negative control on that check itself
        wrong, wrong_pl = prim.build_root(prim.candidate("primitive", [prim.leaf(
            "TGeoTube", {"rmin": 1.0, "rmax": 2.05, "dz": 5.0}, frame)], "probe"), "probe_wrong")
        bad_wrong = 0
        random.seed(11)
        for _ in range(20000):
            p = (random.uniform(-2, 8), random.uniform(-9, 1), random.uniform(0, 10))
            rel = prim._sub(p, tuple(frame["origin"]))
            zc = prim._dot(rel, tuple(frame["z"]))
            rc = math.sqrt(max(prim._dot(rel, rel) - zc * zc, 0.0))
            want = (1.0 <= rc <= 2.0) and abs(zc) <= 5.0
            if want != bool(wrong.Contains(array("d", list(prim.placement_to_local(wrong_pl, p))))):
                bad_wrong += 1
        check("the same check does report a wrong radius", bad_wrong > 0,
              f"{bad_wrong} disagreement(s) with rmax 2.05")
        # ... and transposing the placement rotation has to move the count.
        transposed = [[placement[r][c] for r in range(3)] + [placement[c][3]] for c in range(3)]
        bad_transposed = 0
        random.seed(11)
        for _ in range(20000):
            p = (random.uniform(-2, 8), random.uniform(-9, 1), random.uniform(0, 10))
            rel = prim._sub(p, tuple(frame["origin"]))
            zc = prim._dot(rel, tuple(frame["z"]))
            rc = math.sqrt(max(prim._dot(rel, rel) - zc * zc, 0.0))
            want = (1.0 <= rc <= 2.0) and abs(zc) <= 5.0
            got = bool(shape.Contains(array("d", list(prim.placement_to_local(transposed, p)))))
            if want != got:
                bad_transposed += 1
        check("a transposed placement rotation does move the count", bad_transposed > 0,
              f"{bad_transposed} disagreement(s) with R^T")
        # the round trip through the artefact: placement written, placement read back
        placed_target = Path("/tmp/csg_selftest_placed.root")
        write_shape_root(moved_record["candidate"], placed_target)
        fp = ROOT.TFile.Open(str(placed_target))
        back_shape = fp.Get("shape")
        back_matrix = fp.Get("placement")
        back_placement = prim.placement_from_root_matrix(back_matrix) if back_matrix else None
        worst_pl = (max(abs(back_placement[r][c] - placement[r][c])
                        for r in range(3) for c in range(4))
                    if back_placement is not None else float("inf"))
        check("shape_<part>.root round-trips the placement under the key \"placement\"",
              back_shape is not None and back_shape.ClassName() == "TGeoTube"
              and worst_pl < 1.0e-15,
              f"read {back_shape.ClassName() if back_shape else 'nothing'}, worst placement "
              f"element deviation {worst_pl:.3g}")
        fp.Close()
        # the two-leaf union must round-trip through a file and keep its class
        target = Path("/tmp/csg_selftest_shape.root")
        written = write_shape_root(ram_record["candidate"], target)
        f = ROOT.TFile.Open(str(target))
        back = f.Get("shape")
        check("a two-leaf union round-trips through shape_<part>.root",
              back and back.InheritsFrom("TGeoShape"),
              f"wrote {written.ClassName()}, read {back.ClassName() if back else 'nothing'}")
        f.Close()
        dev = crosscheck_bbox(ram_record["candidate"])
        check("the OCCT and ROOT realisations agree on the bounding box", dev < 1.0e-9,
              f"max deviation {dev:.3g} cm")

        # An axis-aligned box must come out as a bare TGeoBBox carrying its own origin.
        box_record = process_solid(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 3.0, 4.0).Shape(),
                                   "box-emission")
        box_shape, box_placement = prim.build_root(box_record["candidate"], "boxprobe")
        origin = [box_shape.GetOrigin()[i] for i in range(3)]
        check("an axis-aligned box emits a bare TGeoBBox with its own origin",
              box_shape.ClassName() == "TGeoBBox" and box_placement is None
              and max(abs(origin[0] - 1.0), abs(origin[1] - 1.5), abs(origin[2] - 2.0)) < 1e-12
              and abs(box_shape.Capacity() - 24.0) < 1e-12,
              f"{box_shape.ClassName()}, origin {origin}, capacity {box_shape.Capacity():.6f}, "
              f"placement {'present' if box_placement else 'absent'}")

        # A genuine multi-leaf boolean stays an unplaced composite.
        ram_shape, ram_placement = prim.build_root(ram_record["candidate"], "ramprobe")
        check("a genuine two-leaf union is still an unplaced TGeoCompositeShape",
              ram_shape.ClassName() == "TGeoCompositeShape" and ram_placement is None,
              f"{ram_shape.ClassName()}, placement "
              f"{'present' if ram_placement else 'absent'}")

        # --- the ROOT half of the revolved matcher ---
        stepped_record = process_solid(stepped, "pcon-emission")
        pcon_shape, pcon_placement = prim.build_root(stepped_record["candidate"], "pconprobe")
        # 100 pi: pi (3^2 - 1^2) 5 below z = 0 and pi (4^2 - 2^2) 5 above it.
        want_capacity = math.pi * ((3.0 ** 2 - 1.0 ** 2) * 5.0 + (4.0 ** 2 - 2.0 ** 2) * 5.0)
        rel_capacity = abs(pcon_shape.Capacity() - want_capacity) / want_capacity
        check("an axis-aligned polycone emits a bare TGeoPcon with an analytic Capacity()",
              pcon_shape.ClassName() == "TGeoPcon" and pcon_placement is None
              and rel_capacity < 1.0e-14,
              f"{pcon_shape.ClassName()}, capacity {pcon_shape.Capacity():.9f} vs "
              f"{want_capacity:.9f} (rel {rel_capacity:.2e}), placement "
              f"{'present' if pcon_placement else 'absent'}")

        placed_pcon_shape, placed_pcon_placement = prim.build_root(
            moved_pcon_record["candidate"], "placedpconprobe")
        check("a placed polycone is a bare TGeoPcon plus a placement",
              placed_pcon_shape.ClassName() == "TGeoPcon" and placed_pcon_placement is not None,
              f"{placed_pcon_shape.ClassName()}, placement "
              f"{'present' if placed_pcon_placement else 'absent'}")
        # The closed form uses the inverse of the transform that built the OCCT solid.
        pcon_inverse = pcon_place.Inverted()
        bad_pcon = 0
        scored_pcon = 0
        random.seed(23)
        for _ in range(20000):
            p3 = (random.uniform(-3, 9), random.uniform(-10, 2), random.uniform(-1, 11))
            probe = gp_Pnt(*p3)
            probe.Transform(pcon_inverse)
            zc, rc = probe.Z(), math.hypot(probe.X(), probe.Y())
            if min(abs(zc + 5.0), abs(zc), abs(zc - 5.0), abs(rc - 1.0), abs(rc - 2.0),
                   abs(rc - 3.0), abs(rc - 4.0)) < 1.0e-6:
                continue
            scored_pcon += 1
            want = (1.0 <= rc <= 3.0) if -5.0 <= zc <= 0.0 else (
                (2.0 <= rc <= 4.0) if 0.0 < zc <= 5.0 else False)
            got = bool(placed_pcon_shape.Contains(
                array("d", list(prim.placement_to_local(placed_pcon_placement, p3)))))
            if want != got:
                bad_pcon += 1
        check("the emitted placed polycone answers Contains like the closed form",
              bad_pcon == 0, f"{bad_pcon} disagreement(s) over {scored_pcon} points")
        cc = crosscheck_contains(moved_pcon_record["candidate"], moved_pcon)
        check("the ROOT polycone and the CAD solid agree on Contains",
              cc["disagreements"] == 0,
              f"{cc['disagreements']} disagreement(s) over {cc['points']} points")

        pcon_target = Path("/tmp/csg_selftest_pcon.root")
        write_shape_root(stepped_record["candidate"], pcon_target)
        fpcon = ROOT.TFile.Open(str(pcon_target))
        back_pcon = fpcon.Get("shape")
        sections_ok = (back_pcon is not None and back_pcon.ClassName() == "TGeoPcon"
                       and back_pcon.GetNz() == 4
                       and max(abs(back_pcon.GetZ(i) - step_z[i]) for i in range(4)) < 1e-15
                       and max(abs(back_pcon.GetRmin(i) - step_rmin[i]) for i in range(4)) < 1e-15
                       and max(abs(back_pcon.GetRmax(i) - step_rmax[i]) for i in range(4)) < 1e-15)
        check("shape_<part>.root round-trips a TGeoPcon with all its sections", sections_ok,
              f"read {back_pcon.ClassName() if back_pcon else 'nothing'}, nz "
              f"{back_pcon.GetNz() if back_pcon else 0}")
        fpcon.Close()

        # --- the ROOT half of the prism family ---
        # Each class must come out as itself, with an analytic Capacity().
        for name, solid, want_class, want_capacity in (
                ("Trd1", prism(trd_rings(3, 1, 2, 2, 5)), "TGeoTrd1",
                 4.0 * 2.0 * (3.0 + 1.0) * 5.0),
                ("Trd2", prism(trd_rings(3, 1, 2, 4, 5)), "TGeoTrd2", None),
                ("Arb8", para, "TGeoArb8", None),
                ("Xtru", prism([polygon_ring(ell_poly, -2), polygon_ring(ell_poly, 2)]),
                 "TGeoXtru", 5.0 * 4.0),
                ("Pgon", prism([regular_ring(3, 6, -5), regular_ring(3, 6, 5)]), "TGeoPgon",
                 6.0 * 9.0 * math.tan(math.pi / 6.0) * 10.0)):
            record = process_solid(solid, f"{name}-emission")
            if not record["accepted"]:
                check(f"an axis-aligned {want_class} emits a bare {want_class}", False,
                      f"not accepted: {record['reason']}")
                continue
            shape, placed = prim.build_root(record["candidate"], f"{name}probe")
            ok = shape.ClassName() == want_class and placed is None
            detail = (f"{shape.ClassName()}, capacity {shape.Capacity():.9f}, placement "
                      f"{'present' if placed else 'absent'}")
            if want_capacity is not None:
                rel = abs(shape.Capacity() - want_capacity) / want_capacity
                ok = ok and rel < 1.0e-12
                detail += f", closed form {want_capacity:.9f} (rel {rel:.2e})"
            check(f"an axis-aligned {want_class} emits a bare {want_class} with an analytic "
                  "Capacity()", ok, detail)

        # A placed Trd1, checked through the inverse of the transform that built the OCCT solid.
        trd_shape, trd_placement = prim.build_root(moved_trd_record["candidate"], "movedtrdprobe")
        check("a placed Trd1 is a bare TGeoTrd1 plus a placement",
              trd_shape.ClassName() == "TGeoTrd1" and trd_placement is not None,
              f"{trd_shape.ClassName()}, placement "
              f"{'present' if trd_placement else 'absent'}")
        trd_inverse = prism_place.Inverted()
        bad_trd = 0
        scored_trd = 0
        random.seed(37)
        for _ in range(20000):
            p3 = (random.uniform(-3, 9), random.uniform(-10, 2), random.uniform(-2, 12))
            probe = gp_Pnt(*p3)
            probe.Transform(trd_inverse)
            xc, yc, zc = probe.X(), probe.Y(), probe.Z()
            half = 2.0 - 0.2 * zc                      # dx1 = 3, dx2 = 1, dz = 5
            if min(abs(abs(zc) - 5.0), abs(abs(yc) - 2.0), abs(abs(xc) - half)) < 1.0e-6:
                continue
            scored_trd += 1
            want = abs(zc) <= 5.0 and abs(yc) <= 2.0 and abs(xc) <= half
            got = bool(trd_shape.Contains(
                array("d", list(prim.placement_to_local(trd_placement, p3)))))
            if want != got:
                bad_trd += 1
        check("the emitted placed Trd1 answers Contains like the closed form",
              bad_trd == 0, f"{bad_trd} disagreement(s) over {scored_trd} points")
        cc_prism = crosscheck_contains(moved_trd_record["candidate"], moved_trd)
        check("the ROOT Trd1 and the CAD solid agree on Contains",
              cc_prism["disagreements"] == 0,
              f"{cc_prism['disagreements']} disagreement(s) over {cc_prism['points']} points")

        # The artefact must carry a TGeoXtru's polygon and its sections.
        xtru_record = process_solid(scaled, "xtru-emission")
        xtru_target = Path("/tmp/csg_selftest_xtru.root")
        write_shape_root(xtru_record["candidate"], xtru_target)
        fxtru = ROOT.TFile.Open(str(xtru_target))
        back_xtru = fxtru.Get("shape")
        xtru_ok = (back_xtru is not None and back_xtru.ClassName() == "TGeoXtru"
                   and back_xtru.GetNvert() == 5 and back_xtru.GetNz() == 3
                   and max(abs(back_xtru.GetZ(k) - z) for k, z in enumerate((-3.0, 0.0, 3.0)))
                   < 1e-12
                   and max(abs(back_xtru.GetScale(k) - v)
                           for k, v in enumerate((1.0, 1.4, 0.6))) < 1e-12)
        check("shape_<part>.root round-trips a TGeoXtru with its polygon and its sections",
              xtru_ok, f"read {back_xtru.ClassName() if back_xtru else 'nothing'}, "
                       f"nvert {back_xtru.GetNvert() if back_xtru else 0}, "
                       f"nz {back_xtru.GetNz() if back_xtru else 0}")
        fxtru.Close()

        # --- the ROOT half of the single cell ---
        window_shape, window_placement = prim.build_root(window_record["candidate"],
                                                         "cellwindowprobe")
        node = window_shape.GetBoolNode()
        check("a single cell emits an unplaced TGeoCompositeShape over a TGeoSubtraction node",
              window_shape.ClassName() == "TGeoCompositeShape" and window_placement is None
              and node.ClassName() == "TGeoSubtraction"
              and node.GetLeftShape().ClassName() == "TGeoTube"
              and node.GetRightShape().ClassName() == "TGeoTube",
              f"{window_shape.ClassName()} over {node.ClassName()}"
              f"({node.GetLeftShape().ClassName()}, {node.GetRightShape().ClassName()}), "
              f"placement {'present' if window_placement else 'absent'}")
        steinmetz_shape, _pl = prim.build_root(steinmetz_record["candidate"], "cellsteinprobe")
        check("an intersection cell emits a TGeoIntersection node",
              steinmetz_shape.GetBoolNode().ClassName() == "TGeoIntersection",
              steinmetz_shape.GetBoolNode().ClassName())
        for label, record, solid_of in (("the window", window_record, window),
                                        ("the Steinmetz solid", steinmetz_record, steinmetz),
                                        ("the drilled cube", drilled_record, drilled)):
            cc = crosscheck_contains(record["candidate"], solid_of, n_points=20000)
            check(f"the ROOT cell and the CAD solid agree on Contains for {label}",
                  cc["disagreements"] == 0,
                  f"{cc['disagreements']} disagreement(s) over {cc['points']} points")
            dev = crosscheck_bbox(record["candidate"])
            check(f"the OCCT and ROOT realisations agree on the bounding box for {label}",
                  dev < 1.0e-9, f"max deviation {dev:.3g} cm")
        # 16/3 r^3 is the Steinmetz volume; the composite's Capacity() is a Monte-Carlo estimate.
        want_steinmetz = 16.0 / 3.0
        rel_steinmetz = abs(steinmetz_shape.Capacity() - want_steinmetz) / want_steinmetz
        check("the emitted Steinmetz composite has the closed-form volume, to sampling noise",
              rel_steinmetz < 0.02,
              f"{steinmetz_shape.Capacity():.6f} vs {want_steinmetz:.6f} "
              f"(rel {rel_steinmetz:.2e}, Monte-Carlo)")
        cell_target = Path("/tmp/csg_selftest_cell.root")
        write_shape_root(window_record["candidate"], cell_target)
        fcell = ROOT.TFile.Open(str(cell_target))
        back_cell = fcell.Get("shape")
        check("shape_<part>.root round-trips a single cell as a TGeoCompositeShape",
              back_cell is not None and back_cell.ClassName() == "TGeoCompositeShape"
              and back_cell.GetBoolNode().ClassName() == "TGeoSubtraction",
              f"read {back_cell.ClassName() if back_cell else 'nothing'}")
        fcell.Close()

        # --- the ROOT half of the torus and the elliptic cylinder ---
        # The bounding box is checked against the closed form, since OCCT's torus box is loose.
        for label, record, solid_of, want_class, want_capacity, want_half in (
                ("the solid torus", solid_torus_record, solid_torus, "TGeoTorus",
                 2.0 * math.pi ** 2 * 4.0 * 1.0 ** 2, (5.0, 5.0, 1.0)),
                ("the torus shell", ply_record, ply, "TGeoTorus",
                 2.0 * math.pi ** 2 * 5.0 * (0.30 ** 2 - 0.28 ** 2), (5.3, 5.3, 0.3)),
                ("the elliptic cylinder", eltu_record, eltu_solid, "TGeoEltu",
                 math.pi * 3.0 * 1.5 * 10.0, (3.0, 1.5, 5.0))):
            shape, placement = prim.build_root(record["candidate"], f"probe_{want_class}")
            rel = abs(shape.Capacity() - want_capacity) / want_capacity
            check(f"{label} emits a bare {want_class} with the closed-form Capacity()",
                  shape.ClassName() == want_class and placement is None and rel < 1.0e-12,
                  f"{shape.ClassName()}, capacity {shape.Capacity():.9f} vs "
                  f"{want_capacity:.9f} (rel {rel:.2e}), placement "
                  f"{'present' if placement else 'absent'}")
            cc = crosscheck_contains(record["candidate"], solid_of, n_points=20000)
            check(f"the ROOT {want_class} and the CAD solid agree on Contains for {label}",
                  cc["disagreements"] == 0,
                  f"{cc['disagreements']} disagreement(s) over {cc['points']} points")
            half = (shape.GetDX(), shape.GetDY(), shape.GetDZ())
            worst = max(abs(h - w) for h, w in zip(half, want_half))
            check(f"the emitted {want_class}'s bounding box is the closed form for {label}",
                  worst < 1.0e-12,
                  f"{tuple(round(h, 9) for h in half)} vs {want_half}, worst {worst:.3g} cm")
        # A torus phi wedge, where a mirrored frame convention would go unnoticed by volume.
        wedge_shape, wedge_placement = prim.build_root(wedge_torus_record["candidate"],
                                                       "probe_toruswedge")
        cc = crosscheck_contains(wedge_torus_record["candidate"], wedge_torus, n_points=20000)
        check("the ROOT TGeoTorus and the CAD solid agree on Contains for the hollow wedge",
              wedge_shape.ClassName() == "TGeoTorus" and cc["disagreements"] == 0,
              f"{wedge_shape.ClassName()}, {cc['disagreements']} disagreement(s) over "
              f"{cc['points']} points")
        placed_torus_shape, placed_torus_placement = prim.build_root(
            placed_torus_record["candidate"], "probe_placedtorus")
        cc = crosscheck_contains(placed_torus_record["candidate"], placed_torus, n_points=20000)
        check("a placed torus is a bare TGeoTorus plus a placement that composes correctly",
              placed_torus_shape.ClassName() == "TGeoTorus"
              and placed_torus_placement is not None and cc["disagreements"] == 0,
              f"{placed_torus_shape.ClassName()}, {cc['disagreements']} disagreement(s) over "
              f"{cc['points']} points")
        placed_eltu_shape, placed_eltu_placement = prim.build_root(
            placed_eltu_record["candidate"], "probe_placedeltu")
        cc = crosscheck_contains(placed_eltu_record["candidate"], placed_eltu, n_points=20000)
        check("a placed elliptic cylinder is a bare TGeoEltu plus a placement",
              placed_eltu_shape.ClassName() == "TGeoEltu"
              and placed_eltu_placement is not None and cc["disagreements"] == 0,
              f"{placed_eltu_shape.ClassName()}, {cc['disagreements']} disagreement(s) over "
              f"{cc['points']} points")
        for label, record in (("a TGeoTorus", solid_torus_record),
                              ("a TGeoEltu", eltu_record)):
            target = Path(f"/tmp/csg_selftest_{record['candidate']['leaves'][0]['type']}.root")
            write_shape_root(record["candidate"], target)
            handle = ROOT.TFile.Open(str(target))
            back = handle.Get("shape")
            check(f"shape_<part>.root round-trips {label}",
                  back is not None
                  and back.ClassName() == record["candidate"]["leaves"][0]["type"],
                  f"read {back.ClassName() if back else 'nothing'}")
            handle.Close()

        # --- a sidecar written in Python, loaded in C++, answers Contains as `flat_contains` ---
        ROOT.gInterpreter.AddIncludePath(f"{ROOT.gSystem.Getenv('O2_ROOT')}/include")
        ROOT.gSystem.Load("libO2CADSupport")
        ROOT.gInterpreter.Declare(
            '#include "CADSupport/O2FlatCSG.h"\n'
            'namespace o2 { namespace cad {\n'
            'bool LoadFlatCSG(const std::string& file, O2FlatCSG& solid);\n'
            '} }')
        flat_rt_bad = flat_rt_scored = 0
        flat_rt_failed = []
        for flat_index, result in enumerate(flat_results):
            blocks = result["blocks"]
            xlo, ylo, zlo, xhi, yhi, zhi = recognise._bbox_of(result["solid"])
            # the part bbox is an outer bound of this cell, because the cell IS the part here
            cells = [{"first": 0, "count": len(blocks), "volume": 1.0,
                      "lo": [xlo, ylo, zlo], "hi": [xhi, yhi, zhi]}]
            sidecar = Path(f"/tmp/csg_selftest_flatrt_{flat_index}.bin")
            flatmod.write_sidecar(sidecar, blocks, cells)
            loaded = ROOT.o2.cad.O2FlatCSG(f"probe_flat_{flat_index}")
            if not ROOT.o2.cad.LoadFlatCSG(str(sidecar), loaded):
                flat_rt_failed.append(f"{result['name']}: LoadFlatCSG refused the sidecar")
                continue
            loaded.CloseShape()
            if loaded.GetNhalfspaces() != len(blocks) or loaded.GetNcells() != 1:
                flat_rt_failed.append(f"{result['name']}: loaded "
                                      f"{loaded.GetNhalfspaces()}/{loaded.GetNcells()}")
                continue
            for point, _occ in result["points"]:
                flat_rt_scored += 1
                if bool(loaded.Contains(array("d", list(point)))) != \
                        flatmod.flat_contains(blocks, point):
                    flat_rt_bad += 1
        check("a sidecar written in Python and loaded in C++ answers Contains identically",
              not flat_rt_failed and flat_rt_bad == 0 and flat_rt_scored > 0,
              f"{flat_rt_bad} of {flat_rt_scored} points disagree over {len(flat_results)} "
              f"fixtures" + ("; " + "; ".join(flat_rt_failed) if flat_rt_failed else ""))

        # --- R5: the shipped shape of a multi-cell flat candidate, through its own sidecar ---
        flat_shape, flat_placement = prim.build_root(cone_record, "probe_flat_cells")
        check("a multi-cell flat candidate builds an O2FlatCSG through its own sidecar",
              flat_shape.ClassName() == "o2::cad::O2FlatCSG" and flat_shape.IsClosed()
              and flat_shape.GetNcells() == len(cone_record["cells"])
              and flat_shape.GetNhalfspaces() == cone_record["notes"]["nHalfspaces"]
              and flat_placement is None,
              f"{flat_shape.GetNcells()} cell(s), {flat_shape.GetNhalfspaces()} halfspace(s), "
              f"{flat_shape.GetNboxes()} sub-cell box(es)")
        # the accelerated queries against the twin that defines them, and both against the
        # Python side that wrote the file: three implementations, one answer
        flat_rng = flat_random.Random(5150)
        blo = [min(c["lo"][i] for c in cone_record["cells"]) for i in range(3)]
        bhi = [max(c["hi"][i] for c in cone_record["cells"]) for i in range(3)]
        twin_bad = python_bad = 0
        for _ in range(20000):
            point = [blo[i] + flat_rng.random() * (bhi[i] - blo[i]) for i in range(3)]
            probe = array("d", point)
            accelerated = bool(flat_shape.Contains(probe))
            if accelerated != bool(flat_shape.Contains_Loop(probe)):
                twin_bad += 1
            if accelerated != any(flatmod.flat_contains(c["blocks"], tuple(point))
                                  for c in cone_record["cells"]):
                python_bad += 1
        check("the shipped flat shape agrees with its own _Loop twin and with cadsupport/flat.py",
              twin_bad == 0 and python_bad == 0,
              f"{twin_bad} twin and {python_bad} emitter disagreement(s) over 20000 points")
        # the same twin comparison the converter now runs on every emitted part, through the
        # function that runs it, and the field it reports it in
        cone_cross = crosscheck_contains(cone_record, l_cone_solid)
        plain_cross = crosscheck_contains(disjoint_record["candidate"],
                                          disjoint_record.get("solid", disjoint))
        check("crosscheck_contains measures the twin on a flat part and nothing on a tree part",
              cone_cross["twinDisagreements"] == 0 and cone_cross["disagreements"] == 0
              and cone_cross["points"] > 0 and plain_cross["twinDisagreements"] is None,
              f"flat {cone_cross['twinDisagreements']}/{cone_cross['points']}, tree twin "
              f"{plain_cross['twinDisagreements']}")

        check("the flat shape's Capacity is the sum of its cells' own volumes",
              abs(flat_shape.Capacity()
                  - sum(c["volume"] for c in cone_record["cells"])) <= 1.0e-9,
              f"{flat_shape.Capacity():.9g} vs "
              f"{sum(c['volume'] for c in cone_record['cells']):.9g} cm^3")


    n_ok = sum(1 for _n, ok, _d in checks if ok)
    if verbose:
        print(f"  {n_ok}/{len(checks)} recognise/emit self-checks passed")
    return n_ok, len(checks)

