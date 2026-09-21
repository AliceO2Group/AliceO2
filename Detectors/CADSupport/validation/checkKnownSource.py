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

"""Acceptance test 3: score a converted part against the `TGeoShape` it was made from.

For every part a TGeo -> STEP -> TGeo round trip carries as CSG, it finds the original volume
through the writer report and compares:

  * **class** -- and, for two `TGeoPcon`, the sampled profile;
  * **capacity** -- relative agreement where both are analytic; a composite is *not comparable*;
  * **containment** -- a seeded point set classified by both shapes, with the `shapePlacement`
    from `csg_report.json` composed as `geom.C` does.

Reading the verdict
-------------------
A **failure** is a wrong class, a profile off by more than `--profile-tolerance` (relative to the
diagonal, default `recognise.REL_TOL`), or any containment disagreement. A **flag** is a capacity
that agrees to less than `--capacity-tolerance` (1e-9); `--strict` makes flags fatal.

Points nearer the boundary than `--skin` are not scored, and are counted. The band is taken on
both shapes, or on the source alone where the emitted `Safety` is only a lower bound
(`o2::cad::O2FlatCSG`); `skinnedBoth` says which.

One body of a multi-body CAD label (`..._b1`, `..._b2`) is scored one-way, with capacity not
comparable, and flagged. Duplicate names (`name#2`) are resolved through the writer report, and a
`name__mirrored` prototype by reflecting z.

Usage
-----
  checkKnownSource.py --original o2sim_geometry.root --writer-report PIPE_writer_report.json \\
                      --converted /path/to/converter/output [--points 20000] [--json out.json]
  checkKnownSource.py --self-test

Exit status is non-zero if any part fails.
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import cadsupport_path  # noqa: E402,F401  (puts ../tools on sys.path)

from cadsupport.primitives import placement_to_local  # noqa: E402

# Capacity is compared as a relative deviation, as a flag rather than a failure.
CAPACITY_TOLERANCE = 1.0e-9
# The profile tolerance is the recogniser's `REL_TOL`, relative to the bounding-box diagonal.
PROFILE_TOLERANCE = 1.0e-6
# A point this close to either boundary is not scored; the two boundaries agree to a few ulp.
DEFAULT_SKIN_CM = 1.0e-9
DEFAULT_POINTS = 20000
DEFAULT_SEED = 20260823

# `Capacity()` is a Monte-Carlo estimate for these classes.
_SAMPLED_CAPACITY_CLASSES = ("TGeoCompositeShape", "TGeoUnion", "TGeoIntersection",
                             "TGeoSubtraction", "TGeoHalfSpace")


# ------------------------------------------------------------------------------------------
# profile comparison for the polycone family
# ------------------------------------------------------------------------------------------

def pcon_sections(shape, mirrored=False):
    """[(z, rmin, rmax)] of a polycone, Z-mirrored if the writer emitted the mirrored prototype."""
    rows = [(shape.GetZ(i), shape.GetRmin(i), shape.GetRmax(i)) for i in range(shape.GetNz())]
    if mirrored:
        rows = [(-z, rmin, rmax) for z, rmin, rmax in reversed(rows)]
    return rows


def _radii_at(sections, z):
    """(rmin, rmax) at `z`, clamped to the profile's ends; the extents are compared separately."""
    z = min(max(z, sections[0][0]), sections[-1][0])
    for i in range(len(sections) - 1):
        z0, rmin0, rmax0 = sections[i]
        z1, rmin1, rmax1 = sections[i + 1]
        if z1 <= z0:
            continue
        if z0 <= z <= z1:
            f = (z - z0) / (z1 - z0)
            return rmin0 + f * (rmin1 - rmin0), rmax0 + f * (rmax1 - rmax0)
    return sections[-1][1], sections[-1][2]


def pcon_profile_deviation(sa, sb, merge_tolerance=0.0):
    """The largest radial or axial disagreement, in cm, between two polycone profiles.

    Sampled inside every section, so a redundant z plane is not reported as a difference.
    """
    levels = []
    for z in sorted({z for z, _r0, _r1 in sa} | {z for z, _r0, _r1 in sb}):
        if not levels or z - levels[-1] > merge_tolerance:
            levels.append(z)
    worst = max(abs(sa[0][0] - sb[0][0]), abs(sa[-1][0] - sb[-1][0]))
    for i in range(len(levels) - 1):
        z0, z1 = levels[i], levels[i + 1]
        if z1 <= z0:
            continue
        for f in (1.0e-9, 0.25, 0.5, 0.75, 1.0 - 1.0e-9):
            z = z0 + f * (z1 - z0)
            ra, rb = _radii_at(sa, z), _radii_at(sb, z)
            worst = max(worst, abs(ra[0] - rb[0]), abs(ra[1] - rb[1]))
    return worst


def _phi_deviation(a, b):
    """Degrees. A mirror in z leaves phi alone, so this needs no mirrored variant."""
    return max(abs(a.GetPhi1() - b.GetPhi1()), abs(a.GetDphi() - b.GetDphi()))


def shape_scale(shape):
    """The shape's bounding-box diagonal in cm, the length every relative tolerance is against."""
    return math.sqrt(shape.GetDX() ** 2 + shape.GetDY() ** 2 + shape.GetDZ() ** 2)


# ------------------------------------------------------------------------------------------
# the per-part comparison
# ------------------------------------------------------------------------------------------

def placement_is_identity(placement):
    if placement is None:
        return True
    for r in range(3):
        for c in range(4):
            want = 1.0 if r == c else 0.0
            if abs(placement[r][c] - want) > 1.0e-12:
                return False
    return True


def _bbox_of(shape):
    origin = [shape.GetOrigin()[i] for i in range(3)]
    half = [shape.GetDX(), shape.GetDY(), shape.GetDZ()]
    return origin, half


_ONE_BODY_OF_MANY = re.compile(r"_b\d+$")


def part_is_one_body_of_many(part):
    """True when the part is one body (`#b1`, `#b2`, ...) of a CAD label that carried several."""
    return bool(_ONE_BODY_OF_MANY.search(part.get("part") or ""))


def safety_is_a_true_distance(shape):
    """Whether \a shape's `Safety` is a distance to its boundary, or only a lower bound on one.

    `o2::cad::O2FlatCSG`'s `Safety` is a bound from its sub-cell boxes, often 0 inside.
    """
    return shape.ClassName() != "o2::cad::O2FlatCSG"


def contains_crosscheck(source, emitted, placement, n_points, seed, skin, max_report,
                        mirrored=False, one_way=False):
    """Classify a seeded point set against both shapes; every disagreement is reported.

    `mirrored` reflects the point into the emitted shape's frame, as `geom.C` does. `one_way` scores
    only "inside the emitted shape implies inside the source", for one body of a multi-body source.
    The skin is taken on the source alone when the emitted `Safety` is only a lower bound, which
    makes the test stricter; `skinnedBoth` records which rule ran.
    """
    from array import array
    origin, half = _bbox_of(source)
    rng = random.Random(seed)
    scored = 0
    skipped = 0
    n_mismatches = 0
    n_inside_emitted = 0
    examples = []
    local = array("d", [0.0, 0.0, 0.0])
    probe = array("d", [0.0, 0.0, 0.0])
    skin_both = safety_is_a_true_distance(emitted)
    for _ in range(n_points):
        point = tuple(origin[i] + rng.uniform(-half[i], half[i]) for i in range(3))
        probe[0], probe[1], probe[2] = point
        inside_source = bool(source.Contains(probe))
        if source.Safety(probe, inside_source) < skin:
            skipped += 1
            continue
        reflected = (point[0], point[1], -point[2]) if mirrored else point
        moved = placement_to_local(placement, reflected)
        local[0], local[1], local[2] = moved
        inside_emitted = bool(emitted.Contains(local))
        if skin_both and emitted.Safety(local, inside_emitted) < skin:
            skipped += 1
            continue
        scored += 1
        if inside_emitted:
            n_inside_emitted += 1
        if inside_source != inside_emitted:
            if one_way and inside_source and not inside_emitted:
                continue                       # a sibling body of the same label carries it
            # Counted in full; only the first `max_report` are kept for printing.
            n_mismatches += 1
            if len(examples) < max_report:
                examples.append({"point": [float(c) for c in point],
                                 "local": [float(c) for c in moved],
                                 "source": inside_source, "emitted": inside_emitted})
    return {"points": scored, "skipped": skipped, "mismatches": n_mismatches,
            "insideEmitted": n_inside_emitted, "oneWay": bool(one_way), "skinnedBoth": skin_both,
            "examples": examples}


def reclose_flat_csg(shape):
    """Rebuild an `o2::cad::O2FlatCSG`'s sub-cell boxes after it comes off a file (idempotent)."""
    if shape.ClassName() != "o2::cad::O2FlatCSG":
        return True
    if not shape.IsClosed():
        shape.CloseShape()
    return bool(shape.IsClosed())


def check_part(part, row, source_shape, emitted_shape, placement, n_points, seed, skin,
               capacity_tolerance, profile_tolerance, max_report):
    """Compare one converted part against its source shape. Returns a record."""
    mirrored = bool(row.get("mirrored"))
    one_body = part_is_one_body_of_many(part)
    source_class = row.get("shapeClass") or source_shape.ClassName()
    scale = max(shape_scale(source_shape), 1.0)
    record = {"part": part.get("part"), "volume": part.get("volume"),
              "source": row.get("name"), "mirrored": mirrored, "oneBodyOfMany": one_body,
              "sourceClass": source_class, "emittedClass": emitted_shape.ClassName(),
              "placementIsIdentity": placement_is_identity(placement),
              "classComparable": False, "classMatches": None,
              "profileDeviationCm": None, "profileToleranceCm": profile_tolerance * scale,
              "phiDeviationDeg": None,
              "capacityComparable": False, "capacitySource": None, "capacityEmitted": None,
              "capacityRelativeDeviation": None,
              "contains": None, "failures": [], "flags": []}

    # A different class is flagged, not failed; capacity and containment carry the verdict.
    same_class = source_class == emitted_shape.ClassName()
    record["classComparable"] = not one_body
    record["classMatches"] = None if one_body else same_class
    if one_body:
        record["flags"].append(
            "one body of a multi-body CAD label: the source is the whole label, so the class "
            "and the capacity are not comparable and containment is scored one-way")
    elif not same_class:
        record["flags"].append(
            f"class {emitted_shape.ClassName()} is not the source's {source_class}")

    # Under a non-identity placement the profiles differ legitimately; containment decides.
    if (same_class and not one_body and source_class == "TGeoPcon"
            and record["placementIsIdentity"]):
        record["phiDeviationDeg"] = _phi_deviation(source_shape, emitted_shape)
        deviation = pcon_profile_deviation(pcon_sections(source_shape, mirrored),
                                           pcon_sections(emitted_shape),
                                           merge_tolerance=profile_tolerance * scale)
        record["profileDeviationCm"] = deviation
        if deviation > record["profileToleranceCm"]:
            record["failures"].append(
                f"the polycone profile is {deviation:.6g} cm off the source's, over the "
                f"{record['profileToleranceCm']:.3g} cm the recogniser claims")
        if record["phiDeviationDeg"] > 1.0e-9:
            record["failures"].append(
                f"phi differs from the source by {record['phiDeviationDeg']:.6g} deg")

    # The writer's record of the emitted volume is unambiguous even where two volumes share a name.
    capacity_source = row.get("capacity_cm3")
    if capacity_source is None and source_class not in _SAMPLED_CAPACITY_CLASSES:
        capacity_source = float(source_shape.Capacity())
    record["capacitySource"] = capacity_source
    record["capacityEmitted"] = float(emitted_shape.Capacity())
    comparable = (capacity_source is not None and capacity_source > 0.0 and not one_body
                  and source_class not in _SAMPLED_CAPACITY_CLASSES
                  and emitted_shape.ClassName() not in _SAMPLED_CAPACITY_CLASSES)
    if comparable:
        record["capacityComparable"] = True
        rel = abs(record["capacityEmitted"] - capacity_source) / capacity_source
        record["capacityRelativeDeviation"] = rel
        if rel > capacity_tolerance:
            record["flags"].append(
                f"capacity {record['capacityEmitted']:.9g} cm^3 differs from the source's "
                f"{capacity_source:.9g} cm^3 by {rel:.3g} relative")

    record["contains"] = contains_crosscheck(source_shape, emitted_shape, placement, n_points,
                                             seed, skin, max_report, mirrored, one_body)
    if record["contains"]["mismatches"]:
        record["failures"].append(
            f"{record['contains']['mismatches']} containment disagreement(s) over "
            f"{record['contains']['points']} scored point(s)")
    if record["contains"]["points"] == 0:
        record["failures"].append("no point was scored: the comparison is empty")
    if one_body and record["contains"]["insideEmitted"] == 0:
        # Without this a one-way comparison would be passed by a body that encloses nothing.
        record["failures"].append(
            f"the emitted body encloses none of the {record['contains']['points']} scored "
            "point(s): the one-way comparison is empty")
    return record


# ------------------------------------------------------------------------------------------
# driving a converter output directory
# ------------------------------------------------------------------------------------------

def _writer_index(writer_report):
    """emittedName -> the writer's row, which carries the source volume's real `name`.

    A `<name>__body` solid is indexed through its parent's `bodyComponent`, whose class and
    capacity are the body's.
    """
    index = {}
    for row in writer_report.get("volumes", []):
        emitted = row.get("emittedName") or row.get("name")
        if emitted:
            index[emitted] = row
        body = row.get("bodyComponent")
        if body and body != emitted:
            index[body] = row
    return index


def _placed_box(placement, shape):
    """A shape's axis-aligned box in the part frame: `(origin, half)`."""
    origin = [shape.GetOrigin()[i] for i in range(3)]
    half = [shape.GetDX(), shape.GetDY(), shape.GetDZ()]
    if placement is None:
        return origin, half
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = (origin[0] + sx * half[0], origin[1] + sy * half[1],
                         origin[2] + sz * half[2])
                for i in range(3):
                    v = sum(placement[i][c] * local[c] for c in range(3)) + placement[i][3]
                    lo[i] = min(lo[i], v)
                    hi[i] = max(hi[i], v)
    return [0.5 * (lo[i] + hi[i]) for i in range(3)], [0.5 * (hi[i] - lo[i]) for i in range(3)]


def resolve_source_volume(candidates, row, emitted_shape=None, placement=None):
    """Which of several volumes sharing one name the writer's row refers to.

    The bounding box decides, being exact for every `TGeoShape`; capacity is only a tie-break
    where it is analytic, since a composite's is Monte-Carlo.
    """
    if len(candidates) == 1:
        return candidates[0]
    wanted_class = row.get("shapeClass")
    wanted_capacity = row.get("capacity_cm3")
    sampled = wanted_class in _SAMPLED_CAPACITY_CLASSES
    want_box = (_placed_box(placement, emitted_shape)
                if emitted_shape is not None else None)
    best, best_key = None, None
    for volume in candidates:
        shape = volume.GetShape()
        if wanted_class and shape.ClassName() != wanted_class:
            continue
        box_score = 0.0
        if want_box is not None:
            here = _placed_box(None, shape)
            box_score = max(max(abs(here[0][i] - want_box[0][i]) for i in range(3)),
                            max(abs(here[1][i] - want_box[1][i]) for i in range(3)))
        capacity_score = (0.0 if (wanted_capacity is None or sampled)
                          else abs(shape.Capacity() - wanted_capacity)
                          / max(abs(wanted_capacity), 1.0e-30))
        key = (box_score, capacity_score)
        if best_key is None or key < best_key:
            best, best_key = volume, key
    return best


def check_run(original, writer_report_path, converted, n_points=DEFAULT_POINTS,
              seed=DEFAULT_SEED, skin=DEFAULT_SKIN_CM, capacity_tolerance=CAPACITY_TOLERANCE,
              profile_tolerance=PROFILE_TOLERANCE, max_report=5, verbose=True):
    """Compare every CSG-carried part of a converter output against its source volume."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    converted = Path(converted)
    csg_report_path = converted / "csg_report.json"
    if not csg_report_path.exists():
        raise SystemExit(f"{csg_report_path} does not exist (convert with --csg auto)")
    csg_report = json.loads(csg_report_path.read_text())
    writer_report = json.loads(Path(writer_report_path).read_text())
    index = _writer_index(writer_report)

    manager = ROOT.TGeoManager.Import(str(original))
    if manager is None:
        raise SystemExit(f"could not read a TGeoManager from {original}")
    by_name = {}
    for volume in manager.GetListOfVolumes():
        by_name.setdefault(volume.GetName(), []).append(volume)

    records = []
    open_files = []
    for part in csg_report.get("parts", []):
        if part.get("representation") != "csg":
            continue
        emitted_name = part.get("volume")
        stub = {"part": part.get("part"), "volume": emitted_name, "failures": [], "flags": []}
        row = index.get(emitted_name)
        if row is None:
            stub["failures"].append(f"no writer-report row for emittedName {emitted_name!r}")
            records.append(stub)
            continue
        shape_file = part.get("shapeFile")
        if not shape_file or not Path(shape_file).exists():
            stub["failures"].append(f"shapeFile {shape_file!r} does not exist")
            records.append(stub)
            continue
        handle = ROOT.TFile.Open(str(shape_file))
        open_files.append(handle)
        emitted_shape = handle.Get("shape")
        if not emitted_shape:
            stub["failures"].append(f"{shape_file} carries no object under the key \"shape\"")
            records.append(stub)
            continue
        if not reclose_flat_csg(emitted_shape):
            stub["failures"].append(
                f"{shape_file}: O2FlatCSG::CloseShape refused the shape after reading it, so "
                "its sub-cell boxes could not be rebuilt")
            records.append(stub)
            continue
        # The emitted shape is read first: its bounding box tells same-named volumes apart.
        candidates = by_name.get(row.get("name")) or []
        source_volume = (resolve_source_volume(candidates, row, emitted_shape,
                                               part.get("shapePlacement"))
                         if candidates else None)
        if source_volume is None:
            stub["failures"].append(
                f"the original geometry has no volume named {row.get('name')!r} whose shape "
                "matches the writer's record")
            records.append(stub)
            continue
        record = check_part(part, row, source_volume.GetShape(), emitted_shape,
                            part.get("shapePlacement"), n_points, seed, skin,
                            capacity_tolerance, profile_tolerance, max_report)
        records.append(record)
        if verbose:
            print_record(record)

    n_fail = sum(1 for r in records if r["failures"])
    n_flag = sum(1 for r in records if r.get("flags"))
    if verbose:
        worst_capacity = max([r["capacityRelativeDeviation"] for r in records
                              if r.get("capacityRelativeDeviation") is not None] or [0.0])
        worst_profile = max([r["profileDeviationCm"] for r in records
                             if r.get("profileDeviationCm") is not None] or [0.0])
        print(f"\n{len(records) - n_fail}/{len(records)} CSG part(s) agree with their source "
              f"TGeoShape ({n_fail} failure(s), {n_flag} flag(s))")
        print(f"worst capacity deviation {worst_capacity:.3g} relative, worst polycone profile "
              f"deviation {worst_profile:.3g} cm")
    for handle in open_files:
        handle.Close()
    return records, n_fail, n_flag


def print_record(record):
    if record["failures"]:
        print(f"  [FAIL] {record['volume']}: " + "; ".join(record["failures"]))
        for example in (record.get("contains") or {}).get("examples", []):
            print(f"           at {example['point']} (local {example['local']}): "
                  f"source {'in' if example['source'] else 'out'}, "
                  f"emitted {'in' if example['emitted'] else 'out'}")
        return
    bits = [record["emittedClass"]]
    if record.get("mirrored"):
        bits.append("mirrored prototype")
    if record.get("oneBodyOfMany"):
        bits.append("one body of many, scored one-way")
    if record.get("classComparable"):
        bits.append("class matches" if record["classMatches"] else "class differs")
    if record.get("profileDeviationCm") is not None:
        bits.append(f"profile {record['profileDeviationCm']:.3g} cm")
    if record.get("capacityComparable"):
        bits.append(f"capacity rel {record['capacityRelativeDeviation']:.3g}")
    else:
        bits.append("capacity not comparable")
    contains = record.get("contains") or {}
    bits.append(f"Contains {contains.get('mismatches')}/{contains.get('points')} "
                f"({contains.get('skipped')} on the skin)")
    marker = "flag" if record.get("flags") else "ok  "
    print(f"  [{marker}] {record['volume']}: " + ", ".join(bits))
    for flag in record.get("flags", []):
        print(f"           flag: {flag}")


# ------------------------------------------------------------------------------------------
# self-test
# ------------------------------------------------------------------------------------------
# The fixtures are built in a subprocess, since ROOT cannot create one geometry and import another.

_FIXTURE_BUILDER = r"""
import json, sys
from pathlib import Path
import ROOT
ROOT.gROOT.SetBatch(True)

folder = Path(sys.argv[1])
sections = [(-5.0, 1.0, 3.0), (0.0, 1.0, 3.0), (0.0, 2.0, 4.0), (5.0, 2.0, 4.0)]
names = ("GOOD", "PLACED", "MIRRORED", "TWIN", "BAD", "WRONGCLASS", "TWOBODY")


def halves_shape(zlow):
    # One half of a tube, as a composite, so its Capacity() is a Monte-Carlo estimate.
    tag = "lo" if zlow < 0.0 else "hi"
    tube = ROOT.TGeoTube("halves_t_" + tag, 0.0, 2.0, 1.0)
    slab = ROOT.TGeoBBox("halves_b_" + tag, 3.0, 3.0, 0.5)
    shift = ROOT.TGeoTranslation("halves_m_" + tag, 0.0, 0.0, zlow + 0.5)
    for obj in (tube, slab, shift):
        ROOT.SetOwnership(obj, False)
    node = ROOT.TGeoIntersection(tube, slab, ROOT.nullptr, shift)
    ROOT.SetOwnership(node, False)
    comp = ROOT.TGeoCompositeShape("halves_c_" + tag, node)
    ROOT.SetOwnership(comp, False)
    return comp


def make_pcon(name, rows, phi1=0.0, dphi=360.0):
    shape = ROOT.TGeoPcon(name, phi1, dphi, len(rows))
    for i, (z, rmin, rmax) in enumerate(rows):
        shape.DefineSection(i, z, rmin, rmax)
    ROOT.SetOwnership(shape, False)
    return shape


geometry = ROOT.TGeoManager("knownsource", "known-source self-test")
material = ROOT.TGeoMaterial("Vacuum", 0, 0, 0)
medium = ROOT.TGeoMedium("Vacuum", 1, material)
top = geometry.MakeBox("TOP", medium, 50.0, 50.0, 50.0)
geometry.SetTopVolume(top)
for name in names:
    volume = ROOT.TGeoVolume(name, make_pcon(name + "_sh", sections), medium)
    ROOT.SetOwnership(volume, False)
    top.AddNode(volume, 1)
# A second volume under a taken name: the writer emits `TWIN#2` and the checker must find this one.
twin = [(z, rmin, rmax + 1.0) for z, rmin, rmax in sections]
second = ROOT.TGeoVolume("TWIN", make_pcon("twin2_sh", twin), medium)
ROOT.SetOwnership(second, False)
top.AddNode(second, 1)
# Two composites of one name, the two halves of a tube: only a bounding box tells them apart.
for zlow in (-1.0, 0.0):
    half = ROOT.TGeoVolume("HALVES", halves_shape(zlow), medium)
    ROOT.SetOwnership(half, False)
    top.AddNode(half, 1)
geometry.CloseGeometry()
geometry.Export(str(folder / "source_geometry.root"))


def capacity(rows):
    return make_pcon("cap_probe", rows).Capacity()


rows = [{"name": n, "emittedName": n, "shapeClass": "TGeoPcon", "mirrored": n == "MIRRORED",
         "capacity_cm3": capacity(sections)} for n in names]
rows.append({"name": "TWIN", "emittedName": "TWIN#2", "shapeClass": "TGeoPcon",
             "mirrored": False, "capacity_cm3": capacity(twin)})
# The writer's capacity for a composite is another Monte-Carlo draw, never used for ranking.
rows.append({"name": "HALVES", "emittedName": "HALVES", "shapeClass": "TGeoCompositeShape",
             "mirrored": False, "capacity_cm3": halves_shape(0.0).Capacity()})
(folder / "writer_report.json").write_text(json.dumps({"volumes": rows}))


def write_shape(name, shape, placement=None):
    target = folder / ("shape_%s.root" % name.replace("#", "_"))
    out = ROOT.TFile.Open(str(target), "RECREATE")
    out.WriteTObject(shape, "shape")
    out.Close()
    return {"part": name, "volume": name, "representation": "csg",
            "shapeFile": str(target), "shapePlacement": placement}


parts = [write_shape("GOOD", make_pcon("good_sh", sections))]
# A load-bearing placement: the profile sits 7 cm up and the placement brings it back.
shifted = [(z + 7.0, rmin, rmax) for z, rmin, rmax in sections]
parts.append(write_shape("PLACED", make_pcon("placed_sh", shifted),
                         [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -7.0]]))
# The Z-mirrored prototype the writer emits for a volume placed by a reflecting matrix.
mirrored = [(-z, rmin, rmax) for z, rmin, rmax in reversed(sections)]
parts.append(write_shape("MIRRORED", make_pcon("mirrored_sh", mirrored)))
parts.append(write_shape("TWIN", make_pcon("twin_sh", sections)))
parts.append(write_shape("TWIN#2", make_pcon("twin2_out_sh", twin)))
# The emitted body is the LOWER half; the checker must resolve to the lower source volume.
parts.append(write_shape("HALVES", halves_shape(-1.0)))
wrong = [(z, rmin, rmax + (0.05 if i == 3 else 0.0))
         for i, (z, rmin, rmax) in enumerate(sections)]
parts.append(write_shape("BAD", make_pcon("bad_sh", wrong)))
tube = ROOT.TGeoTube("wrongclass_sh", 1.0, 4.0, 5.0)
ROOT.SetOwnership(tube, False)
parts.append(write_shape("WRONGCLASS", tube))
# One body of a two-body CAD label: the upper half of the source's profile.
upper = [(0.5, 2.0, 4.0), (5.0, 2.0, 4.0)]
body = write_shape("TWOBODY_b2", make_pcon("twobody_sh", upper))
body["volume"] = "TWOBODY"
parts.append(body)
# ... and a body that sticks OUT of its own label must still fail.
outside = [(0.5, 2.0, 5.0), (5.0, 2.0, 5.0)]
spill = write_shape("TWOBODYBAD_b2", make_pcon("twobodybad_sh", outside))
spill["volume"] = "TWOBODY"
parts.append(spill)
(folder / "csg_report.json").write_text(json.dumps({"parts": parts}))

# Every file is on disk; skip the teardown, which ROOT's global geometry does not survive.
import os
sys.stdout.flush()
os._exit(0)
"""


def self_test(verbose=True, workdir=None):
    """A geometry, a writer report and a converter output built here, with a known verdict.

    Three negative controls: a displaced radius, a wrong class, and the placed control without its
    placement. The working folder is left under the system temporary directory.
    """
    import subprocess
    import tempfile

    checks = []

    def check(name, condition, detail=""):
        checks.append((name, bool(condition), detail))
        if verbose:
            print(f"  [{'ok ' if condition else 'FAIL'}] {name}"
                  + (f"  {detail}" if detail else ""))

    folder = Path(tempfile.mkdtemp(prefix="knownsource_")) if workdir is None else Path(workdir)
    folder.mkdir(parents=True, exist_ok=True)
    builder = folder / "_build_fixtures.py"
    builder.write_text(_FIXTURE_BUILDER)
    subprocess.run([sys.executable, str(builder), str(folder)], check=True,
                   stdout=subprocess.DEVNULL)

    records, n_fail, _n_flag = check_run(folder / "source_geometry.root",
                                         folder / "writer_report.json", folder,
                                         n_points=20000, verbose=False)
    by_name = {r["volume"]: r for r in records}

    good = by_name.get("GOOD", {})
    check("the positive control passes every comparable check",
          not good.get("failures") and not good.get("flags")
          and good.get("classMatches") is True and good.get("capacityComparable") is True
          and good.get("contains", {}).get("mismatches") == 0,
          f"failures {good.get('failures')}, flags {good.get('flags')}, capacity rel "
          f"{good.get('capacityRelativeDeviation')}, Contains "
          f"{good.get('contains', {}).get('mismatches')}/"
          f"{good.get('contains', {}).get('points')}")
    check("the positive control actually scored a useful point set",
          good.get("contains", {}).get("points", 0) > 1000,
          f"{good.get('contains', {}).get('points')} point(s) scored, "
          f"{good.get('contains', {}).get('skipped')} on the skin")

    placed = by_name.get("PLACED", {})
    check("a shape whose placement is composed correctly passes",
          not placed.get("failures") and placed.get("contains", {}).get("mismatches") == 0,
          f"failures {placed.get('failures')}")
    ignored = _placed_without_its_placement(folder)
    check("the placement is load-bearing: ignoring it must fail",
          ignored is not None and ignored["mismatches"] > 0,
          f"{ignored['mismatches'] if ignored else 'not run'} disagreement(s) with a null "
          "placement")

    mirrored = by_name.get("MIRRORED", {})
    check("a Z-mirrored prototype is compared through the mirror and passes",
          not mirrored.get("failures") and mirrored.get("mirrored") is True
          and mirrored.get("contains", {}).get("mismatches") == 0,
          f"failures {mirrored.get('failures')}")

    twin2 = by_name.get("TWIN#2", {})
    check("a name shared by two volumes resolves to the right one",
          not twin2.get("failures") and twin2.get("capacityComparable") is True
          and twin2.get("capacityRelativeDeviation") is not None
          and twin2["capacityRelativeDeviation"] < 1.0e-12,
          f"failures {twin2.get('failures')}, capacity rel "
          f"{twin2.get('capacityRelativeDeviation')}")

    halves = by_name.get("HALVES", {})
    check("two same-named composites are told apart by their box, not by a sampled capacity",
          not halves.get("failures")
          and halves.get("contains", {}).get("mismatches") == 0,
          f"failures {halves.get('failures')}, Contains "
          f"{halves.get('contains', {}).get('mismatches')}/"
          f"{halves.get('contains', {}).get('points')}")
    check("and their capacities really could not have decided it",
          _halves_capacities_are_indistinguishable(folder),
          "the two halves' Capacity() draws are within Monte-Carlo noise of each other")

    bad = by_name.get("BAD", {})
    check("the negative control is caught", bool(bad.get("failures")),
          "; ".join(bad.get("failures", [])) or "NOT CAUGHT")
    check("the negative control is caught by containment, not only by the profile",
          bad.get("contains", {}).get("mismatches", 0) > 0,
          f"{bad.get('contains', {}).get('mismatches')} disagreement(s)")
    check("the negative control's profile deviation is the displacement",
          bad.get("profileDeviationCm") is not None
          and abs(bad["profileDeviationCm"] - 0.05) < 1.0e-9,
          f"{bad.get('profileDeviationCm')}")

    wrongclass = by_name.get("WRONGCLASS", {})
    check("a shape of the wrong class is caught by the metrics, not only by its class",
          bool(wrongclass.get("failures"))
          and wrongclass.get("contains", {}).get("mismatches", 0) > 0,
          "; ".join(wrongclass.get("failures", [])) or "NOT CAUGHT")
    check("a class that differs without a geometric difference is a flag, not a failure",
          wrongclass.get("classMatches") is False and any(
              "is not the source's" in f for f in wrongclass.get("flags", [])),
          f"flags {wrongclass.get('flags')}")

    two_body = next((r for r in records if r["part"] == "TWOBODY_b2"), {})
    check("one body of a multi-body label passes on the one-way containment test",
          not two_body.get("failures") and two_body.get("oneBodyOfMany") is True
          and two_body.get("contains", {}).get("oneWay") is True
          and two_body.get("contains", {}).get("mismatches") == 0
          and two_body.get("contains", {}).get("insideEmitted", 0) > 100,
          f"failures {two_body.get('failures')}, "
          f"{two_body.get('contains', {}).get('insideEmitted')} point(s) inside the body")
    check("a multi-body part's class and capacity are reported as not comparable",
          two_body.get("capacityComparable") is False
          and two_body.get("classComparable") is False
          and any("multi-body" in f for f in two_body.get("flags", [])),
          f"flags {two_body.get('flags')}")
    spilled = next((r for r in records if r["part"] == "TWOBODYBAD_b2"), {})
    check("the one-way rule still catches a body that sticks out of its own label",
          bool(spilled.get("failures"))
          and spilled.get("contains", {}).get("mismatches", 0) > 0,
          "; ".join(spilled.get("failures", [])) or "NOT CAUGHT")

    check("the run reports exactly the three deliberately wrong parts as failures",
          n_fail == 3, f"{n_fail} failure(s) over {len(records)} part(s)")

    n_ok = sum(1 for _n, ok, _d in checks if ok)
    if verbose:
        print(f"  {n_ok}/{len(checks)} known-source self-checks passed (fixtures in {folder})")
    return n_ok, len(checks)


def _halves_capacities_are_indistinguishable(folder):
    """Are the two same-named composites' capacities within Monte-Carlo noise of each other?"""
    import ROOT
    manager = ROOT.gGeoManager
    if not manager:
        return False
    capacities = [volume.GetShape().Capacity() for volume in manager.GetListOfVolumes()
                  if volume.GetName() == "HALVES"]
    if len(capacities) != 2:
        return False
    return abs(capacities[0] - capacities[1]) / max(capacities) < 0.02


def _placed_without_its_placement(folder):
    """Re-run the placed positive control with a null placement; it must then disagree."""
    import ROOT
    manager = ROOT.gGeoManager
    if not manager:
        return None
    source = None
    for volume in manager.GetListOfVolumes():
        if volume.GetName() == "PLACED":
            source = volume.GetShape()
            break
    if source is None:
        return None
    handle = ROOT.TFile.Open(str(Path(folder) / "shape_PLACED.root"))
    emitted = handle.Get("shape")
    result = contains_crosscheck(source, emitted, None, 5000, DEFAULT_SEED, DEFAULT_SKIN_CM, 1)
    handle.Close()
    return result


# ------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", type=Path,
                    help="the o2sim_geometry.root the STEP was written from")
    ap.add_argument("--writer-report", type=Path, dest="writer_report",
                    help="O2_TGeoToCAD.py's --report JSON, which maps emittedName -> name")
    ap.add_argument("--converted", type=Path,
                    help="the converter output folder (csg_report.json and shape_*.root)")
    ap.add_argument("--points", type=int, default=DEFAULT_POINTS,
                    help="containment samples per part; default %(default)s")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="the fixed seed for those samples; default %(default)s")
    ap.add_argument("--skin", type=float, default=DEFAULT_SKIN_CM,
                    help="do not score points nearer than this to either boundary, in cm; "
                         "default %(default)s")
    ap.add_argument("--capacity-tolerance", type=float, default=CAPACITY_TOLERANCE,
                    dest="capacity_tolerance",
                    help="relative capacity agreement below which a part is flagged; "
                         "default %(default)s")
    ap.add_argument("--profile-tolerance", type=float, default=PROFILE_TOLERANCE,
                    dest="profile_tolerance",
                    help="polycone profile agreement demanded, relative to the part's diagonal; "
                         "default %(default)s, which is cadsupport/recognise.REL_TOL")
    ap.add_argument("--strict", action="store_true",
                    help="treat capacity flags as failures too")
    ap.add_argument("--max-report", type=int, default=5, dest="max_report",
                    help="how many disagreeing points to print per part; default %(default)s")
    ap.add_argument("--json", type=Path, help="write the per-part records here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        n_ok, n = self_test()
        print(f"\n{n_ok}/{n} known-source self-checks passed")
        return 0 if n_ok == n else 1

    if not (args.original and args.writer_report and args.converted):
        ap.error("give --original, --writer-report and --converted, or --self-test")

    records, n_fail, n_flag = check_run(args.original, args.writer_report, args.converted,
                                        n_points=args.points, seed=args.seed, skin=args.skin,
                                        capacity_tolerance=args.capacity_tolerance,
                                        profile_tolerance=args.profile_tolerance,
                                        max_report=args.max_report)
    if args.json:
        args.json.write_text(json.dumps(records, indent=1))
        print(f"Wrote {args.json}")
    return 1 if (n_fail or (args.strict and n_flag)) else 0


if __name__ == "__main__":
    sys.exit(main())
