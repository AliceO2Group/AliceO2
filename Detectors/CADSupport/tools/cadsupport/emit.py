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

"""Recognise CAD leaf solids as CSG, prove it, and emit `shape_<VOL>_<LID>.root`.

A part converts only if the OCCT symmetric difference and the oracle gate both accept it.
`--db` walks the `brep_*.brep` files of a gate run, `O2_CADtoTGeo.py --csg` reaches the same code
through `cadsupport.hook`, and `--from-json` completes descriptions written without PyROOT.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from cadsupport.occ_env import ensure_occ  # noqa: E402

ensure_occ()

from cadsupport import accept, primitives as prim, recognise  # noqa: E402


# ------------------------------------------------------------------------------------------
# per-solid pipeline
# ------------------------------------------------------------------------------------------

# The shape-tolerance helper lives in `cadsupport/accept.py`, because `cadsupport/recognise.py` needs it too
# and must not import this module. Re-exported here under its long-standing name.
model_tolerance_cm = accept.model_tolerance_cm


# Retried after the acceptance test rejects a candidate, in order of increasing generality.
_RETRIES = (("a revolved profile", recognise.recognise_revolved),
            ("a single cell", recognise.recognise_single_cell),
            ("a union of cells", recognise.recognise_union_of_cells),
            # Last, after the union of cells, as in the cascade.
            ("flat cells", recognise.recognise_flat_cells))


def _build_and_accept(solid, cand, tol, band_factor, cache):
    """`(acceptance|None, reason|None)` for one candidate. Never raises on a bad candidate."""
    occ_shape = recognise.realised_for(cache, cand)
    if occ_shape is None:
        try:
            occ_shape = prim.build_occ(cand)
        except Exception as exc:                                 # noqa: BLE001
            return None, f"candidate failed to build in OCCT: {exc}"
        recognise.remember_realised(cache, cand, occ_shape)
    if "props" not in cache:
        cache["props"] = accept._props(solid)
    result = accept.symmetric_difference(solid, occ_shape, tol, band_factor,
                                         original_props=cache["props"])
    return result, (None if result.get("accepted") else result.get("reason"))


def process_solid(solid, name, tolerance=None, band_factor=1.0, cache=None):
    """recognise -> build -> accept. Returns a record; `record['candidate']` is None if declined.

    A rejected candidate is retried with each of `_RETRIES`. `cache` is the per-solid memo they
    share; `recognise.realised_for` reads the accepted candidate's OCCT shape back from it.
    The memo is keyed by nothing but the solid, so it assumes the solid is not mutated while it lives.
    """
    cache = {} if cache is None else cache
    record = {"part": name, "recognised": False, "accepted": False, "candidate": None,
              "reason": None, "acceptance": None, "recogniser": None, "description": None}
    cand, reason = recognise.recognise(solid, cache=cache)
    if cand is None:
        record["reason"] = reason
        return record
    record["recognised"] = True
    record["recogniser"] = cand["recogniser"]
    record["description"] = prim.describe(cand)
    tol = model_tolerance_cm(solid) if tolerance is None else tolerance
    result, why_not = _build_and_accept(solid, cand, tol, band_factor, cache)
    if result is not None:
        record["acceptance"] = result
        record["accepted"] = bool(result.get("accepted"))
    if record["accepted"]:
        record["candidate"] = cand
        return record
    record["reason"] = why_not

    notes = []
    for label, propose in _RETRIES:
        alternative, alt_declined = propose(solid, cache=cache)
        if alternative is None:
            notes.append(f"as {label}: {alt_declined}")
            continue
        if alternative["recogniser"] == cand["recogniser"]:
            # This matcher is what produced the candidate that was just refused; retrying it
            # would refuse it again.
            notes.append(f"as {label}: the same proposal that was just rejected")
            continue
        alt_result, alt_why_not = _build_and_accept(solid, alternative, tol, band_factor, cache)
        if alt_result is not None and alt_result.get("accepted"):
            record["retriedAfter"] = {"recogniser": cand["recogniser"],
                                      "description": record["description"], "reason": why_not}
            record["recogniser"] = alternative["recogniser"]
            record["description"] = prim.describe(alternative)
            record["acceptance"] = alt_result
            record["accepted"] = True
            record["candidate"] = alternative
            record["reason"] = None
            return record
        notes.append(f"retried as {alternative['recogniser']} "
                     f"({prim.describe(alternative)}): {alt_why_not}")
    record["reason"] = "; ".join([why_not] + notes)
    return record


def write_shape_root(cand, path):
    """Write the description as `shape_<part>.root`, per the convention in O2SolidHarness.h.

    The shape is under `shape`, in cm, with an optional `placement` TGeoHMatrix from its own frame
    to the part frame; no `placement` means the identity.
    """
    return write_shape_object(*prim.build_root(cand, "shape"), path)


def write_shape_object(shape, placement, path):
    """`write_shape_root`'s second half, for a caller that already built and checked the shape."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    out = ROOT.TFile.Open(str(path), "RECREATE")
    out.WriteTObject(shape, "shape")
    matrix = prim.root_placement_matrix(placement, "placement")
    if matrix is not None:
        out.WriteTObject(matrix, "placement")
    out.Close()
    return shape


# Points twin_parity draws by default: max(floor, per-cell * cells), so each cell gets enough.
_TWIN_PARITY_FLOOR = 4000
_TWIN_PARITY_PER_CELL = 500


def twin_parity(shape, n_points=None, seed=7771, grow=1.0):
    """`Contains` against `Contains_Loop` on a shape that has twins. `None` when it has none.

    It samples the union of the declared cell boxes grown by `grow` about its centre, where a cell
    reaching past its box shows up; `n_points` defaults to `max(4000, 500 * cells)`.
    """
    if not (hasattr(shape, "Contains_Loop") and hasattr(shape, "GetCellBBox")):
        return None
    if n_points is None:
        n_points = max(_TWIN_PARITY_FLOOR, _TWIN_PARITY_PER_CELL * shape.GetNcells())
    import random
    from array import array
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    cell_lo, cell_hi = array("d", [0.0] * 3), array("d", [0.0] * 3)
    for cell in range(shape.GetNcells()):
        shape.GetCellBBox(cell, cell_lo, cell_hi)
        for axis in range(3):
            lo[axis] = min(lo[axis], cell_lo[axis])
            hi[axis] = max(hi[axis], cell_hi[axis])
    if not all(math.isfinite(lo[i]) and math.isfinite(hi[i]) for i in range(3)):
        return {"points": 0, "disagreements": 0, "insideAccelerated": 0, "growFactor": grow}
    centre = [0.5 * (lo[i] + hi[i]) for i in range(3)]
    half = [0.5 * (hi[i] - lo[i]) * (1.0 + grow) for i in range(3)]
    rng = random.Random(seed)
    probe = array("d", [0.0, 0.0, 0.0])
    disagreements = inside = 0
    for _ in range(n_points):
        for axis in range(3):
            probe[axis] = centre[axis] - half[axis] + rng.random() * 2.0 * half[axis]
        accelerated = bool(shape.Contains(probe))
        inside += int(accelerated)
        if accelerated != bool(shape.Contains_Loop(probe)):
            disagreements += 1
    return {"points": n_points, "disagreements": disagreements, "insideAccelerated": inside,
            "growFactor": grow}


def twin_decline_reason(parity):
    """The one wording both emission paths use when the twin-parity gate refuses a part."""
    return (f"the emitted shape disagrees with its own _Loop twin about "
            f"{parity['disagreements']} of {parity['points']} classified point(s): a cell "
            "reaches past the bounding box declared for it, so the accelerated queries and the "
            "reference ones are not describing the same solid")


def _occ_bbox(shape):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    # OCCT's box carries the shape's tolerance gap; ROOT's is tight.
    box.SetGap(0.0)
    return box.Get()


def crosscheck_bbox(cand, occ_shape=None, built=None):
    """Max deviation, in cm, between the ROOT realisation's bounding box and the OCCT one.

    Exact for an unplaced primitive; for a placed or boolean shape ROOT's box is a hull, so
    `crosscheck_contains` is the sharp check.
    """
    occ_shape = occ_shape if occ_shape is not None else prim.build_occ(cand)
    xmin, ymin, zmin, xmax, ymax, zmax = _occ_bbox(occ_shape)
    shape, placement = built if built is not None else prim.build_root(cand, "bboxprobe")
    origin = [shape.GetOrigin()[i] for i in range(3)]
    half = [shape.GetDX(), shape.GetDY(), shape.GetDZ()]
    lo_root = [origin[i] - half[i] for i in range(3)]
    hi_root = [origin[i] + half[i] for i in range(3)]
    if placement is not None:
        lo_root, hi_root = _placed_box(placement, lo_root, hi_root)
    worst = 0.0
    for i, (lo, hi) in enumerate(((xmin, xmax), (ymin, ymax), (zmin, zmax))):
        worst = max(worst, abs(lo_root[i] - lo), abs(hi_root[i] - hi))
    return worst


def _placed_box(placement, lo, hi):
    """The axis-aligned hull, in the part frame, of a local box under a rigid placement."""
    out_lo = [float("inf")] * 3
    out_hi = [float("-inf")] * 3
    for ix in (lo[0], hi[0]):
        for iy in (lo[1], hi[1]):
            for iz in (lo[2], hi[2]):
                for i in range(3):
                    v = (placement[i][0] * ix + placement[i][1] * iy + placement[i][2] * iz
                         + placement[i][3])
                    out_lo[i] = min(out_lo[i], v)
                    out_hi[i] = max(out_hi[i], v)
    return out_lo, out_hi


def crosscheck_contains(cand, original, n_points=4000, seed=1234, built=None):
    """Classify random points against the original CAD solid and against the emitted ROOT shape.

    Points within one model tolerance of the boundary are skipped. For an `O2FlatCSG` the same
    points also count `twinDisagreements`, `Contains` against `Contains_Loop`.
    """
    import random
    from array import array
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
    from OCC.Core.gp import gp_Pnt

    xmin, ymin, zmin, xmax, ymax, zmax = _occ_bbox(original)
    pad = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    shape, placement = built if built is not None else prim.build_root(cand, "containsprobe")
    tol = max(model_tolerance_cm(original), 1.0e-9)
    classifier = BRepClass3d_SolidClassifier(original)
    rng = random.Random(seed)
    disagreements = 0
    scored = 0
    has_twin = hasattr(shape, "Contains_Loop")
    twin_disagreements = 0 if has_twin else None
    for _ in range(n_points):
        p = (rng.uniform(xmin - pad, xmax + pad), rng.uniform(ymin - pad, ymax + pad),
             rng.uniform(zmin - pad, zmax + pad))
        classifier.Perform(gp_Pnt(*p), tol)
        state = classifier.State()
        if state == TopAbs_ON:
            continue
        scored += 1
        # The point is in the part frame; the shape answers in its own.
        local = prim.placement_to_local(placement, p)
        probe = array("d", list(local))
        accelerated = bool(shape.Contains(probe))
        if accelerated != (state == TopAbs_IN):
            disagreements += 1
        if has_twin and accelerated != bool(shape.Contains_Loop(probe)):
            twin_disagreements += 1
    return {"points": scored, "disagreements": disagreements,
            "twinDisagreements": twin_disagreements}


# ------------------------------------------------------------------------------------------
# driving a converter output directory
# ------------------------------------------------------------------------------------------

def load_brep(path):
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Shape, topods
    shape = TopoDS_Shape()
    builder = BRep_Builder()
    if not breptools.Read(shape, str(path), builder):
        raise RuntimeError(f"failed to read {path}")
    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(topods.Solid(exp.Current()))
        exp.Next()
    if len(solids) != 1:
        return shape, len(solids)
    return solids[0], 1


def run_db(db_dir, write_root=True, band_factor=1.0, quiet=False):
    db_dir = Path(db_dir)
    breps = sorted(db_dir.glob("*/brep_*.brep")) or sorted(db_dir.glob("brep_*.brep"))
    if not breps:
        raise SystemExit(f"no brep_*.brep under {db_dir}")
    records = []
    for brep in breps:
        suffix = brep.name[len("brep_"):-len(".brep")]
        part = f"{brep.parent.name}/{suffix}"
        solid, n_solids = load_brep(brep)
        record = process_solid(solid, part, band_factor=band_factor)
        record["brep"] = str(brep)
        record["nSolids"] = n_solids
        if record["accepted"] and write_root:
            target = brep.parent / f"shape_{suffix}.root"
            write_shape_root(record["candidate"], target)
            record["shape"] = str(target)
            record["bboxRootVsOcctCm"] = crosscheck_bbox(record["candidate"])
            record["containsCrosscheck"] = crosscheck_contains(record["candidate"], solid)
        json_target = brep.parent / f"csg_{suffix}.json"
        json_target.write_text(json.dumps(
            {"part": part, "candidate": record["candidate"], "acceptance": record["acceptance"],
             "recogniser": record["recogniser"]}, indent=1))
        records.append(record)
        if not quiet:
            _print_record(record)
    return records


def from_json(folder, quiet=False):
    """Turn every accepted `csg_<part>.json` in a folder into its `shape_<part>.root`.

    Nothing is re-recognised, but `twin_parity` gates each part: a refused part gets no
    `shape_<part>.root` and no `flatcsg_<part>.bin`. Returns `(written, refused)`.
    """
    folder = Path(folder)
    files = sorted(folder.glob("csg_*.json")) or sorted(folder.glob("*/csg_*.json"))
    written, refused = [], []
    for path in files:
        payload = json.loads(path.read_text())
        if not payload.get("candidate"):
            continue
        suffix = path.name[len("csg_"):-len(".json")]
        target = path.parent / f"shape_{suffix}.root"
        shape, placement = prim.build_root(payload["candidate"], "shape")
        parity = twin_parity(shape)
        if parity is not None and parity["disagreements"]:
            refused.append((suffix, parity))
            if not quiet:
                print(f"  [REFUSED] {suffix}: {twin_decline_reason(parity)}; no shape file and "
                      "no sidecar written, so geom.C ships this part one tier down")
            continue
        if payload["candidate"].get("op") == "flatCells":
            # The macro loads the flat sidecar, so a deferred part writes it here, after the gate.
            from cadsupport import flat as flat_writer
            blocks, cells = prim.flat_sidecar_records(payload["candidate"])
            flat_writer.write_sidecar(path.parent / f"flatcsg_{suffix}.bin", blocks, cells)
        write_shape_object(shape, placement, target)
        written.append(target)
        if not quiet:
            print(f"  wrote {target} ({shape.ClassName()})")
    if not quiet:
        print(f"{len(written)} shape file(s) written from {len(files)} description(s)"
              + (f"; {len(refused)} REFUSED by the twin-parity gate" if refused else ""))
    return written, refused


def _print_record(record):
    if record["accepted"]:
        acc = record["acceptance"]
        extra = ""
        if record.get("containsCrosscheck") is not None:
            cc = record["containsCrosscheck"]
            twin = ("" if cc.get("twinDisagreements") is None
                    else f", twin {cc['twinDisagreements']}/{cc['points']}")
            parity = record.get("twinParity")
            box_twin = ("" if not parity
                        else f", twin(boxes x2) {parity['disagreements']}/{parity['points']}")
            extra = (f", ROOT-vs-CAD Contains {cc['disagreements']}/{cc['points']}{twin}{box_twin}"
                     f", bbox(ROOT vs OCCT) {record['bboxRootVsOcctCm']:.2e} cm")
        print(f"  [CSG ] {record['part']}: {record['description']}  "
              f"[{record['recogniser']}]  dV_sym={acc['symmetricDifference']:.3g} cm^3 "
              f"(band {acc['band']:.3g}, rel {acc['relativeToVolume']:.2e}){extra}")
    elif record["recognised"]:
        print(f"  [rej ] {record['part']}: {record['description']} rejected -- {record['reason']}")
    else:
        print(f"  [decl] {record['part']}: {record['reason']}")


def summarise(records):
    n_csg = sum(1 for r in records if r["accepted"])
    n_rej = sum(1 for r in records if r["recognised"] and not r["accepted"])
    n_dec = sum(1 for r in records if not r["recognised"])
    print(f"\n{n_csg}/{len(records)} part(s) accepted as CSG "
          f"({n_rej} recognised but rejected by the symmetric difference, {n_dec} declined "
          f"by the recogniser)")
    return n_csg, n_rej, n_dec


# ------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, help="a gate workdir's db/ directory (walks brep_*.brep)")
    ap.add_argument("--brep", type=Path, help="a single .brep file, in cm")
    ap.add_argument("--from-json", type=Path, dest="from_json",
                    help="write shape_*.root for every accepted csg_*.json in this folder "
                         "(needs PyROOT only; nothing is re-recognised)")
    ap.add_argument("--report", type=Path, help="write the per-part record as JSON")
    ap.add_argument("--no-root", action="store_true",
                    help="recognise and accept, but do not write shape_*.root (no PyROOT needed)")
    ap.add_argument("--band-factor", type=float, default=1.0,
                    help="multiplier on the acceptance band (model tolerance x area); "
                         "default %(default)s")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-self-test", action="store_true",
                    help="skip the self-test that otherwise runs before any emission")
    args = ap.parse_args()
    from cadsupport.selftest_emit import self_test

    if args.self_test:
        ok_a, n_a = accept.self_test()
        ok_e, n_e = self_test(with_root=not args.no_root)
        print(f"\n{ok_a + ok_e}/{n_a + n_e} self-checks passed")
        return 0 if (ok_a == n_a and ok_e == n_e) else 1

    if args.from_json:
        _written, refused = from_json(args.from_json)
        if refused:
            # A refused part is a broken description, not a cosmetic warning: it would have
            # shipped a solid that disagrees with its own reference implementation.
            return 1
        return 0

    if not args.db and not args.brep:
        ap.error("give --db, --brep, --from-json or --self-test")

    if not args.no_self_test:
        ok_a, n_a = accept.self_test(verbose=False)
        ok_e, n_e = self_test(verbose=False, with_root=not args.no_root)
        if ok_a != n_a or ok_e != n_e:
            raise SystemExit(f"self-test failed ({ok_a}/{n_a} acceptance, {ok_e}/{n_e} "
                             "recognise/emit); refusing to emit")
        print(f"[self-test] {ok_a + ok_e}/{n_a + n_e} checks passed")

    if args.brep:
        solid, _n = load_brep(args.brep)
        suffix = args.brep.name[len("brep_"):-len(".brep")]
        record = process_solid(solid, suffix, band_factor=args.band_factor)
        if record["accepted"] and not args.no_root:
            target = args.brep.parent / f"shape_{suffix}.root"
            write_shape_root(record["candidate"], target)
            record["shape"] = str(target)
            record["bboxRootVsOcctCm"] = crosscheck_bbox(record["candidate"])
            record["containsCrosscheck"] = crosscheck_contains(record["candidate"], solid)
        _print_record(record)
        records = [record]
    else:
        records = run_db(args.db, write_root=not args.no_root, band_factor=args.band_factor)
    summarise(records)
    if args.report:
        args.report.write_text(json.dumps(records, indent=1))
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
