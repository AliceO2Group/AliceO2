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

"""The converter's single CSG integration point, `recognise_and_emit()`.

With `--csg auto` each part ships as CSG, else exact surfaces, else tessellated; the other
representations are still written for the gate. `shape_<VOL>_<LID>.root` is written only where
PyROOT imports, and `geom.C` never references a file that was not written.
"""

import json
import sys
from pathlib import Path

from cadsupport import emit, planar, primitives as prim, recognise  # noqa: E402


def have_root():
    try:
        import ROOT  # noqa: F401
        return True
    except Exception:                                            # noqa: BLE001
        return False


def scaled_to_cm(shape, scale_to_cm):
    """A copy of `shape` scaled to cm, the frame and units of the sidecar, mesh, `.brep` and oracle."""
    if scale_to_cm == 1.0:
        return shape
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Pnt, gp_Trsf
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), scale_to_cm)
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def recognise_and_emit(def_shapes, def_names, scale_to_cm, out_folder, sanitize_filename,
                       mode="auto", band_factor=1.0, verbose=True, scaled=None):
    """Recognise every leaf solid; emit what both acceptance tests admit.

    Returns `(csg_files, flat_files, records)`: lid -> `shape_*.root`, lid -> `flatcsg_*.bin` for
    `O2FlatCSG` parts (a part is in exactly one map), and the per-part evidence. `scaled` maps a
    lid to the cm copy the caller already made.
    """
    out_folder = Path(out_folder)
    root_available = have_root()
    csg_files = {}
    flat_files = {}
    records = []
    for lid, shape in def_shapes.items():
        display = def_names.get(lid, "")
        volname = sanitize_filename(display) if display else "vol"
        suffix = f"{volname}_{sanitize_filename(lid)}"
        solid = scaled[lid] if scaled and lid in scaled else scaled_to_cm(shape, scale_to_cm)
        cache = {}
        record = emit.process_solid(solid, suffix, band_factor=band_factor, cache=cache)
        record["lid"] = lid
        record["volume"] = display
        # The placement is derived from the description alone (no ROOT needed), so the deferred
        # `--from-json` path and this one cannot disagree about it. None means identity.
        record["placement"] = (prim.placement_for_candidate(record["candidate"])
                               if record["candidate"] else None)
        (out_folder / f"csg_{suffix}.json").write_text(json.dumps(
            {"part": suffix, "lid": lid, "candidate": record["candidate"],
             "acceptance": record["acceptance"], "recogniser": record["recogniser"],
             "placement": record["placement"]}, indent=1))
        if record["accepted"]:
            is_flat = record["candidate"]["op"] == "flatCells"
            if root_available:
                # Built once and checked before any file is written.
                built = prim.build_root(record["candidate"], "shape")
                record["twinParity"] = emit.twin_parity(built[0])
                record["bboxRootVsOcctCm"] = emit.crosscheck_bbox(
                    record["candidate"], occ_shape=recognise.realised_for(cache, record["candidate"]),
                    built=built)
                record["containsCrosscheck"] = emit.crosscheck_contains(
                    record["candidate"], solid, built=built)
                # Either twin sampling refuses the part: a cell reaches past its declared box.
                parity = record["twinParity"]
                cross_twin = (record["containsCrosscheck"] or {}).get("twinDisagreements")
                if parity is not None and parity["disagreements"]:
                    record["accepted"] = False
                    record["reason"] = emit.twin_decline_reason(parity)
                elif cross_twin:
                    record["accepted"] = False
                    record["reason"] = emit.twin_decline_reason(
                        {"disagreements": cross_twin,
                         "points": record["containsCrosscheck"]["points"]})
                if not record["accepted"]:
                    record["shape"] = None
                    record["flatSidecar"] = None
                else:
                    if is_flat:
                        # Written only here: a deferred part must not advertise a sidecar.
                        record["flatSidecar"] = write_flat_sidecar(
                            record["candidate"], out_folder, suffix)
                    target = (out_folder / f"shape_{suffix}.root").resolve()
                    emit.write_shape_object(built[0], built[1], target)
                    record["shape"] = str(target)
                    if is_flat:
                        flat_files[lid] = record["flatSidecar"]
                    else:
                        csg_files[lid] = str(target)
            else:
                record["shape"] = None
                record["shapeDeferred"] = True
                # Name the real cause: the environment, not the geometry.
                record["reason"] = ("csg deferred: ROOT unavailable in this interpreter; the "
                                    f"accepted candidate is in csg_{suffix}.json -- run "
                                    "`python3 -m cadsupport.emit --from-json <output folder>` from the directory holding the "
                                    "cadsupport package to complete it")
        records.append(record)
        if verbose:
            emit._print_record(record)
            if record.get("shapeDeferred"):
                print(f"  [WARN] {display or lid}: accepted as CSG but NOT emitted -- "
                      "ROOT unavailable; geom.C will dispatch this part one tier down")

    n_csg = sum(1 for r in records if r["accepted"])
    if verbose:
        print(f"CSG recognition ({mode}): {n_csg}/{len(records)} leaf solid(s) accepted as native "
              f"ROOT shapes ({len(csg_files) + len(flat_files)} written, of which "
              f"{len(flat_files)} as flat halfspace solids)")
        if n_csg and not root_available:
            n_deferred = sum(1 for r in records if r.get("shapeDeferred"))
            print(f"  [WARN] PyROOT is not importable in this interpreter: {n_deferred} accepted "
                  "CSG part(s) were NOT emitted and geom.C dispatches them one tier down. "
                  "csg_report.json records each as 'csg deferred: ROOT unavailable'. Run "
                  "`python3 -m cadsupport.emit --from-json <output folder>` from the directory holding the cadsupport "
                  "package, under the O2 environment, then "
                  "reconvert (or re-run the gate), to ship them as CSG.")
    if mode == "required":
        failed = [r for r in records if not r["accepted"]]
        if failed:
            lines = [f"--csg required: {len(failed)}/{len(records)} leaf solid(s) are not CSG:"]
            for r in failed:
                lines.append(f"  {r['volume'] or r['lid']}: {r['reason']}")
            raise ValueError("\n".join(lines))
    return csg_files, flat_files, records


def write_flat_sidecar(cand, out_folder, suffix):
    """Write `flatcsg_<part>.bin` for a `flatCells` candidate; returns its absolute path."""
    from cadsupport import flat
    target = (Path(out_folder) / f"flatcsg_{suffix}.bin").resolve()
    blocks, cells = prim.flat_sidecar_records(cand)
    flat.write_sidecar(target, blocks, cells)
    return str(target)


def write_report(records, path, surface_lids, facet_lids):
    """The per-part cascade report: which representation carries each part, and on what evidence.

    Each row also records `tessellationExact` (`cadsupport/planar.py`). `surface_lids` is a set of
    lids or the lid -> sidecar mapping; only the mapping lets that exactness be computed.
    """
    surface_paths = surface_lids if isinstance(surface_lids, dict) else {}
    rows = []
    tiers = {"csg": 0, "surface": 0, "mesh": 0}
    exactness = {"exact": 0, "approximate": 0, "unknown": 0}
    for record in records:
        lid = record["lid"]
        if record["accepted"] and record.get("shape"):
            tier = "csg"
            why_not_csg = None
            evidence = {
                "recogniser": record["recogniser"],
                "description": record["description"],
                "symmetricDifferenceCm3": record["acceptance"]["symmetricDifference"],
                "bandCm3": record["acceptance"]["band"],
                "relativeToVolume": record["acceptance"]["relativeToVolume"],
                "rootVsCadContains": record.get("containsCrosscheck"),
            }
        elif lid in surface_lids:
            tier = "surface"
            why_not_csg = record["reason"]
            evidence = {"declinedCsgBecause": record["reason"]}
        else:
            tier = "mesh"
            why_not_csg = record["reason"]
            evidence = {"declinedCsgBecause": record["reason"]}
        tiers[tier] += 1
        sidecar = surface_paths.get(lid)
        if sidecar:
            mesh_exact, mesh_reason, mesh_census = planar.tessellation_is_exact(sidecar)
        else:
            mesh_exact, mesh_reason, mesh_census = None, "no exact sidecar for this part", None
        exactness["exact" if mesh_exact else
                  ("approximate" if mesh_exact is False else "unknown")] += 1
        # `part` is the artifact stem, which joins this row to manifest.json and gate.json.
        rows.append({"lid": lid, "part": record.get("part"), "volume": record["volume"],
                     "representation": tier, "shapeFile": record.get("shape"),
                     # The flatcsg_*.bin an O2FlatCSG part ships with; null otherwise.
                     "flatSidecar": record.get("flatSidecar"),
                     # Brief decline reason; None when the part ships as CSG.
                     "whyNotCSG": why_not_csg,
                     "shapeDeferred": bool(record.get("shapeDeferred", False)),
                     # [R | t] from the shape frame to the part frame (3x4 row-major), or null.
                     "shapePlacement": record.get("placement"),
                     # Whether the mesh IS the exact surface solid; null without a sidecar.
                     "tessellationExact": mesh_exact,
                     "tessellationExactWhy": mesh_reason,
                     "surfaceCensus": mesh_census,
                     "evidence": evidence})
    report = {"tiers": tiers, "tessellationExactness": exactness,
              "nLeafSolids": len(records), "parts": rows}
    Path(path).write_text(json.dumps(report, indent=1))
    return report


def print_tier_table(report):
    print("\n=== REPRESENTATION CASCADE (per leaf solid) ===")
    print(f"  {'volume':<28} {'carried by':<10} evidence")
    for row in report["parts"]:
        ev = row["evidence"]
        if row["representation"] == "csg":
            detail = (f"{ev['description']} [{ev['recogniser']}], dV_sym="
                      f"{ev['symmetricDifferenceCm3']:.3g} cm^3 (band {ev['bandCm3']:.3g})")
        else:
            detail = f"declined CSG: {ev['declinedCsgBecause']}"
        print(f"  {(row['volume'] or row['lid'])[:28]:<28} {row['representation']:<10} {detail}")
    exact = report.get("tessellationExactness") or {}
    if exact.get("exact"):
        total = sum(exact.values()) or 1
        print(f"  tessellation is EXACT (every face a planar polygon) for {exact['exact']} of "
              f"{total} part(s) -- {100.0 * exact['exact'] / total:.1f} %; for those the mesh is "
              f"not an approximation of the part, it is the part")
    tiers = report["tiers"]
    print(f"  tiers: CSG {tiers['csg']}, exact surfaces {tiers['surface']}, "
          f"tessellated {tiers['mesh']}  (of {report['nLeafSolids']} leaf solids)")


def csg_placement_var(lid, sanitize_cpp_name):
    """The macro variable holding a CSG part's shape placement. One namer, two call sites."""
    return f"shapePlace_{sanitize_cpp_name(lid)}"


def emit_csg_shape_cpp(lid, vol_display_name, shape_abspath, medium_var, sanitize_cpp_name):
    """geom.C branch for a CSG part: load the TGeoShape and its placement from its own file."""
    safe = sanitize_cpp_name(lid)
    shape_name = vol_display_name if vol_display_name else lid
    return "\n".join([
        f'  TGeoShape *solid_{safe} = LoadShape("{shape_abspath}", "{shape_name}");',
        f'  TGeoVolume *vol_{safe} = new TGeoVolume("{shape_name}", solid_{safe}, {medium_var});',
        f'  TGeoHMatrix *{csg_placement_var(lid, sanitize_cpp_name)} = '
        f'LoadShapePlacement("{shape_abspath}");',
    ])


def emit_csg_composed_placement_cpp(matrix_var, placement_var, composed_var):
    """`composed = partPlacement * shapePlacement`, in that order.

    A point goes shape -> part -> parent; `TGeoHMatrix::Multiply(right)` is `this = this * right`,
    so the part placement is copied and the shape placement is the right operand.
    """
    return "\n".join([
        f"  TGeoHMatrix *{composed_var} = new TGeoHMatrix(*{matrix_var});",
        f"  {composed_var}->Multiply({placement_var});",
    ])


def emit_flat_csg_shape_cpp(lid, vol_display_name, sidecar_abspath, medium_var,
                            sanitize_cpp_name):
    """geom.C branch for an `O2FlatCSG` part: construct, load the sidecar, close.

    A sidecar that fails to load is fatal: a geometry that cannot be built must stop the job.
    """
    safe = sanitize_cpp_name(lid)
    shape_name = vol_display_name if vol_display_name else lid
    return "\n".join([
        f'  auto *solid_{safe} = new o2::cad::O2FlatCSG("{shape_name}");',
        f'  if (!o2::cad::LoadFlatCSG("{sidecar_abspath}", *solid_{safe})) {{',
        f'    ::Fatal("geom", "flat-CSG sidecar for {shape_name} failed to load: '
        f'{sidecar_abspath}");',
        '  }',
        f'  solid_{safe}->CloseShape();',
        f'  if (!solid_{safe}->IsClosed()) {{',
        f'    ::Fatal("geom", "flat-CSG shape {shape_name} refused to close; see the Error above");',
        '  }',
        f'  TGeoVolume *vol_{safe} = new TGeoVolume("{shape_name}", solid_{safe}, {medium_var});',
    ])


FLAT_CPP_PRELUDE = r'''
// --- flat-CSG parts: o2::cad::O2FlatCSG filled from a flatcsg_*.bin sidecar ---
// Both headers are included, never declared by prototype: loadCADGeometryHook JITs this
// macro inside a unique namespace and hoists only '#' lines to global scope, so a
// `namespace o2 { namespace cad {` block here becomes `<wrapper>::o2::cad` and shadows
// the real one -- every later o2::cad:: name then fails to resolve and the module
// silently does not load. O2SurfaceSolidIO.h declares LoadFlatCSG and LoadSurfaceSolid both.
R__ADD_INCLUDE_PATH($O2_ROOT/include)
R__LOAD_LIBRARY(libO2CADSupport)
#include "CADSupport/O2FlatCSG.h"
#include "CADSupport/O2SurfaceSolidIO.h"
#include <TError.h>
'''


CPP_LOADER = r'''
// --- CSG parts: one ROOT-serialised TGeoShape per part, written by Detectors/CADSupport/tools/cadsupport ---
// The file holds exactly one object inheriting from TGeoShape under the key "shape", in cm; and
// optionally a TGeoHMatrix under the key "placement", the rigid transform from the shape's own
// canonical frame into the part's local frame. No "placement" key means the identity, which is
// what every file written before that change means (see O2SolidHarness.h, next to the C++ loader
// that reads the same convention).
TGeoHMatrix* LoadShapePlacement(const char* path) {
  TFile* f = TFile::Open(path, "READ");
  if (!f || f->IsZombie()) {
    throw std::runtime_error(std::string("cannot open CSG shape file: ") + path);
  }
  auto* stored = dynamic_cast<TGeoHMatrix*>(f->Get("placement"));
  // Identity when the file records none. Returning a matrix rather than a null pointer keeps the
  // composition below unconditional, so the placed and unplaced cases go down one code path.
  auto* placement = stored ? new TGeoHMatrix(*stored) : new TGeoHMatrix("identity");
  f->Close();
  delete f;
  return placement;
}

TGeoShape* LoadShape(const char* path, const char* name) {
  TFile* f = TFile::Open(path, "READ");
  if (!f || f->IsZombie()) {
    throw std::runtime_error(std::string("cannot open CSG shape file: ") + path);
  }
  auto* shape = dynamic_cast<TGeoShape*>(f->Get("shape"));
  if (!shape) {
    delete f;
    throw std::runtime_error(std::string("no TGeoShape under key \"shape\" in ") + path);
  }
  // The shape registers itself with gGeoManager on construction and is owned by it; the file can
  // go away.
  shape->SetName(name);
  f->Close();
  delete f;
  return shape;
}
'''
