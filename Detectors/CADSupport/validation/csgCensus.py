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
# Since: 2026-09

"""The recognition census: per solid and per input model, what the CSG tiers would face.

Nothing here emits anything. Per solid it reports

  1. face count and the breakdown by surface type;
  2. whether the solid is quadric-only (plane/cylinder/cone/sphere/torus faces only);
  3. edge count, and for every edge shared by exactly two distinct faces, whether the dihedral is
     convex, concave or tangential — the concave count is the input to the Tier-3 cell estimate;
  4. whether the face set matches a whole-part TGeo primitive template (Tier 1);
  5. how many non-quadric faces are *secretly* analytic and would canonicalise (Tier 0);
  6. volume and bounding box, as reference numbers for later acceptance work.

`--self-test` checks every column against solids with closed-form answers; it also runs before
every census unless `--no-self-test` is given.

Usage
-----
  csgCensus.py --self-test
  csgCensus.py --model .../ExcavatorArm.step [--model ...] --cache /tmp/csgcache --markdown
  csgCensus.py --report --cache /tmp/csgcache          # re-render tables from cache, no OCCT work

The script re-execs itself under the aliBuild Python that can import pythonOCC (see `occ_env.py`).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cadsupport_path  # noqa: F401
from cadsupport.occ_env import ensure_occ  # noqa: E402

ensure_occ()

from OCC.Core.BRep import BRep_Tool  # noqa: E402
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # noqa: E402
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse  # noqa: E402
from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone,  # noqa: E402
                                  BRepPrimAPI_MakeCylinder,
                                  BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus)
from OCC.Core.Geom import (Geom_RectangularTrimmedSurface, Geom_SurfaceOfLinearExtrusion,  # noqa: E402
                           Geom_SurfaceOfRevolution)
from OCC.Core.GeomAbs import GeomAbs_BSplineSurface  # noqa: E402
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve  # noqa: E402
from OCC.Core.ShapeAnalysis import ShapeAnalysis_CanonicalRecognition  # noqa: E402
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader  # noqa: E402
from OCC.Core.IFSelect import IFSelect_RetDone  # noqa: E402
from OCC.Core.TCollection import TCollection_AsciiString  # noqa: E402
from OCC.Core.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool  # noqa: E402
from OCC.Core.TDocStd import TDocStd_Document  # noqa: E402
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID  # noqa: E402
from OCC.Core.TopExp import TopExp_Explorer, topexp  # noqa: E402
from OCC.Core.TopTools import TopTools_IndexedMapOfShape  # noqa: E402
from OCC.Core.TopoDS import topods  # noqa: E402
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool  # noqa: E402
from OCC.Core.gp import gp_Dir, gp_Vec  # noqa: E402
from cadsupport.analytic import CURVE_TYPE_NAME, SURFACE_TYPE_NAME  # noqa: E402
from cadsupport.census import _xyz, bounding_box, edge_census, halfspace_side, volume_of  # noqa: E402
from cadsupport.primitives import _cross, _dot, _norm, _sub  # noqa: E402

CENSUS_FORMAT_VERSION = 3

QUADRIC_TYPES = ("plane", "cylinder", "cone", "sphere", "torus")

# Relative tolerance for template matching (directions, radii, offsets).
TEMPLATE_REL_TOL = 1.0e-6
TEMPLATE_ANG_TOL = 1.0e-6


def _parallel(a, b, tol=TEMPLATE_ANG_TOL):
    return _norm(_cross(a, b)) <= tol


def _antiparallel(a, b, tol=TEMPLATE_ANG_TOL):
    return _parallel(a, b, tol) and _dot(a, b) < 0


def _perp(a, b, tol=TEMPLATE_ANG_TOL):
    return abs(_dot(a, b)) <= tol


def _point_on_axis(p, loc, direction, tol):
    d = _sub(p, loc)
    perp = _sub(d, tuple(c * _dot(d, direction) for c in direction))
    return _norm(perp) <= tol


# --------------------------------------------------------------------------------------------
# surface classification and Tier-0 canonicalisation
# --------------------------------------------------------------------------------------------

def _basis_curve_type(surface, stype):
    """Return the GeomAbs type name of a swept surface's basis curve, or None.

    The cast is attempted only for the type the adaptor reported, since `DownCast` raises otherwise.
    """
    if stype not in ("revolution", "extrusion"):
        return None
    s = surface
    while isinstance(s, Geom_RectangularTrimmedSurface):
        s = s.BasisSurface()
    caster = (Geom_SurfaceOfRevolution if stype == "revolution"
              else Geom_SurfaceOfLinearExtrusion)
    try:
        swept = caster.DownCast(s)
        basis = None if swept is None else swept.BasisCurve()
    except Exception:
        return None
    if basis is None:
        return None
    try:
        return CURVE_TYPE_NAME.get(GeomAdaptor_Curve(basis).GetType(), "other")
    except Exception:
        return "other"


def _canonical_recognition(face, tol):
    """OCCT's own recogniser: is this face secretly a quadric? It has no torus test."""
    from OCC.Core.gp import gp_Cone, gp_Cylinder, gp_Pln, gp_Sphere
    try:
        rec = ShapeAnalysis_CanonicalRecognition(face)
    except Exception:
        return None, None
    for name, meth, holder in (("plane", "IsPlane", gp_Pln()),
                               ("cylinder", "IsCylinder", gp_Cylinder()),
                               ("cone", "IsCone", gp_Cone()),
                               ("sphere", "IsSphere", gp_Sphere())):
        try:
            rec.ClearStatus()
            if getattr(rec, meth)(tol, holder):
                return name, rec.GetGap()
        except Exception:
            continue
    return None, None


def classify_face(face, canonical_tol, do_canonical=True):
    """Classify one face: its carrier type, and — if not a quadric — what it could become."""
    ad = BRepAdaptor_Surface(face, True)
    stype = SURFACE_TYPE_NAME.get(ad.GetType(), "other")
    info = {"type": stype}
    if stype in QUADRIC_TYPES:
        side = halfspace_side(face, ad, stype)
        if side:
            info["side"] = side
            if stype != "plane":
                # A plane has no intrinsic inside; for a curved carrier the two must agree.
                info["orientationAgrees"] = (side == "interior") == \
                    (face.Orientation() != TopAbs_REVERSED)
        return info

    surface = BRep_Tool.Surface(face)
    basis = _basis_curve_type(surface, stype)
    if basis:
        info["basisCurve"] = basis
    # Revolving or extruding a line or a circle always produces a quadric.
    if stype == "revolution" and basis in ("line", "circle"):
        info["canonicalStructural"] = "cone/cylinder/plane" if basis == "line" else "torus/sphere"
    elif stype == "extrusion" and basis in ("line", "circle"):
        info["canonicalStructural"] = "plane" if basis == "line" else "cylinder"

    if do_canonical:
        name, gap = _canonical_recognition(face, canonical_tol)
        if name:
            info["canonicalOCCT"] = name
            info["canonicalGap"] = gap
    return info


# --------------------------------------------------------------------------------------------
# Tier-1 template matching
# --------------------------------------------------------------------------------------------

def _carriers(faces):
    """Extract the analytic carrier of every face; returns None if any face is not a quadric."""
    out = []
    for f in faces:
        ad = BRepAdaptor_Surface(f, True)
        t = SURFACE_TYPE_NAME.get(ad.GetType(), "other")
        if t == "plane":
            pl = ad.Plane()
            ax = pl.Axis()
            n = _xyz(ax.Direction())
            if f.Orientation() == TopAbs_REVERSED:
                n = (-n[0], -n[1], -n[2])
            out.append({"t": "plane", "n": n, "p": _xyz(ax.Location())})
        elif t == "cylinder":
            cy = ad.Cylinder()
            ax = cy.Axis()
            out.append({"t": "cylinder", "d": _xyz(ax.Direction()), "p": _xyz(ax.Location()),
                        "r": cy.Radius()})
        elif t == "cone":
            co = ad.Cone()
            ax = co.Axis()
            out.append({"t": "cone", "d": _xyz(ax.Direction()), "p": _xyz(ax.Location()),
                        "r": co.RefRadius(), "a": co.SemiAngle()})
        elif t == "sphere":
            sp = ad.Sphere()
            out.append({"t": "sphere", "p": _xyz(sp.Location()), "r": sp.Radius()})
        elif t == "torus":
            to = ad.Torus()
            ax = to.Axis()
            out.append({"t": "torus", "d": _xyz(ax.Direction()), "p": _xyz(ax.Location()),
                        "R": to.MajorRadius(), "r": to.MinorRadius()})
        else:
            return None
    return out


def _scale_of(carriers, bbox_diag):
    return max(bbox_diag, 1.0)


def distinct_carriers(carriers, scale):
    """How many distinct halfspaces the face set spans; CAD splits a cylinder into several faces."""
    if carriers is None:
        return None
    tol = TEMPLATE_REL_TOL * scale
    uniq = []
    for c in carriers:
        for u in uniq:
            if u["t"] != c["t"]:
                continue
            if c["t"] == "plane":
                if _parallel(u["n"], c["n"]) and \
                        abs(_dot(_sub(c["p"], u["p"]), u["n"])) <= tol:
                    break
            elif c["t"] == "sphere":
                if _norm(_sub(u["p"], c["p"])) <= tol and abs(u["r"] - c["r"]) <= tol:
                    break
            elif c["t"] == "torus":
                if _parallel(u["d"], c["d"]) and _norm(_sub(u["p"], c["p"])) <= tol \
                        and abs(u["R"] - c["R"]) <= tol and abs(u["r"] - c["r"]) <= tol:
                    break
            else:                                  # cylinder / cone
                if _parallel(u["d"], c["d"]) and _point_on_axis(c["p"], u["p"], u["d"], tol) \
                        and abs(u["r"] - c["r"]) <= tol \
                        and abs(u.get("a", 0.0) - c.get("a", 0.0)) <= TEMPLATE_ANG_TOL:
                    break
        else:
            uniq.append(c)
            continue
    return len(uniq)


def carrier_clusters(carriers, scale):
    """Group the analytic carriers by shared axis / shared normal direction."""
    if carriers is None:
        return None
    tol = TEMPLATE_REL_TOL * scale
    axial = [c for c in carriers if c["t"] in ("cylinder", "cone", "torus")]
    clusters = []
    for c in axial:
        for cl in clusters:
            if _parallel(c["d"], cl["dir"]) and _point_on_axis(c["p"], cl["loc"], cl["dir"], tol):
                cl["members"].append(c)
                break
        else:
            clusters.append({"dir": c["d"], "loc": c["p"], "members": [c]})
    out = []
    for cl in clusters:
        radii = sorted(round(m.get("r", m.get("R", 0.0)), 9) for m in cl["members"])
        out.append({"dir": [round(x, 9) for x in cl["dir"]],
                    "types": sorted({m["t"] for m in cl["members"]}),
                    "n": len(cl["members"]), "radii": radii})
    normals = []
    for c in carriers:
        if c["t"] != "plane":
            continue
        for nd in normals:
            if _parallel(c["n"], nd["dir"]):
                nd["n"] += 1
                break
        else:
            normals.append({"dir": [round(x, 9) for x in c["n"]], "n": 1})
    return {"axisClusters": sorted(out, key=lambda z: (-z["n"], z["radii"])),
            "planeDirections": sorted(normals, key=lambda z: -z["n"]),
            "nAxisClusters": len(out), "nPlaneDirections": len(normals)}


def match_box(carriers, scale):
    planes = [c for c in carriers if c["t"] == "plane"]
    if len(planes) != 6 or len(planes) != len(carriers):
        return None
    used = [False] * 6
    axes = []
    for i in range(6):
        if used[i]:
            continue
        for j in range(i + 1, 6):
            if used[j]:
                continue
            if _antiparallel(planes[i]["n"], planes[j]["n"]):
                used[i] = used[j] = True
                sep = abs(_dot(_sub(planes[j]["p"], planes[i]["p"]), planes[i]["n"]))
                axes.append((planes[i]["n"], sep))
                break
        else:
            return None
    if len(axes) != 3:
        return None
    for a in range(3):
        for b in range(a + 1, 3):
            if not _perp(axes[a][0], axes[b][0]):
                return None
    dims = sorted(round(a[1], 9) for a in axes)
    return {"template": "TGeoBBox", "params": {"dx": dims[0] / 2, "dy": dims[1] / 2,
                                               "dz": dims[2] / 2}}


def _coaxial(items, scale):
    """All items share one axis line (direction up to sign, and location on that line)."""
    if not items:
        return None
    d0 = items[0]["d"]
    p0 = items[0]["p"]
    tol = TEMPLATE_REL_TOL * scale
    for it in items[1:]:
        if not _parallel(it["d"], d0):
            return None
        if not _point_on_axis(it["p"], p0, d0, tol):
            return None
    return d0, p0


def match_tube_or_cone(carriers, scale):
    """Two coaxial cylinders (or cones) + caps perpendicular to the axis, +/- a phi wedge."""
    cyls = [c for c in carriers if c["t"] == "cylinder"]
    cones = [c for c in carriers if c["t"] == "cone"]
    planes = [c for c in carriers if c["t"] == "plane"]
    others = [c for c in carriers if c["t"] not in ("cylinder", "cone", "plane")]
    if others or (not cyls and not cones):
        return None
    if cyls and cones:
        lateral, kind = cyls + cones, "cone"       # mixed cylinder/cone stack -> pcon-like
    elif cyls:
        lateral, kind = cyls, "tube"
    else:
        lateral, kind = cones, "cone"
    if len(lateral) > 2:
        return None
    ax = _coaxial(lateral, scale)
    if ax is None:
        return None
    d, _p = ax
    caps = [pl for pl in planes if _parallel(pl["n"], d)]
    wedge = [pl for pl in planes if _perp(pl["n"], d)]
    if len(caps) != 2 or len(caps) + len(wedge) != len(planes):
        return None
    if len(wedge) not in (0, 2):
        return None
    seg = len(wedge) == 2
    if kind == "tube":
        radii = sorted(c["r"] for c in lateral)
        params = {"rmin": radii[0] if len(radii) == 2 else 0.0, "rmax": radii[-1]}
        name = "TGeoTubeSeg" if seg else "TGeoTube"
    else:
        params = {"nlateral": len(lateral)}
        name = "TGeoConeSeg" if seg else "TGeoCone"
    dz = abs(_dot(_sub(caps[1]["p"], caps[0]["p"]), d)) / 2.0
    params["dz"] = dz
    return {"template": name, "params": params}


def match_sphere(carriers, scale):
    sph = [c for c in carriers if c["t"] == "sphere"]
    if len(sph) != 1:
        return None
    planes = [c for c in carriers if c["t"] == "plane"]
    if len(sph) + len(planes) != len(carriers):
        return None
    return {"template": "TGeoSphere", "params": {"r": sph[0]["r"], "cuts": len(planes)}}


def match_torus(carriers, scale):
    tor = [c for c in carriers if c["t"] == "torus"]
    if len(tor) != 1:
        return None
    planes = [c for c in carriers if c["t"] == "plane"]
    if len(tor) + len(planes) != len(carriers):
        return None
    return {"template": "TGeoTorus", "params": {"R": tor[0]["R"], "r": tor[0]["r"],
                                                "cuts": len(planes)}}


def match_revolution(carriers, scale, faces_info):
    """Every carrier is a surface of revolution about one common axis (TGeoPcon)."""
    if any(fi["type"] not in QUADRIC_TYPES and fi["type"] != "revolution" for fi in faces_info):
        return None
    axial = [c for c in carriers if c["t"] in ("cylinder", "cone", "torus")] if carriers else []
    if not carriers:
        return None
    if not axial:
        return None
    ax = _coaxial(axial, scale)
    if ax is None:
        return None
    d, p = ax
    tol = TEMPLATE_REL_TOL * scale
    nwedge = 0
    for c in carriers:
        if c["t"] in ("cylinder", "cone", "torus"):
            continue
        if c["t"] == "sphere":
            if not _point_on_axis(c["p"], p, d, tol):
                return None
        elif c["t"] == "plane":
            if _parallel(c["n"], d):
                continue                      # a plane perpendicular to the axis: a pcon step
            if _perp(c["n"], d) and _point_on_axis(c["p"], p, d, tol):
                nwedge += 1                   # a plane through the axis: a phi cut
            else:
                return None
        else:
            return None
    if nwedge not in (0, 2):
        return None
    return {"template": "revolution/TGeoPcon-like",
            "params": {"nlateral": len(axial), "phiCut": nwedge == 2}}


def match_extrusion(carriers, scale, faces_info):
    """A closed 2D profile swept along one direction (TGeoXtru)."""
    if any(fi["type"] not in ("plane", "cylinder", "extrusion") for fi in faces_info):
        return None
    if not carriers:
        return None
    cyls = [c for c in carriers if c["t"] == "cylinder"]
    planes = [c for c in carriers if c["t"] == "plane"]
    if len(cyls) + len(planes) != len(carriers):
        return None
    # Candidate extrusion directions: a cylinder axis or a cap-plane normal, not every plane pair.
    candidates = []
    for d in [c["d"] for c in cyls] + [p["n"] for p in planes]:
        if not any(_parallel(d, e) for e in candidates):
            candidates.append(d)
        if len(candidates) > 64:
            return None
    for d in candidates:
        caps = [pl for pl in planes if _parallel(pl["n"], d)]
        walls = [pl for pl in planes if _perp(pl["n"], d)]
        if len(caps) != 2 or len(caps) + len(walls) != len(planes):
            continue
        if any(not _parallel(c["d"], d) for c in cyls):
            continue
        dz = abs(_dot(_sub(caps[1]["p"], caps[0]["p"]), d)) / 2.0
        return {"template": "extrusion/TGeoXtru-like",
                "params": {"nwall": len(walls), "nround": len(cyls), "dz": dz}}
    return None


def tier2_sketch(clusters):
    """A one-line description of what a Tier-2 recogniser would have to build, or why it cannot."""
    if clusters is None:
        return "non-quadric"
    na = clusters["nAxisClusters"]
    np_ = clusters["nPlaneDirections"]
    if na == 0:
        return f"planes only ({np_} directions)"
    sizes = "+".join(str(c["n"]) for c in clusters["axisClusters"])
    return f"{na} axis clusters ({sizes}), {np_} plane directions"


def match_template(faces, faces_info, scale):
    carriers = _carriers(faces)
    if carriers is not None:
        for matcher in (match_box, match_tube_or_cone, match_sphere, match_torus):
            m = matcher(carriers, scale)
            if m:
                return m
    if carriers is not None:
        m = match_revolution(carriers, scale, faces_info)
        if m:
            return m
        m = match_extrusion(carriers, scale, faces_info)
        if m:
            return m
    return {"template": "none", "params": {}}


# --------------------------------------------------------------------------------------------
# per-solid and per-model census
# --------------------------------------------------------------------------------------------

def solid_faces(solid):
    fmap = TopTools_IndexedMapOfShape()
    topexp.MapShapes(solid, TopAbs_FACE, fmap)
    return [topods.Face(fmap.FindKey(i)) for i in range(1, fmap.Size() + 1)]


def census_solid(solid, name, canonical_tol, do_canonical=True, carrier_face_cap=400):
    t0 = time.time()
    faces = solid_faces(solid)
    faces_info = [classify_face(f, canonical_tol, do_canonical) for f in faces]

    by_type = {}
    for fi in faces_info:
        by_type[fi["type"]] = by_type.get(fi["type"], 0) + 1

    nquad = sum(by_type.get(t, 0) for t in QUADRIC_TYPES)
    nfaces = len(faces)
    canon_struct = sum(1 for fi in faces_info if "canonicalStructural" in fi)
    canon_occt = sum(1 for fi in faces_info if "canonicalOCCT" in fi)
    canon_either = sum(1 for fi in faces_info
                       if "canonicalStructural" in fi or "canonicalOCCT" in fi)
    canon_by_type = {}
    canon_to = {}
    for fi in faces_info:
        if "canonicalStructural" in fi or "canonicalOCCT" in fi:
            canon_by_type[fi["type"]] = canon_by_type.get(fi["type"], 0) + 1
        if "canonicalOCCT" in fi:
            k = f"{fi['type']}->{fi['canonicalOCCT']}"
            canon_to[k] = canon_to.get(k, 0) + 1
    gaps = [fi["canonicalGap"] for fi in faces_info
            if fi.get("canonicalGap") is not None]
    basis_hist = {}
    for fi in faces_info:
        if "basisCurve" in fi:
            k = f"{fi['type']}({fi['basisCurve']})"
            basis_hist[k] = basis_hist.get(k, 0) + 1

    bbox = bounding_box(solid)
    diag = 0.0 if bbox is None else _norm(_sub(bbox[3:], bbox[:3]))

    rec = {
        "name": name,
        "faces": nfaces,
        "byType": by_type,
        "quadricFaces": nquad,
        "quadricOnly": nquad == nfaces and nfaces > 0,
        "quadricOnlyAfterTier0": (nquad + canon_either) == nfaces and nfaces > 0,
        "canonicalisableStructural": canon_struct,
        "canonicalisableOCCT": canon_occt,
        "canonicalisableEither": canon_either,
        "canonicalisableByType": canon_by_type,
        "canonicalisableTo": canon_to,
        "basisCurves": basis_hist,
        "maxCanonicalGap": max(gaps) if gaps else None,
        "exteriorHalfspaces": sum(1 for fi in faces_info if fi.get("side") == "exterior"),
        "orientationDisagreements": sum(1 for fi in faces_info
                                        if fi.get("orientationAgrees") is False),
        "bbox": bbox,
        "bboxDiagonal": diag,
        "volume": volume_of(solid),
    }
    rec.update({"edgeCensus": edge_census(solid)})
    rec["concaveEdges"] = rec["edgeCensus"]["concave"] + rec["edgeCensus"]["mixed"]
    rec["concaveEdgesTrusted"] = (rec["concaveEdges"]
                                  - rec["edgeCensus"]["concaveNearTangential"]
                                  - rec["edgeCensus"]["mixedNearTangential"])
    # Zero concave edges means a single CSG cell, not convexity: a through hole has none.
    rec["singleCell"] = rec["concaveEdges"] == 0
    scale = _scale_of(None, diag)
    # The carrier analyses are quadratic in the face count, so they are skipped above the cap.
    if nfaces <= carrier_face_cap:
        carriers = _carriers(faces)
        rec.update(match_template(faces, faces_info, scale))
        rec["distinctCarriers"] = distinct_carriers(carriers, scale)
        clusters = carrier_clusters(carriers, scale)
        rec["carrierClusters"] = clusters
        rec["tier2Sketch"] = tier2_sketch(clusters)
    else:
        rec.update({"template": "not-attempted(size)", "params": {}})
        rec["distinctCarriers"] = None
        rec["carrierClusters"] = None
        rec["tier2Sketch"] = "not-attempted(size)"
    rec["seconds"] = time.time() - t0

    # Instrument identity: the per-type histogram must account for every face, always.
    assert sum(by_type.values()) == nfaces, f"face-type histogram lost faces on {name}"
    return rec


def load_step_solids(path):
    """Read a STEP file and return [(name, TopoDS_Solid)], names from XCAF when present."""
    doc = TDocStd_Document("csg-census")
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {path}")
    reader.Transfer(doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())

    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)
    # Solids come from exploding the free shapes, which keeps their locations; names attach after.
    solids = []
    for i in range(1, labels.Length() + 1):
        exp = TopExp_Explorer(shape_tool.GetShape(labels.Value(i)), TopAbs_SOLID)
        while exp.More():
            solids.append(topods.Solid(exp.Current()))
            exp.Next()

    named = []

    def walk(label, prefix, depth=0):
        if depth > 32:
            return
        nm = ""
        try:
            nm = str(label.GetLabelName() or "")
        except Exception:
            nm = ""
        entry = TCollection_AsciiString()
        TDF_Tool.Entry(label, entry)
        full = f"{prefix}/{nm}" if nm else f"{prefix}/{entry.ToCString()}"
        children = TDF_LabelSequence()
        shape_tool.GetComponents(label, children)
        if children.Length() > 0:
            for k in range(1, children.Length() + 1):
                walk(children.Value(k), full, depth + 1)
            return
        ref = TDF_Label()
        if shape_tool.GetReferredShape(label, ref) and not ref.IsNull():
            sub = TDF_LabelSequence()
            shape_tool.GetComponents(ref, sub)
            if sub.Length() > 0:
                for k in range(1, sub.Length() + 1):
                    walk(sub.Value(k), full, depth + 1)
                return
            label = ref
        shape = shape_tool.GetShape(label)
        if shape is not None and not shape.IsNull():
            named.append((full, shape))

    for i in range(1, labels.Length() + 1):
        walk(labels.Value(i), "")

    name_of = []
    for nm, shape in named:
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp.More():
            name_of.append((nm, topods.Solid(exp.Current())))
            exp.Next()

    out = []
    for i, s in enumerate(solids):
        label = f"solid{i}"
        for nm, proto in name_of:
            if s.IsPartner(proto):
                label = nm
                break
        out.append((label, s))
    return out


def detect_unit_scale_to_cm(path):
    """Same heuristic `O2_CADtoTGeo.py` uses: read the STEP header and look for a unit token."""
    data = Path(path).open("rb").read(4 * 1024 * 1024).decode("latin-1", "ignore").upper()
    for token, scale, name in ((".MILLI.", 0.1, "mm"), (".CENTI.", 1.0, "cm"),
                               (".METRE.", 100.0, "m"), (".METER.", 100.0, "m"),
                               ("INCH", 2.54, "in")):
        if token in data:
            return scale, name
    return 0.1, "mm"


def census_model(path, canonical_tol, do_canonical=True, max_faces=None, progress=True):
    path = Path(path)
    t0 = time.time()
    scale, unit = detect_unit_scale_to_cm(path)
    solids = load_step_solids(path)
    t_load = time.time() - t0

    # Prototypes are keyed by `hash(shape.TShape())`, the `IsPartner` class in O(1).
    protos = []
    proto_of = []
    proto_key = {}
    for _name, solid in solids:
        key = hash(solid.TShape())
        if key not in proto_key:
            proto_key[key] = len(protos)
            protos.append(solid)
        proto_of.append(proto_key[key])

    # The census is per prototype; the record carries its placement count.
    placements = {}
    names = {}
    for i, (nm, _solid) in enumerate(solids):
        p = proto_of[i]
        placements[p] = placements.get(p, 0) + 1
        names.setdefault(p, nm)

    records = []
    for p, solid in enumerate(protos):
        name = names[p]
        nf = len(solid_faces(solid))
        if max_faces is not None and nf > max_faces:
            rec = {"name": name, "faces": nf, "skipped": "face budget"}
        else:
            try:
                rec = census_solid(solid, name, canonical_tol, do_canonical)
            except Exception as exc:                   # a bad solid must not lose the model
                rec = {"name": name, "faces": nf, "error": f"{type(exc).__name__}: {exc}"}
        rec["name"] = name
        rec["index"] = p
        rec["proto"] = p
        rec["placements"] = placements[p]
        rec["isFirstInstance"] = True
        records.append(rec)
        if progress:
            tag = rec.get("template") or rec.get("skipped") or f"ERROR {rec.get('error')}"
            print(f"    proto {p + 1}/{len(protos)} x{placements[p]} "
                  f"{rec.get('faces', '?'):>5} faces {rec.get('seconds', 0.0):6.2f}s  "
                  f"{tag}  {name[:60]}", flush=True)

    # Instrument identity on real data: the placement counts must add up to the bodies found.
    total = sum(r["placements"] for r in records)
    assert total == len(solids), f"placement accounting lost bodies: {total} != {len(solids)}"

    return {
        "formatVersion": CENSUS_FORMAT_VERSION,
        "model": str(path),
        "modelSize": path.stat().st_size,
        "modelMtime": path.stat().st_mtime,
        "unit": unit,
        "unitScaleToCm": scale,
        "canonicalTol": canonical_tol,
        "canonicalEnabled": do_canonical,
        "loadSeconds": t_load,
        "placedSolids": len(solids),
        "prototypeSolids": len(protos),
        "totalSeconds": time.time() - t0,
        "solids": records,
    }


# --------------------------------------------------------------------------------------------
# self-test: check the instrument before believing the table
# --------------------------------------------------------------------------------------------

def self_test(verbose=True):
    """Every column of the census, against solids whose answers are known in closed form."""
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        elif verbose:
            print(f"    ok   {msg}")

    tol = 1e-7

    box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    box_solid = next_solid(box)
    r = census_solid(box_solid, "box", tol)
    check(r["faces"] == 6, "box has 6 faces")
    check(r["byType"].get("plane") == 6, "box faces are all planes")
    check(r["quadricOnly"], "box is quadric-only")
    check(r["edgeCensus"]["edges"] == 12, f"box has 12 edges (got {r['edgeCensus']['edges']})")
    check(r["edgeCensus"]["convex"] == 12,
          f"box has 12 convex edges (got {r['edgeCensus']}) -- SIGN OF THE CONCAVITY TEST")
    check(r["concaveEdges"] == 0 and r["singleCell"], "box is a single cell")
    check(r["template"] == "TGeoBBox", f"box matches TGeoBBox (got {r['template']})")
    check(abs(r["volume"] - 6000.0) < 1e-6, f"box volume 6000 (got {r['volume']})")
    check(r["exteriorHalfspaces"] == 0, "box has no exterior halfspace")
    check(r["distinctCarriers"] == 6, f"box has 6 distinct carriers (got {r['distinctCarriers']})")

    # The trap this project already paid for: VolumeProperties on a single face is 0, silently.
    faces = solid_faces(box_solid)
    face_sum = sum(volume_of(f) for f in faces)
    check(face_sum == 0.0,
          f"per-face VolumeProperties sums to 0, not the solid volume (got {face_sum}) -- "
          "the documented trap; volumes must be taken on the solid")

    cyl = next_solid(BRepPrimAPI_MakeCylinder(3.0, 10.0).Shape())
    r = census_solid(cyl, "cylinder", tol)
    check(r["byType"].get("cylinder") == 1 and r["byType"].get("plane") == 2,
          f"cylinder is 1 cylinder + 2 planes (got {r['byType']})")
    check(r["template"] == "TGeoTube", f"cylinder matches TGeoTube (got {r['template']})")
    check(abs(r["params"]["rmax"] - 3.0) < 1e-9 and abs(r["params"]["dz"] - 5.0) < 1e-9,
          f"cylinder params rmax=3 dz=5 (got {r['params']})")
    check(r["concaveEdges"] == 0, f"cylinder has no concave edge (got {r['edgeCensus']})")
    check(abs(r["volume"] - math.pi * 9.0 * 10.0) < 1e-6, "cylinder volume")
    check(r["exteriorHalfspaces"] == 0,
          f"solid cylinder: material inside its own carrier (got {r['exteriorHalfspaces']})")

    tube = next_solid(BRepAlgoAPI_Cut(BRepPrimAPI_MakeCylinder(3.0, 10.0).Shape(),
                                      BRepPrimAPI_MakeCylinder(1.0, 30.0).Shape()).Shape())
    r = census_solid(tube, "tube", tol)
    check(r["template"] == "TGeoTube", f"annulus matches TGeoTube (got {r['template']})")
    check(abs(r["params"]["rmin"] - 1.0) < 1e-9,
          f"annulus rmin=1 (got {r['params']})")
    # An annulus has no concave edge and is not convex, yet is one CSG cell with an exterior bore.
    check(r["edgeCensus"]["concave"] == 0,
          f"annulus has no concave edge (got {r['edgeCensus']})")
    check(r["exteriorHalfspaces"] == 1,
          f"annulus bore is an exterior halfspace (got {r['exteriorHalfspaces']})")
    check(r["faces"] == 4 and r["distinctCarriers"] == 4,
          f"annulus: 4 faces, 4 distinct carriers (got {r['faces']}, {r['distinctCarriers']})")

    sph = next_solid(BRepPrimAPI_MakeSphere(4.0).Shape())
    r = census_solid(sph, "sphere", tol)
    check(r["template"] == "TGeoSphere", f"sphere matches TGeoSphere (got {r['template']})")
    check(r["concaveEdges"] == 0, f"sphere has no concave edge (got {r['edgeCensus']})")
    check(r["edgeCensus"]["degenerate"] == 2,
          f"sphere has 2 degenerate pole edges (got {r['edgeCensus']})")

    tor = next_solid(BRepPrimAPI_MakeTorus(10.0, 2.0).Shape())
    r = census_solid(tor, "torus", tol)
    check(r["template"] == "TGeoTorus", f"torus matches TGeoTorus (got {r['template']})")
    check(r["quadricOnly"], "torus is quadric-only")
    check(r["exteriorHalfspaces"] == 0, "torus material is inside its own carrier")

    # An L-shape: exactly one concave edge, by construction.
    from OCC.Core.gp import gp_Ax2, gp_Pnt as _P
    b1 = BRepPrimAPI_MakeBox(10.0, 10.0, 2.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Ax2(_P(0, 0, 0), gp_Dir(0, 0, 1)), 2.0, 10.0, 10.0).Shape()
    ell = next_solid(BRepAlgoAPI_Fuse(b1, b2).Shape())
    r = census_solid(ell, "Lshape", tol)
    check(r["edgeCensus"]["concave"] == 1,
          f"L-shape has exactly 1 concave edge (got {r['edgeCensus']})")
    check(not r["singleCell"], "L-shape needs more than one cell")

    plate = BRepPrimAPI_MakeBox(gp_Ax2(_P(-5, -5, 0), gp_Dir(0, 0, 1)), 10.0, 10.0, 4.0).Shape()

    # A THROUGH hole: no concave edge (the material fills a quadrant at each rim), one exterior
    # halfspace, one CSG cell -- box halfspaces intersected with the outside of the cylinder.
    holed = next_solid(BRepAlgoAPI_Cut(
        plate, BRepPrimAPI_MakeCylinder(1.5, 20.0).Shape()).Shape())
    r = census_solid(holed, "through_hole", tol)
    check(r["edgeCensus"]["concave"] == 0,
          f"through hole has no concave edge (got {r['edgeCensus']})")
    check(r["singleCell"] and r["exteriorHalfspaces"] == 1,
          f"through hole is one cell with one exterior halfspace (got "
          f"cell={r['singleCell']} ext={r['exteriorHalfspaces']})")
    check(r["quadricOnly"], "through hole is quadric-only")

    # A BLIND hole: the bottom rim IS concave, because the cylinder's carrier extended would cut
    # material that the solid keeps. That is exactly the witness Tier 3's split loop consumes.
    blind = next_solid(BRepAlgoAPI_Cut(
        plate,
        BRepPrimAPI_MakeCylinder(gp_Ax2(_P(0, 0, 2), gp_Dir(0, 0, 1)), 1.5, 10.0).Shape()).Shape())
    r = census_solid(blind, "blind_hole", tol)
    check(r["edgeCensus"]["concave"] == 1,
          f"blind hole has exactly 1 concave edge (got {r['edgeCensus']})")

    # A groove across the top face: 2 concave edges at the slot floor.
    slot = BRepPrimAPI_MakeBox(gp_Ax2(_P(-2, -20, 2), gp_Dir(0, 0, 1)), 4.0, 40.0, 10.0).Shape()
    r = census_solid(next_solid(BRepAlgoAPI_Cut(plate, slot).Shape()), "groove", tol)
    check(r["edgeCensus"]["concave"] == 2,
          f"groove has exactly 2 concave edges (got {r['edgeCensus']})")

    # --- Tier-0 recogniser: a positive and a NEGATIVE control -------------------------------
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_NurbsConvert
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCC.Core.TColgp import TColgp_Array2OfPnt

    nurbs_cyl = BRepBuilderAPI_NurbsConvert(BRepPrimAPI_MakeCylinder(3.0, 10.0).Shape()).Shape()
    lateral = [f for f in solid_faces(next_solid(nurbs_cyl))
               if BRepAdaptor_Surface(f, True).GetType() == GeomAbs_BSplineSurface]
    check(len(lateral) >= 1, f"NURBS-converted cylinder has a B-spline face (got {len(lateral)})")
    if lateral:
        got, gap = _canonical_recognition(lateral[0], 1e-7)
        check(got == "cylinder" and gap is not None and gap < 1e-7,
              f"positive control: a NURBS-encoded cylinder is recognised as a cylinder "
              f"(got {got}, gap {gap})")

    grid = TColgp_Array2OfPnt(1, 5, 1, 5)
    for i in range(1, 6):
        for j in range(1, 6):
            x, y = (i - 3) * 2.0, (j - 3) * 2.0
            grid.SetValue(i, j, _P(x, y, 0.15 * x * y))     # a saddle: not any quadric of ours
    saddle = BRepBuilderAPI_MakeFace(
        GeomAPI_PointsToBSplineSurface(grid).Surface(), 1e-9).Face()
    got, gap = _canonical_recognition(saddle, 1e-7)
    check(got is None,
          f"NEGATIVE control: a genuine free-form saddle is NOT recognised as a quadric "
          f"(got {got}, gap {gap}) -- without this the Tier-0 count means nothing")

    # `hash(TShape())` and a pairwise IsPartner sweep must define the same classes.
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf, gp_Vec
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(50.0, 0.0, 0.0))
    moved = next_solid(BRepBuilderAPI_Transform(box_solid, trsf, False).Shape())
    other = next_solid(BRepPrimAPI_MakeBox(10.0, 20.0, 30.001).Shape())
    check(box_solid.IsPartner(moved) and hash(box_solid.TShape()) == hash(moved.TShape()),
          "prototype key: a relocated instance is a partner and hashes equal")
    check((not box_solid.IsPartner(other))
          and hash(box_solid.TShape()) != hash(other.TShape()),
          "prototype key: a different body is not a partner and hashes differently")

    # The geometric halfspace-side test and the ORIENTATION flag must never disagree; if they do,
    # one of the two is being read wrong and every exterior-halfspace count is suspect.
    for nm, sh in (("through_hole", holed), ("blind_hole", blind), ("annulus", tube)):
        rr = census_solid(sh, nm, tol)
        check(rr["orientationDisagreements"] == 0,
              f"{nm}: geometric halfspace side agrees with the ORIENTATION flag on every face")

    if verbose:
        if failures:
            print("\n  SELF-TEST FAILURES:")
            for f in failures:
                print(f"    FAIL {f}")
        else:
            print("\n  self-test: all checks passed")
    return failures


def ladder_shapes():
    """The boolean ladder fixtures, rebuilt here so they can be censused too.

    `make_boolean_fixtures.py` is not imported or run; these are
    independent constructions of the same geometry (same radii, same axes, mm) so that the
    census can answer questions about `tube_window` and its siblings — which are synthetic
    fixtures, present in no input model, and therefore invisible to a census of STEP files.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.gp import gp_Ax2, gp_Pnt as P

    def cyl(r, h, o=(0., 0., 0.), d=(0., 0., 1.)):
        return BRepPrimAPI_MakeCylinder(gp_Ax2(P(*o), gp_Dir(*d)), r, h).Shape()

    cz = cyl(10., 60., (0., 0., -30.))
    cx = cyl(10., 60., (-30., 0., 0.), (1., 0., 0.))
    tube = cyl(15., 60., (0., 0., -30.))
    drill = cyl(8., 60., (-30., 0., 0.), (1., 0., 0.))
    return [
        ("cyl_cross_cyl", BRepAlgoAPI_Fuse(cz, cx).Shape()),
        ("cyl_inter_cyl", BRepAlgoAPI_Common(cz, cx).Shape()),
        ("tube_window", BRepAlgoAPI_Cut(tube, drill).Shape()),
        ("cyl_plus_cone", BRepAlgoAPI_Fuse(
            cyl(10., 30.),
            BRepPrimAPI_MakeCone(gp_Ax2(P(0., 0., 30.), gp_Dir(0., 0., 1.)),
                                 10., 5., 20.).Shape()).Shape()),
    ]


def next_solid(shape):
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    if not exp.More():
        raise RuntimeError("no solid in shape")
    return topods.Solid(exp.Current())


# --------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------

def summarise(data, unique=False):
    """Roll up a model. `unique=True` counts each geometric prototype once, not once per
    placement — the basis on which the published ALICE3 numbers were taken."""
    solids = [s for s in data["solids"] if "error" not in s and "skipped" not in s]
    if not unique:
        solids = [s for s in solids for _ in range(s.get("placements", 1))]
    if not solids:
        # Never return an empty summary: failed solids must not look like an empty model.
        return {"solids": 0, "errors": sum(1 for s in data["solids"] if "error" in s),
                "skipped": sum(1 for s in data["solids"] if "skipped" in s),
                "firstError": next((s["error"] for s in data["solids"] if "error" in s), None)}
    faces_total = sum(s["faces"] for s in solids)
    by_type = {}
    for s in solids:
        for k, v in s["byType"].items():
            by_type[k] = by_type.get(k, 0) + v
    canon = {}
    for s in solids:
        for k, v in s.get("canonicalisableByType", {}).items():
            canon[k] = canon.get(k, 0) + v
    canon_to = {}
    for s in solids:
        for k, v in s.get("canonicalisableTo", {}).items():
            canon_to[k] = canon_to.get(k, 0) + v
    gaps = [s["maxCanonicalGap"] for s in solids if s.get("maxCanonicalGap") is not None]
    basis = {}
    for s in solids:
        for k, v in s.get("basisCurves", {}).items():
            basis[k] = basis.get(k, 0) + v
    tmpl = {}
    for s in solids:
        tmpl[s["template"]] = tmpl.get(s["template"], 0) + 1
    concave_hist = {}
    for s in solids:
        b = s["concaveEdges"]
        key = ("0" if b == 0 else "1-2" if b <= 2 else "3-10" if b <= 10 else
               "11-50" if b <= 50 else "51-200" if b <= 200 else ">200")
        concave_hist[key] = concave_hist.get(key, 0) + 1
    concave_hist_trusted = {}
    for s in solids:
        b = s["concaveEdgesTrusted"]
        key = ("0" if b == 0 else "1-2" if b <= 2 else "3-10" if b <= 10 else
               "11-50" if b <= 50 else "51-200" if b <= 200 else ">200")
        concave_hist_trusted[key] = concave_hist_trusted.get(key, 0) + 1
    return {
        "solids": len(solids),
        "faces": faces_total,
        "byType": by_type,
        "basisCurves": basis,
        "canonicalisableByType": canon,
        "canonicalisableTo": canon_to,
        "maxCanonicalGap": max(gaps) if gaps else None,
        "quadricOnly": sum(1 for s in solids if s["quadricOnly"]),
        "quadricOnlyAfterTier0": sum(1 for s in solids if s["quadricOnlyAfterTier0"]),
        "tier0Rescues": sum(1 for s in solids
                            if s["quadricOnlyAfterTier0"] and not s["quadricOnly"]),
        "singleCell": sum(1 for s in solids if s["singleCell"]),
        "singleCellAndQuadric": sum(1 for s in solids if s["singleCell"] and s["quadricOnly"]),
        "singleCellAndQuadricAfterTier0": sum(1 for s in solids if s["singleCell"]
                                              and s["quadricOnlyAfterTier0"]),
        "exteriorHalfspaces": sum(s["exteriorHalfspaces"] for s in solids),
        "orientationDisagreements": sum(s["orientationDisagreements"] for s in solids),
        "tangentialEdges": sum(s["edgeCensus"]["tangential"] for s in solids),
        "mixedEdges": sum(s["edgeCensus"]["mixed"] for s in solids),
        "edgeErrors": sum(s["edgeCensus"]["error"] for s in solids),
        "nonManifoldEdges": sum(s["edgeCensus"]["nonManifold"] for s in solids),
        "templates": tmpl,
        # "not-attempted(size)" is not a match.
        "templateMatched": sum(1 for s in solids
                               if s["template"] not in ("none", "not-attempted(size)")),
        "templateNotAttempted": sum(1 for s in solids
                                    if s["template"] == "not-attempted(size)"),
        "primitiveMatched": sum(1 for s in solids if s["template"].startswith("TGeo")),
        "concaveHistogram": concave_hist,
        "concaveTotal": sum(s["concaveEdges"] for s in solids),
        "concaveHistogramTrusted": concave_hist_trusted,
        "concaveTotalTrusted": sum(s["concaveEdgesTrusted"] for s in solids),
        "singleCellTrusted": sum(1 for s in solids if s["concaveEdgesTrusted"] == 0),
        "carriersVsFaces": [sum(s["distinctCarriers"] for s in solids
                                if s.get("distinctCarriers") is not None),
                            sum(s["faces"] for s in solids
                                if s.get("distinctCarriers") is not None)],
        "concaveNearTangential": sum(s["edgeCensus"]["concaveNearTangential"] for s in solids),
        "mixedNearTangential": sum(s["edgeCensus"]["mixedNearTangential"] for s in solids),
        "edgesTotal": sum(s["edgeCensus"]["edges"] for s in solids),
        "errors": sum(1 for s in data["solids"] if "error" in s),
        "skipped": sum(1 for s in data["solids"] if "skipped" in s),
    }


def markdown_model(data, limit=None, unique=True):
    lines = []
    name = Path(data["model"]).name
    s = summarise(data, unique=unique)
    lines.append(f"### `{name}`")
    lines.append("")
    lines.append(f"Unit `{data['unit']}` (x{data['unitScaleToCm']} to cm); "
                 f"{data.get('prototypeSolids', '?')} prototype solids in "
                 f"{data.get('placedSolids', '?')} placements; "
                 f"load {data['loadSeconds']:.1f} s, census total {data['totalSeconds']:.1f} s. "
                 f"One row per {'prototype' if unique else 'placement'}.")
    lines.append("")
    lines.append("| # | n | part | faces | halfsp | plane | cyl | cone | sph | tor | free-form |"
                 " swept | quadric-only | edges | concave | trusted | 1 cell? | Tier-0 | "
                 "template | volume | bbox diag |")
    lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
                 " ---: | :---: | ---: | ---: | ---: | :---: | ---: | --- | ---: | ---: |")
    rows = data["solids"]
    counts = {r.get("proto", r.get("index")): r.get("placements", 1) for r in rows}
    if limit is not None:
        rows = rows[:limit]
    for r in rows:
        n = counts.get(r.get("proto", r.get("index")), 1)
        if "error" in r or "skipped" in r:
            what = f"ERROR {r['error'][:60]}" if "error" in r else "skipped"
            lines.append(f"| {r.get('index', '')} | {n} | `{r['name'][-40:]}` | "
                         f"{r.get('faces', '?')} |" + " |" * 15 + f" {what} | | |")
            continue
        bt = r["byType"]
        free = bt.get("bspline", 0) + bt.get("bezier", 0) + bt.get("offset", 0) + \
            bt.get("other", 0)
        swept = bt.get("revolution", 0) + bt.get("extrusion", 0)
        hs = r.get("distinctCarriers")
        lines.append(
            f"| {r['index']} | {n} | `{r['name'][-40:]}` | {r['faces']} | "
            f"{'-' if hs is None else hs} | {bt.get('plane', 0)} | "
            f"{bt.get('cylinder', 0)} | {bt.get('cone', 0)} | {bt.get('sphere', 0)} | "
            f"{bt.get('torus', 0)} | {free} | {swept} | "
            f"{'Y' if r['quadricOnly'] else '.'} | {r['edgeCensus']['edges']} | "
            f"{r['concaveEdges']} | {r['concaveEdgesTrusted']} | "
            f"{'Y' if r['singleCell'] else '.'} | "
            f"{r['canonicalisableEither']} | {r['template']} | "
            f"{r['volume']:.4g} | {r['bboxDiagonal']:.4g} |")
    lines.append("")
    lines.append("Prototype roll-up: " + json.dumps(summarise(data, unique=True), sort_keys=True))
    lines.append("")
    lines.append("Placement roll-up: " + json.dumps(summarise(data, unique=False),
                                                    sort_keys=True))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------

def cache_path(cache_dir, model):
    return Path(cache_dir) / (Path(model).name.replace(" ", "_") + ".census.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], help="STEP/BREP model to census")
    ap.add_argument("--cache", default="/tmp/csgcache", help="directory for per-model JSON")
    ap.add_argument("--refresh", action="store_true", help="ignore an existing cache entry")
    ap.add_argument("--report", action="store_true", help="render tables from cache only")
    ap.add_argument("--markdown", action="store_true", help="print markdown tables")
    ap.add_argument("--limit-rows", type=int, default=None, help="rows per model in markdown")
    ap.add_argument("--max-faces", type=int, default=None,
                    help="skip solids with more faces than this")
    ap.add_argument("--canonical-tol", type=float, default=1.0e-7,
                    help="tolerance for ShapeAnalysis_CanonicalRecognition")
    ap.add_argument("--no-canonical", action="store_true",
                    help="skip OCCT canonical recognition (much faster, loses Tier-0 column)")
    ap.add_argument("--ladder", action="store_true",
                    help="census the boolean ladder fixtures (rebuilt in-process) and exit")
    ap.add_argument("--self-test", action="store_true", help="run the instrument checks and exit")
    ap.add_argument("--no-self-test", action="store_true",
                    help="do not run the instrument checks before a census")
    args = ap.parse_args()

    if args.self_test:
        print("csg.census self-test")
        return 1 if self_test() else 0

    if args.ladder:
        for name, shape in ladder_shapes():
            r = census_solid(next_solid(shape), name, args.canonical_tol)
            print(f"{name:<24} faces={r['faces']:>3} halfspaces={r['distinctCarriers']:>3} "
                  f"concave={r['concaveEdges']:>3} oneCell={r['singleCell']!s:<5} "
                  f"template={r['template']:<26} {r['tier2Sketch']}")
        return 0

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    if args.report:
        datas = [json.loads(p.read_text()) for p in sorted(cache.glob("*.census.json"))]
    else:
        if not args.no_self_test:
            print("csg.census self-test")
            if self_test(verbose=True):
                print("self-test failed; refusing to produce a table from a broken instrument")
                return 1
            print("")
        datas = []
        for model in args.model:
            cp = cache_path(cache, model)
            if cp.exists() and not args.refresh:
                d = json.loads(cp.read_text())
                if (d.get("formatVersion") == CENSUS_FORMAT_VERSION
                        and d.get("modelMtime") == Path(model).stat().st_mtime
                        and d.get("canonicalEnabled") == (not args.no_canonical)):
                    print(f"  cached: {model}")
                    datas.append(d)
                    continue
            print(f"  census: {model}")
            d = census_model(model, args.canonical_tol, not args.no_canonical, args.max_faces)
            cp.write_text(json.dumps(d, indent=1))
            print(f"  wrote {cp}  ({d['totalSeconds']:.1f} s)")
            datas.append(d)

    if args.markdown:
        for d in datas:
            print(markdown_model(d, args.limit_rows))
    else:
        for d in datas:
            print(f"\n{Path(d['model']).name}")
            print(f"  prototypes: {json.dumps(summarise(d, unique=True), sort_keys=True)}")
            print(f"  placements: {json.dumps(summarise(d, unique=False), sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
