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
# Since: 2026-02

"""
O2_CADtoTGeo.py -- STEP/XCAF -> ROOT TGeo conversion.

It writes a ROOT macro (geom.C) and, into --output-folder, one facet file per leaf logical volume,
facets_<VOLNAME>_<LID>.bin; with --exact-surfaces also surfaces_<VOLNAME>_<LID>.bin sidecars (and
brep_*.brep with --dump-brep), and with --csg native ROOT shapes. Materials come from a BOM CSV
(--materials-csv) or a media sidecar (--media-json). VOLNAME is the XCAF label name and LID the
label entry. The STEP length unit is detected, or set with --step-unit; TGeo uses cm.

Facet file format (little-endian):
  uint32 nTriangles
  then nTriangles * 9 * float32:
    ax ay az bx by bz cx cy cz
"""

import argparse
import csv
import json
import math
import random
import re
import struct
import sys
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Dict, List, Optional, Pattern, Tuple

import numpy as np

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepTools import breptools, BRepTools_WireExplorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom2dAdaptor import Geom2dAdaptor_Curve
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.Geom2d import Geom2d_TrimmedCurve
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.Geom2dConvert import geom2dconvert
from OCC.Core.Convert import Convert_TgtThetaOver2
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
    GeomAbs_BezierCurve, GeomAbs_BSplineCurve,
)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopAbs import TopAbs_REVERSED, TopAbs_WIRE, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import topods
from OCC.Extend.TopologyUtils import TopologyExplorer

from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool
from OCC.Core.TCollection import TCollection_AsciiString
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from cadsupport import accept  # noqa: E402
from cadsupport.analytic import (CURVE_TYPE_NAME, SURFACE_TYPE_NAME,  # noqa: E402
                                 _analytic_surface_gap, _analytic_surface_proposals,
                                 _sample_surface_for_recognition, _self_test_bezier_patch,
                                 _self_test_tapered_near_circle, _v_cross, _v_dot)


# -------------------------------
# STEP/XCAF loading
# -------------------------------

def load_step_with_xcaf(path: str):
    doc = TDocStd_Document("pythonocc-doc")
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetLayerMode(True)

    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed for: {path}")

    reader.Transfer(doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    return doc, shape_tool


def label_id(label: TDF_Label) -> str:
    s = TCollection_AsciiString()
    TDF_Tool.Entry(label, s)
    return s.ToCString()


def label_name(label: TDF_Label) -> str:
    # Uses the XCAF/STEP name when present; can be empty.
    try:
        n = label.GetLabelName()
        if n:
            return str(n)
    except Exception:
        pass
    return ""


# -------------------------------
# Units
# -------------------------------

def step_unit_scale_to_cm(step_unit: str) -> float:
    step_unit = (step_unit or "auto").lower()
    if step_unit == "mm":
        return 0.1
    if step_unit == "cm":
        return 1.0
    if step_unit == "m":
        return 100.0
    if step_unit == "in":
        return 2.54
    if step_unit == "ft":
        return 30.48
    raise ValueError(f"Unknown --step-unit {step_unit} (use auto, mm, cm, m, in, ft)")


def detect_step_length_unit(step_path: str) -> str:
    """
    Heuristic unit detection by scanning STEP file text for common unit tokens.
    This avoids relying on OCCT APIs that can vary across pythonOCC builds.

    Returns one of: mm, cm, m, in, ft. Defaults to mm if uncertain.
    """
    p = _Path(step_path)
    # STEP can be huge: read only the first few MB; units are near the header.
    max_bytes = 4 * 1024 * 1024
    data = p.open("rb").read(max_bytes).decode("latin-1", errors="ignore").upper()

    if ".MILLI." in data:
        return "mm"
    if ".CENTI." in data:
        return "cm"
    if ".METRE." in data or ".METER." in data:
        return "m"
    if "INCH" in data:
        return "in"
    if "FOOT" in data or "FEET" in data:
        return "ft"

    # Conservative default for mechanical CAD STEP is mm
    return "mm"


@dataclass(frozen=True)
class ClipBox:
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float

    @classmethod
    def from_values(cls, values: List[float]) -> "ClipBox":
        if len(values) != 6:
            raise ValueError("--clip-box expects 6 values: xmin ymin zmin xmax ymax zmax")
        xmin, ymin, zmin, xmax, ymax, zmax = (float(v) for v in values)
        if not (xmin < xmax and ymin < ymax and zmin < zmax):
            raise ValueError("--clip-box requires xmin<xmax, ymin<ymax, and zmin<zmax")
        return cls(xmin, ymin, zmin, xmax, ymax, zmax)

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax)


@dataclass(frozen=True)
class NameFilter:
    include: Tuple[Pattern[str], ...]
    exclude: Tuple[Pattern[str], ...]

    @classmethod
    def from_patterns(cls, include: List[str], exclude: List[str], case_sensitive: bool = False) -> "NameFilter":
        flags = 0 if case_sensitive else re.IGNORECASE
        return cls(
            tuple(re.compile(pattern, flags) for pattern in include),
            tuple(re.compile(pattern, flags) for pattern in exclude),
        )

    @property
    def active(self) -> bool:
        return bool(self.include or self.exclude)

    @property
    def has_include(self) -> bool:
        return bool(self.include)

    def _text(self, lid: str, name: str) -> str:
        return f"{name} {lid}".strip()

    def matches_include(self, lid: str, name: str) -> bool:
        text = self._text(lid, name)
        return any(pattern.search(text) for pattern in self.include)

    def matches_exclude(self, lid: str, name: str) -> bool:
        text = self._text(lid, name)
        return any(pattern.search(text) for pattern in self.exclude)


# -------------------------------
# Triangulation helpers
# -------------------------------

def triangulate_asbbox(shape, scale_to_cm: float = 1.0):
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()

    p000 = (xmin, ymin, zmin)
    p001 = (xmin, ymin, zmax)
    p010 = (xmin, ymax, zmin)
    p011 = (xmin, ymax, zmax)
    p100 = (xmax, ymin, zmin)
    p101 = (xmax, ymin, zmax)
    p110 = (xmax, ymax, zmin)
    p111 = (xmax, ymax, zmax)

    triangles = [
        (p000, p100, p110), (p000, p110, p010),
        (p001, p111, p101), (p001, p011, p111),
        (p000, p101, p100), (p000, p001, p101),
        (p010, p110, p111), (p010, p111, p011),
        (p000, p010, p011), (p000, p011, p001),
        (p100, p101, p111), (p100, p111, p110),
    ]
    tris = np.array([a + b + c for (a, b, c) in triangles], dtype=float)
    return tris * scale_to_cm if scale_to_cm != 1.0 else tris


def triangulate_CAD_solid(my_solid, meshparam, scale_to_cm: float = 1.0):
    lin_defl = float(meshparam.get("lin_defl", 0.1))
    ang_defl = float(meshparam.get("ang_defl", 0.1))

    BRepMesh_IncrementalMesh(my_solid, lin_defl, False, ang_defl, True)

    chunks = []
    for face in TopologyExplorer(my_solid).faces():
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None or triangulation.NbTriangles() == 0:
            continue

        trsf = loc.Transformation()
        nodes = np.array([(p.X(), p.Y(), p.Z()) for p in
                          (triangulation.Node(i).Transformed(trsf)
                           for i in range(1, triangulation.NbNodes() + 1))], dtype=float)
        idx = np.array([triangulation.Triangle(i).Get()
                        for i in range(1, triangulation.NbTriangles() + 1)], dtype=np.int64) - 1
        if face.Orientation() == TopAbs_REVERSED:
            idx = idx[:, [0, 2, 1]]
        chunks.append(nodes[idx].reshape(-1, 9))

    tris = np.concatenate(chunks) if chunks else np.zeros((0, 9))
    return tris * scale_to_cm if scale_to_cm != 1.0 else tris


# -------------------------------
# Volume helpers (for density)
# -------------------------------

def volume_cm3_of_shape(shape, scale_to_cm: float) -> float:
    """Compute CAD solid volume in cm^3 (using STEP->cm scale)."""
    try:
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        # volume returned in STEP length units^3
        v = float(props.Mass())
        return v * (scale_to_cm ** 3)
    except Exception:
        pass

    # Fallback: bounding-box volume (rough but always defined)
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    dx, dy, dz = (xmax - xmin) * scale_to_cm, (ymax - ymin) * scale_to_cm, (zmax - zmin) * scale_to_cm
    return max(dx, 0.0) * max(dy, 0.0) * max(dz, 0.0)


def _leaf_volume_cm3(lid: str, scale_to_cm: float) -> float:
    """The CAD volume of a leaf before clipping, in cm^3, or 0.0 when it cannot be computed."""
    shape = def_volume_source.get(lid)
    if shape is None:
        return 0.0
    try:
        return volume_cm3_of_shape(shape, scale_to_cm=scale_to_cm)
    except Exception:
        return 0.0


# -------------------------------
# Naming helpers
# -------------------------------

def import_csg_hook():
    """Import `cadsupport/hook.py` lazily."""
    from cadsupport import hook
    return hook


def sanitize_cpp_name(s: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z]", "_", s)
    if not safe:
        safe = "x"
    if not (safe[0].isalpha() or safe[0] == "_"):
        safe = "_" + safe
    return safe


def sanitize_filename(s: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z]", "_", s)
    return safe or "x"


# -------------------------------
# Binary facet IO
# -------------------------------

def write_facets_bin(path: _Path, triangles):
    path.parent.mkdir(parents=True, exist_ok=True)
    tris = np.asarray(triangles, dtype=float).reshape(-1, 9)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(tris)))
        f.write(tris.astype("<f4").tobytes())


# -------------------------------
# Exact-surface classification probes (--surface-report)
# -------------------------------
# They classify each face of a leaf solid and never modify the emitted geometry.

# The C++ support matrix: planes with line/arc/B-spline wires; cylinder/cone/sphere/torus with a
# parametric-rectangle or a general (phi, v) trim that wraps no more than a full turn.
_SUPPORTED_SURFACE_TYPES = {"plane", "cylinder", "cone", "sphere", "torus"}
# An ellipse is exact here: a conic IS a rational quadratic B-spline.
_SUPPORTED_PLANAR_CURVES = {"line", "circle", "ellipse", "bspline", "bezier"}
# Pcurve kinds the quadric extractor maps exactly into (phi, v): a line stays a line, the rest
# become B-splines under the affine (u, v) map.
_SUPPORTED_QUADRIC_CURVES = {"line", "circle", "ellipse", "bspline", "bezier"}


def _xyz(v, scale: float = 1.0) -> List[float]:
    return [v.X() * scale, v.Y() * scale, v.Z() * scale]


def _surface_params(adaptor: BRepAdaptor_Surface, surf_type: str, scale_to_cm: float) -> dict:
    """Extracts the analytic parameters (lengths in cm, angles in rad) for simple types."""
    s = scale_to_cm
    try:
        if surf_type == "plane":
            pln = adaptor.Plane()
            ax3 = pln.Position()
            return {
                "origin_cm": _xyz(ax3.Location(), s),
                "normal": _xyz(pln.Axis().Direction()),
                "axis_u": _xyz(ax3.XDirection()),
                "axis_v": _xyz(ax3.YDirection()),
            }
        if surf_type == "cylinder":
            cyl = adaptor.Cylinder()
            ax3 = cyl.Position()
            return {
                "origin_cm": _xyz(ax3.Location(), s),
                "axis": _xyz(cyl.Axis().Direction()),
                "ref_axis_u": _xyz(ax3.XDirection()),
                "radius_cm": cyl.Radius() * s,
            }
        if surf_type == "cone":
            cone = adaptor.Cone()
            ax3 = cone.Position()
            return {
                "origin_cm": _xyz(ax3.Location(), s),
                "axis": _xyz(cone.Axis().Direction()),
                "ref_axis_u": _xyz(ax3.XDirection()),
                "ref_radius_cm": cone.RefRadius() * s,
                "half_angle_rad": cone.SemiAngle(),
                "apex_cm": _xyz(cone.Apex(), s),
            }
        if surf_type == "sphere":
            sph = adaptor.Sphere()
            ax3 = sph.Position()
            return {
                "center_cm": _xyz(ax3.Location(), s),
                "polar_axis": _xyz(ax3.Direction()),
                "ref_axis_u": _xyz(ax3.XDirection()),
                "radius_cm": sph.Radius() * s,
            }
        if surf_type == "torus":
            tor = adaptor.Torus()
            ax3 = tor.Position()
            return {
                "center_cm": _xyz(ax3.Location(), s),
                "axis": _xyz(ax3.Direction()),
                "ref_axis_u": _xyz(ax3.XDirection()),
                "major_radius_cm": tor.MajorRadius() * s,
                "minor_radius_cm": tor.MinorRadius() * s,
            }
    except Exception as exc:
        return {"error": f"parameter extraction failed: {exc}"}
    return {}


def _edge_pcurve_is_iso(edge, face, uv_bounds) -> bool:
    """True when the edge's 2D pcurve on the face is iso-parametric (u or v constant)."""
    try:
        curve2d, first, last = BRep_Tool.CurveOnSurface(edge, face)
    except Exception:
        return False
    if curve2d is None:
        return False
    us, vs = [], []
    for i in range(5):
        t = first + (last - first) * i / 4.0
        p = curve2d.Value(t)
        us.append(p.X())
        vs.append(p.Y())
    umin, umax, vmin, vmax = uv_bounds
    tol_u = 1e-6 * max(1.0, abs(umax - umin))
    tol_v = 1e-6 * max(1.0, abs(vmax - vmin))
    return (max(us) - min(us) <= tol_u) or (max(vs) - min(vs) <= tol_v)


def classify_face(face, scale_to_cm: float, recognize_surfaces: bool = True,
                  recognition=None, key=None) -> dict:
    """Classifies a single TopoDS face: analytic type, parameters, wires and edges.

    With `recognize_surfaces` a face whose stored type has no extractor also goes to the
    canonical-form recognizer (a surface-only, optimistic claim). `recognition`, when given,
    receives the recognizer's result under `key`, for the extraction.
    """
    adaptor = BRepAdaptor_Surface(face)
    surf_type = SURFACE_TYPE_NAME.get(adaptor.GetType(), "unknown")

    try:
        uv_bounds = list(breptools.UVBounds(face))
    except Exception:
        uv_bounds = [float("nan")] * 4

    record = {
        "type": surf_type,
        "orientation_reversed": face.Orientation() == TopAbs_REVERSED,
        "uv_bounds": uv_bounds,
        "params": _surface_params(adaptor, surf_type, scale_to_cm),
        "wires": [],
    }

    if recognize_surfaces and surf_type not in _SUPPORTED_SURFACE_TYPES and not any(math.isnan(x) for x in uv_bounds):
        rec = _recognize_analytic_surface(adaptor, uv_bounds)
        if recognition is not None:
            recognition[key] = rec
        if rec is not None:
            record["recognized_type"] = rec["kind"]
            record["recognized_residual"] = rec["residual"]
            # The achieved gap in cm; `recognized_residual` is it relative to the patch diagonal.
            record["recognized_gap_cm"] = rec["gap"] * scale_to_cm
            record["recognized_gap_relative"] = rec["gap_relative"]

    try:
        outer_wire = breptools.OuterWire(face)
    except Exception:
        outer_wire = None

    wx = TopExp_Explorer(face, TopAbs_WIRE)
    while wx.More():
        wire = topods.Wire(wx.Current())
        curve_types: Dict[str, int] = {}
        n_edges = 0
        n_degenerated = 0
        all_pcurves_iso = True

        ex = TopExp_Explorer(wire, TopAbs_EDGE)
        while ex.More():
            edge = topods.Edge(ex.Current())
            n_edges += 1
            if BRep_Tool.Degenerated(edge):
                # degenerate edges (sphere poles, cone apex) carry no 3D curve;
                # their pcurves are iso lines by construction
                n_degenerated += 1
            else:
                try:
                    ctype = CURVE_TYPE_NAME.get(BRepAdaptor_Curve(edge).GetType(), "unknown")
                except Exception:
                    ctype = "unknown"
                curve_types[ctype] = curve_types.get(ctype, 0) + 1
                if not _edge_pcurve_is_iso(edge, face, uv_bounds):
                    all_pcurves_iso = False
            ex.Next()

        record["wires"].append({
            "outer": bool(outer_wire is not None and wire.IsSame(outer_wire)),
            "n_edges": n_edges,
            "n_degenerated": n_degenerated,
            "curve_types": curve_types,
            "all_pcurves_iso": all_pcurves_iso,
        })
        wx.Next()

    return record


def face_supported(record: dict) -> Tuple[bool, Optional[str]]:
    """Evaluates one classify_face record against the current C++ support matrix."""
    surf_type = record["type"]
    if surf_type not in _SUPPORTED_SURFACE_TYPES:
        recognized = record.get("recognized_type")
        if recognized is not None:
            record["trim_kind"] = "recognized"
            return True, None
        return False, f"unsupported surface type '{surf_type}'"

    curve_types = set()
    for w in record["wires"]:
        curve_types.update(w["curve_types"].keys())

    if surf_type == "plane":
        bad = curve_types - _SUPPORTED_PLANAR_CURVES
        if bad:
            return False, f"plane with unsupported boundary curves: {sorted(bad)}"
        record["trim_kind"] = "wires"
        return True, None

    # Quadrics: only the boundary-curve type limits eligibility here.
    bad = curve_types - _SUPPORTED_QUADRIC_CURVES
    if bad:
        record["trim_kind"] = "general"
        return False, f"{surf_type} with unsupported trim curves: {sorted(bad)}"
    is_rectangle = len(record["wires"]) == 1 and all(w["all_pcurves_iso"] for w in record["wires"])
    record["trim_kind"] = "parametric-rectangle" if is_rectangle else "general"
    return True, None


def distill_reasons(reasons: List[str]) -> Optional[str]:
    """Fold a per-face reason list into one brief line, most frequent first.

    "40 face(s): unsupported surface type 'bspline'; 2 face(s): ..." -- the `why_not_surface` field.
    """
    if not reasons:
        return None
    counts: Dict[str, int] = {}
    for r in reasons:
        counts[r] = counts.get(r, 0) + 1
    return "; ".join(f"{n} face(s): {r}"
                     for r, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build_surface_report(step_path: str, scale_to_cm: float, recognize_surfaces: bool = True,
                         recognition=None) -> dict:
    """Builds the JSON-serializable exact-conversion eligibility report over def_shapes.

    With `recognize_surfaces` it also tallies the recognition pre-pass; `recognition` collects each
    recognition by (lid, face index) for `extract_surfaces_for_shape`.
    """
    volumes = {}
    n_eligible = 0
    face_type_counts: Dict[str, int] = {}
    curve_type_counts: Dict[str, int] = {}
    fallback_reasons: Dict[str, int] = {}
    recognized_surface_counts: Dict[str, int] = {}
    recognized_stored_type_counts: Dict[str, int] = {}

    recognized_max_gap_cm: Dict[str, float] = {}
    n_eligible_without_recognition = 0
    n_rescued_by_recognition = 0

    for lid, shape in def_shapes.items():
        faces = []
        for index, face in enumerate(TopologyExplorer(shape).faces()):
            rec = classify_face(face, scale_to_cm, recognize_surfaces=recognize_surfaces,
                                recognition=recognition, key=(lid, index))
            ok, reason = face_supported(rec)
            rec["supported"] = ok
            if reason:
                rec["reason"] = reason
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
            faces.append(rec)

            face_type_counts[rec["type"]] = face_type_counts.get(rec["type"], 0) + 1
            for w in rec["wires"]:
                for ctype, n in w["curve_types"].items():
                    curve_type_counts[ctype] = curve_type_counts.get(ctype, 0) + n
            recognized_kind = rec.get("recognized_type")
            if recognized_kind is not None:
                recognized_surface_counts[recognized_kind] = recognized_surface_counts.get(recognized_kind, 0) + 1
                recognized_stored_type_counts[rec["type"]] = recognized_stored_type_counts.get(rec["type"], 0) + 1
                gap = rec.get("recognized_gap_cm", 0.0)
                recognized_max_gap_cm[recognized_kind] = max(recognized_max_gap_cm.get(recognized_kind, 0.0), gap)

        eligible = bool(faces) and all(f["supported"] for f in faces)
        # The coverage *delta* recognition is responsible for: how the same solid would score with
        # the pre-pass switched off. Quoting `n_eligible` on its own does not say that.
        eligible_without = bool(faces) and all(
            f["supported"] and f.get("recognized_type") is None for f in faces)
        if eligible:
            n_eligible += 1
        if eligible_without:
            n_eligible_without_recognition += 1
        elif eligible:
            n_rescued_by_recognition += 1
        vol_recognized: Dict[str, int] = {}
        vol_gap = 0.0
        for f in faces:
            k = f.get("recognized_type")
            if k is not None:
                vol_recognized[k] = vol_recognized.get(k, 0) + 1
                vol_gap = max(vol_gap, f.get("recognized_gap_cm", 0.0))
        volumes[lid] = {
            "name": def_names.get(lid, ""),
            "n_faces": len(faces),
            "eligible": eligible,
            "eligible_without_recognition": eligible_without,
            "recognized_counts": vol_recognized,
            "recognized_max_gap_cm": vol_gap,
            # Brief reason this solid cannot be a SurfaceSolid; extraction may refine it.
            "why_not_surface": None if eligible else distill_reasons(
                [f.get("reason") or f"unsupported {f['type']} face"
                 for f in faces if not f["supported"]]),
            "faces": faces,
        }

    return {
        "report_version": 1,
        "step_file": step_path,
        "scale_to_cm": scale_to_cm,
        "summary": {
            "n_volumes": len(volumes),
            "n_eligible": n_eligible,
            "face_type_counts": face_type_counts,
            "curve_type_counts": curve_type_counts,
            "fallback_reasons": fallback_reasons,
            "recognized_surface_counts": recognized_surface_counts,
            "recognized_stored_type_counts": recognized_stored_type_counts,
            "recognized_max_gap_cm": recognized_max_gap_cm,
            "recognized_acceptance_tolerance_relative": _RECOGNIZE_TOL_EXACT,
            "n_eligible_without_recognition": n_eligible_without_recognition,
            "n_rescued_by_recognition": n_rescued_by_recognition,
        },
        "volumes": volumes,
    }


# -------------------------------
# Surface sidecar binary IO
# -------------------------------
# Versioned binary sidecar for exact surfaces (surfaces_*.bin), read by o2::cad::LoadSurfaceSolid.

SURFACE_SIDECAR_MAGIC = b"O2SS"
# Version 2 appends a float64 model tolerance (cm) to the fixed header.
# Version 3 appends a uint32 edge-table size and, per surface, its boundary edges' (edgeId, flags).
SURFACE_SIDECAR_VERSION = 3
SURFACE_TYPE_ENUM = {"plane": 1, "cylinder": 2, "cone": 3, "sphere": 4, "torus": 5}
CURVE_TYPE_ENUM = {"line": 0, "arc": 1, "bspline": 2}
SURFACE_FLAG_INNER_WALL = 1 << 0

# Per-boundary-edge flag bits, version 3.
EDGE_FLAG_REVERSED = 1 << 0    # the face traverses the edge against the edge's own direction
EDGE_FLAG_DEGENERATE = 1 << 1  # BRep_Tool.Degenerated: a cone apex / sphere pole, no 3D curve
EDGE_FLAG_ANCHORED = 1 << 2    # entry i is trim curve i of this surface, in flattened wire order


def build_edge_table(shape):
    """Index every TopoDS_Edge of \\a shape once, and return (map, edge_id).

    `edge_id(edge)` is a 0-based id stable for the whole solid: two faces' trims share an edge by id.
    """
    edge_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

    def edge_id(edge) -> int:
        return edge_map.FindIndex(edge) - 1  # FindIndex is 1-based; 0 means "not in the map"

    return edge_map, edge_id


def face_boundary_edge_refs(face, edge_id, anchored: bool, wires=None) -> List[Tuple[int, int]]:
    """The face's boundary edges as ordered (edgeId, flags) pairs, in `_face_wire_edges` order.

    `anchored` says whether the record carries the wire block; `wires` is
    `list(_face_wire_edges(face))` when the caller has it.
    """
    refs: List[Tuple[int, int]] = []
    base_flags = EDGE_FLAG_ANCHORED if anchored else 0
    for _wire, _is_outer, edges in (_face_wire_edges(face) if wires is None else wires):
        for edge, _start_vertex in edges:
            flags = base_flags
            if edge.Orientation() == TopAbs_REVERSED:
                flags |= EDGE_FLAG_REVERSED
            if BRep_Tool.Degenerated(edge):
                flags |= EDGE_FLAG_DEGENERATE
            refs.append((edge_id(edge), flags))
    return refs


def write_surfaces_bin(path: _Path, surfaces: List[dict], model_tolerance_cm: float = 0.0,
                       n_model_edges: int = 0):
    """Write a surfaces_*.bin sidecar, version 3."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(SURFACE_SIDECAR_MAGIC)
        f.write(struct.pack("<III", SURFACE_SIDECAR_VERSION, len(surfaces), 0))
        f.write(struct.pack("<d", float(model_tolerance_cm)))
        f.write(struct.pack("<I", int(n_model_edges)))
        for srec in surfaces:
            stype = SURFACE_TYPE_ENUM[srec["type"]]
            flags = SURFACE_FLAG_INNER_WALL if srec.get("inner_wall") else 0
            params = [float(p) for p in srec.get("params", [])]
            f.write(struct.pack("<III", stype, flags, len(params)))
            if params:
                f.write(struct.pack(f"<{len(params)}d", *params))
            wires = srec.get("wires", [])
            f.write(struct.pack("<I", len(wires)))
            for w in wires:
                role = 0 if w.get("role", "outer") == "outer" else 1
                edges = w.get("edges", [])
                f.write(struct.pack("<II", role, len(edges)))
                for e in edges:
                    ctype = CURVE_TYPE_ENUM[e["curve"]]
                    cparams = [float(x) for x in e["params"]]
                    f.write(struct.pack("<II", ctype, len(cparams)))
                    if cparams:
                        f.write(struct.pack(f"<{len(cparams)}d", *cparams))
            edge_refs = srec.get("edge_refs", [])
            f.write(struct.pack("<I", len(edge_refs)))
            for edge_id, edge_flags in edge_refs:
                f.write(struct.pack("<IB", int(edge_id), int(edge_flags)))


def write_brep_cm(path: _Path, shape, scale_to_cm: float):
    """
    Write `shape` as an OCCT BREP file scaled to cm, like the sidecar and the mesh, and return the
    scaled shape.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = import_csg_hook().scaled_to_cm(shape, scale_to_cm)
    if not breptools.Write(shape, str(path)):
        raise RuntimeError(f"failed to write BREP file {path}")
    return shape


# -------------------------------
# Exact-surface extraction (--exact-surfaces)
# -------------------------------
# A leaf solid becomes an O2BVHSurfaceSolid only when every face extracts exactly; otherwise it
# keeps the tessellated fallback (auto) or fails (required).

_EXTRACT_TOL = 1.e-7


def _planar_frame(face, scale_to_cm: float):
    """Return (origin_cm, axis_u, axis_v) for a planar face, with axisU x axisV pointing
    along the *outward* face normal (OCC surface normal for a FORWARD face, its opposite for
    a REVERSED one). Robust to a left-handed OCC ax3 where XDirection x YDirection = -normal."""
    adaptor = BRepAdaptor_Surface(face)
    pln = adaptor.Plane()
    ax3 = pln.Position()
    s = scale_to_cm
    origin_cm = _xyz(ax3.Location(), s)
    axis_u = _xyz(ax3.XDirection())
    ydir = _xyz(ax3.YDirection())
    normal = _xyz(pln.Axis().Direction())
    outward_sign = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
    if outward_sign * _v_dot(_v_cross(axis_u, ydir), normal) > 0.0:
        axis_v = ydir
    else:
        axis_v = [-c for c in ydir]
    return origin_cm, axis_u, axis_v


def _face_wire_edges(face):
    """Yield (wire, is_outer, [edges-in-connected-order]) for every wire of the face."""
    try:
        outer_wire = breptools.OuterWire(face)
    except Exception:
        outer_wire = None
    wx = TopExp_Explorer(face, TopAbs_WIRE)
    while wx.More():
        wire = topods.Wire(wx.Current())
        edges = []
        we = BRepTools_WireExplorer(wire, face)
        while we.More():
            edges.append((we.Current(), we.CurrentVertex()))
            we.Next()
        is_outer = outer_wire is not None and wire.IsSame(outer_wire)
        yield wire, is_outer, edges
        wx.Next()


def _planar_projector(origin_cm, axis_u, axis_v, s):
    """Return project(gp_Pnt) -> (u, v): the point's plane-local coordinates in cm."""
    def project(pnt) -> Tuple[float, float]:
        rel = [pnt.X() * s - origin_cm[0], pnt.Y() * s - origin_cm[1], pnt.Z() * s - origin_cm[2]]
        return _v_dot(rel, axis_u), _v_dot(rel, axis_v)
    return project


def _arc_edge_params(edge, curve, project, s) -> Tuple[Optional[List[float]], Optional[str]]:
    """Build [cu, cv, radius, startAngle, phiSweep] for a circular boundary edge.

    The signed sweep is recovered by sampling the 3D edge in *wire-traversal* order (the edge
    is walked backwards when its orientation is REVERSED relative to the underlying curve),
    projecting each sample into the plane frame and unwrapping the polar angle. This is robust
    to full circles (single periodic edge -> +/-2pi), arcs wider than pi, and either winding.
    """
    circ = curve.Circle()
    cu, cv = project(circ.Location())
    radius = circ.Radius() * s
    first, last = curve.FirstParameter(), curve.LastParameter()
    reversed_edge = edge.Orientation() == TopAbs_REVERSED
    angles: List[float] = []
    for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = (1.0 - tau) if reversed_edge else tau
        u, v = project(curve.Value(first + t * (last - first)))
        angles.append(math.atan2(v - cv, u - cu))
    unwrapped = [angles[0]]
    for a in angles[1:]:
        d = a - unwrapped[-1]
        d -= 2.0 * math.pi * math.floor((d + math.pi) / (2.0 * math.pi))  # wrap into (-pi, pi]
        unwrapped.append(unwrapped[-1] + d)
    sweep = unwrapped[-1] - unwrapped[0]
    if abs(sweep) < _EXTRACT_TOL:
        return None, "planar arc edge has a degenerate sweep"
    return [cu, cv, radius, unwrapped[0], sweep], None


def _bspline_flat_params(first: float, last: float, reversed_edge: bool, pole_xform, to_bspline):
    """Flat sidecar B-spline record [degree, nPoles, poles(2*nPoles), weights(nPoles),
    knots(nPoles+degree+1)] for a curve segment [first, last].

    `to_bspline(lo, hi)` trims the source curve to [lo, hi] and returns a clamped (Geom or Geom2d)
    BSplineCurve; `pole_xform(pole)` maps one control point to its output (u, v). The curve is
    trimmed *before* conversion so the parametrisation matches the edge; a periodic result is made
    non-periodic. Poles/weights/knots are reversed when the edge runs opposite the curve."""
    lo, hi = (first, last) if first <= last else (last, first)
    bs = to_bspline(lo, hi)
    if bs is None:
        return None
    if bs.IsPeriodic():
        bs.SetNotPeriodic()
    degree = bs.Degree()
    nb = bs.NbPoles()
    if degree < 1 or nb < degree + 1:
        return None
    poles = []
    weights = []
    for i in range(1, nb + 1):
        u, v = pole_xform(bs.Pole(i))
        poles.append((u, v))
        weights.append(bs.Weight(i))
    flat = []
    for i in range(1, bs.NbKnots() + 1):
        flat.extend([bs.Knot(i)] * bs.Multiplicity(i))
    if len(flat) != nb + degree + 1:
        return None
    if reversed_edge:
        poles.reverse()
        weights.reverse()
        span = flat[0] + flat[-1]
        flat = [span - k for k in reversed(flat)]
    params = [float(degree), float(nb)]
    for u, v in poles:
        params.extend([float(u), float(v)])
    params.extend(float(w) for w in weights)
    params.extend(float(k) for k in flat)
    return params


# Relative residual below which a sampled trim curve is taken as EXACTLY a line or a circle;
# an almost-circle stays a B-spline.
_CANONICAL_CURVE_TOL = 1.e-9


def _recognize_canonical_curve(samples, poles=None):
    """Recognize a sampled 2D trim curve as an exact line or circle in its output domain.

    `samples` are points in the output domain, in edge direction; collinear `poles`, when given,
    prove a straight segment. Returns ("line", [u0, v0, u1, v1]), ("arc", [cu, cv, r, a0, sweep])
    or (None, None).
    """
    points = np.asarray(samples, dtype=float)
    if len(points) < 3:
        return None, None
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if extent < _EXTRACT_TOL:
        return None, None

    # --- straight line
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length > _EXTRACT_TOL:
        unit = chord / chord_length
        def off_axis(candidates):
            rel = candidates - points[0]
            return float(np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0]).max() / extent)
        straight = off_axis(points) < _CANONICAL_CURVE_TOL
        if straight and poles is not None and len(poles) >= 2:
            straight = off_axis(np.asarray(poles, dtype=float)) < _CANONICAL_CURVE_TOL
        if straight:
            # reject a curve that doubles back along its own chord: geometrically it is not the
            # segment from the first point to the last one, however collinear the samples are
            along = (points - points[0]) @ unit
            if np.all(np.diff(along) >= -_CANONICAL_CURVE_TOL * extent):
                return "line", [float(points[0][0]), float(points[0][1]),
                                float(points[-1][0]), float(points[-1][1])]

    # --- circle: |P - C|^2 = R^2 linearized as 2 P.C + (R^2 - |C|^2) = |P|^2, one least-squares
    # solve with no initial guess. A closed loop (zero chord) lands here as well as an open arc.
    matrix = np.column_stack([2.0 * points, np.ones(len(points))])
    solution, *_ = np.linalg.lstsq(matrix, np.einsum('ij,ij->i', points, points), rcond=None)
    centre = solution[:2]
    radius_sq = solution[2] + float(centre @ centre)
    if radius_sq <= 0.0:
        return None, None
    radius = math.sqrt(radius_sq)
    if float(np.abs(np.linalg.norm(points - centre, axis=1) - radius).max() / extent) >= _CANONICAL_CURVE_TOL:
        return None, None
    # sweep by accumulating signed angle steps, so a full turn and the traversal sense survive
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    steps = np.diff(angles)
    steps = (steps + math.pi) % (2.0 * math.pi) - math.pi
    sweep = float(steps.sum())
    if abs(sweep) < _EXTRACT_TOL:
        return None, None
    return "arc", [float(centre[0]), float(centre[1]), radius, float(angles[0]), sweep]


def _sample_curve_in_domain(curve, first, last, reversed_edge, point_map, n=64):
    """Sample an OCC curve over [first, last] and map each point into the output domain, ordered
    along the edge. `point_map(p)` takes the curve's own point type to an output (u, v)."""
    lo, hi = (first, last) if first <= last else (last, first)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo <= 0.0:
        return None
    try:
        samples = [point_map(curve.Value(float(t))) for t in np.linspace(lo, hi, n)]
    except Exception:
        return None
    if reversed_edge:
        samples.reverse()
    return samples


def _planar_bspline_edge_params(edge, project) -> Optional[List[float]]:
    """Sidecar B-spline record for a planar face's B-spline / Bezier boundary edge.

    The 3D boundary curve lies in the plane, so projecting its control poles into the plane frame
    (an affine map) yields the exact 2D B-spline. Returns None on failure (caller falls back)."""
    try:
        curve3d, first, last = BRep_Tool.Curve(edge)
        if curve3d is None:
            return None
        reversed_edge = edge.Orientation() == TopAbs_REVERSED

        def to_bspline(lo, hi):
            trimmed = Geom_TrimmedCurve(curve3d, lo, hi)
            return geomconvert.CurveToBSplineCurve(trimmed, Convert_TgtThetaOver2)

        return _bspline_flat_params(first, last, reversed_edge, project, to_bspline)
    except Exception:
        return None


def _planar_canonical_edge(edge, project, params):
    """Recognize a planar face's B-spline boundary edge as an exact line or arc in the plane frame.

    `params` is the already-extracted flat B-spline record, whose poles are reused as the convex
    hull evidence for straightness. Returns ("line"|"arc", canonical_params) or (None, None)."""
    try:
        curve3d, first, last = BRep_Tool.Curve(edge)
        if curve3d is None:
            return None, None
        reversed_edge = edge.Orientation() == TopAbs_REVERSED
        samples = _sample_curve_in_domain(curve3d, first, last, reversed_edge, project)
        if not samples:
            return None, None
        n_poles = int(params[1])
        poles = [(params[2 + 2 * i], params[3 + 2 * i]) for i in range(n_poles)]
        return _recognize_canonical_curve(samples, poles)
    except Exception:
        return None, None


def extract_planar_face(face, scale_to_cm: float, frame_override=None,
                        wires=None) -> Tuple[Optional[dict], Optional[str]]:
    """Convert a planar TopoDS face into a sidecar 'plane' surface record with general
    line/arc/B-spline boundary wires; any other boundary curve forces a fallback.

    `frame_override` (origin_cm, axis_u, axis_v) replaces the face's own plane frame, for a face
    recognized as flat whose stored type is not a plane.
    """
    if frame_override is not None:
        origin_cm, axis_u, axis_v = frame_override
    else:
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() != GeomAbs_Plane:
            return None, f"not a plane ({SURFACE_TYPE_NAME.get(adaptor.GetType(), 'unknown')})"
        origin_cm, axis_u, axis_v = _planar_frame(face, scale_to_cm)
    s = scale_to_cm
    project = _planar_projector(origin_cm, axis_u, axis_v, s)

    wires_out: List[dict] = []
    for wire, is_outer, edges in (_face_wire_edges(face) if wires is None else wires):
        classified = []  # (edge, curve, geom_type, projected start (u, v))
        for edge, start_vertex in edges:
            if BRep_Tool.Degenerated(edge):
                return None, "planar face has a degenerated boundary edge"
            try:
                curve = BRepAdaptor_Curve(edge)
                gt = curve.GetType()
            except Exception:
                gt = None
            if gt not in (GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
                          GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
                name = CURVE_TYPE_NAME.get(gt, "unknown")
                return None, (f"planar boundary edge is a {name} curve "
                              "(only line/circle/ellipse/bspline supported)")
            classified.append((edge, curve, gt, project(BRep_Tool.Pnt(start_vertex))))

        n = len(classified)
        if n == 0:
            return None, "planar face has an empty wire"

        # Canonical-form pre-pass, before the polygon check: a straight B-spline becomes a line.
        resolved = []  # per edge: ("line", None) | ("arc", params) | ("bspline", params)
        for edge, curve, gt, _start_uv in classified:
            if gt == GeomAbs_Line:
                resolved.append(("line", None))
            elif gt == GeomAbs_Circle:
                params, reason = _arc_edge_params(edge, curve, project, s)
                if params is None:
                    return None, reason
                resolved.append(("arc", params))
            else:  # ellipse / B-spline / Bezier: project the 3D poles into the plane frame
                # An ellipse is its exact rational quadratic B-spline; the projection is an isometry.
                params = _planar_bspline_edge_params(edge, project)
                if params is None:
                    return None, "planar B-spline boundary edge extraction failed"
                canonical_kind, canonical = _planar_canonical_edge(edge, project, params)
                if canonical_kind == "line":
                    resolved.append(("line", None))
                elif canonical_kind == "arc":
                    resolved.append(("arc", canonical))
                else:
                    resolved.append(("bspline", params))

        n_curved = sum(1 for kind, _ in resolved if kind != "line")
        if n_curved == 0 and n < 3:
            return None, "planar polygon wire has fewer than 3 edges"

        seg_edges = []
        for i, (kind, params) in enumerate(resolved):
            if kind == "line":
                u0, v0 = classified[i][3]
                u1, v1 = classified[(i + 1) % n][3]
                seg_edges.append({"curve": "line", "params": [u0, v0, u1, v1]})
            else:
                seg_edges.append({"curve": kind, "params": params})
        wires_out.append({"role": "outer" if is_outer else "inner", "edges": seg_edges})

    if not wires_out:
        return None, "planar face has no wires"
    n_outer = sum(1 for w in wires_out if w["role"] == "outer")
    if n_outer != 1:
        return None, f"planar face has {n_outer} outer wires (expected exactly 1)"

    return {"type": "plane", "params": list(origin_cm) + list(axis_u) + list(axis_v), "wires": wires_out}, None


def _quadric_phi_range(ax3, umin: float, umax: float) -> Tuple[float, float]:
    """Map an OCC angular U-range [umin, umax] to the C++ (phiStart, phiSweep).

    The C++ bounded quadrics measure phi in a right-handed frame with YDir = axis x refU.
    OCC's stored YDirection equals that only for a *direct* (right-handed) gp_Ax3; otherwise
    it is negated, so a point at OCC parameter u sits at C++ phi = -u and the range mirrors.
    Returns a positive sweep clamped into (0, 2pi].
    """
    sweep = umax - umin
    two_pi = 2.0 * math.pi
    if sweep <= 0.0:
        sweep += two_pi
    sweep = min(sweep, two_pi)
    phi_start = umin if ax3.Direct() else -umax
    return phi_start, sweep


def _quadric_trim_wire(face, map_uv, wires=None) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Build general line/arc/B-spline trim wires in a quadric face's parametric (phi, v) domain.

    `map_uv(u, v)` is the affine map to the C++ (phi, height/theta) domain; a curved pcurve becomes a
    B-spline whose poles it maps exactly. Returns (wires, None) with exactly one outer wire, or
    (None, reason).
    """
    wires_out: List[dict] = []
    for _wire, is_outer, edges in (_face_wire_edges(face) if wires is None else wires):
        parsed = []  # per edge: {"kind": "line", "start": (phi, v)} or {"kind": "bspline", ...}
        for edge, _start_vertex in edges:
            curve2d, first, last = BRep_Tool.CurveOnSurface(edge, face)
            if curve2d is None:
                return None, "quadric boundary edge has no 2D pcurve"
            reversed_edge = edge.Orientation() == TopAbs_REVERSED
            ctype = Geom2dAdaptor_Curve(curve2d).GetType()
            if ctype == GeomAbs_Line:
                param = last if reversed_edge else first
                p = curve2d.Value(param)
                parsed.append({"kind": "line", "start": map_uv(p.X(), p.Y())})
            elif ctype in (GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
                def to_bspline(lo, hi, c2=curve2d):
                    trimmed = Geom2d_TrimmedCurve(c2, lo, hi)
                    return geom2dconvert.CurveToBSplineCurve(trimmed, Convert_TgtThetaOver2)

                params = _bspline_flat_params(first, last, reversed_edge,
                                              lambda p: map_uv(p.X(), p.Y()), to_bspline)
                if params is None:
                    return None, "quadric B-spline pcurve extraction failed"
                # Pre-pass: a B-spline pcurve that is exactly a line in (phi, v) is stored as one.
                samples = _sample_curve_in_domain(curve2d, first, last, reversed_edge,
                                                  lambda p: map_uv(p.X(), p.Y()))
                n_poles = int(params[1])
                poles = [(params[2 + 2 * i], params[3 + 2 * i]) for i in range(n_poles)]
                kind, canonical = _recognize_canonical_curve(samples, poles) if samples else (None, None)
                if kind == "line":
                    parsed.append({"kind": "line", "start": (canonical[0], canonical[1])})
                elif kind == "arc":
                    parsed.append({"kind": "arc", "params": canonical,
                                   "start": (canonical[0] + canonical[2] * math.cos(canonical[3]),
                                             canonical[1] + canonical[2] * math.sin(canonical[3]))})
                else:
                    parsed.append({"kind": "bspline", "params": params, "start": (params[2], params[3])})
            else:
                name = CURVE_TYPE_NAME.get(ctype, "unknown")
                return None, f"quadric boundary pcurve is a {name} curve (unsupported)"
        n = len(parsed)
        if n == 0:
            return None, "quadric trim wire has no edges"
        if all(p["kind"] == "line" for p in parsed) and n < 3:
            return None, "quadric line trim wire has fewer than 3 edges"
        seg_edges = []
        for i, p in enumerate(parsed):
            if p["kind"] == "line":
                u0, v0 = p["start"]
                u1, v1 = parsed[(i + 1) % n]["start"]
                seg_edges.append({"curve": "line", "params": [u0, v0, u1, v1]})
            elif p["kind"] == "arc":
                seg_edges.append({"curve": "arc", "params": p["params"]})
            else:
                seg_edges.append({"curve": "bspline", "params": p["params"]})
        wires_out.append({"role": "outer" if is_outer else "inner", "edges": seg_edges})
    if not wires_out:
        return None, "quadric face has no wires"
    n_outer = sum(1 for w in wires_out if w["role"] == "outer")
    if n_outer != 1:
        return None, f"quadric face has {n_outer} outer trim wires (expected exactly 1)"
    return wires_out, None


def _quadric_trim_fills_uv_box(face, uv_bounds, wires=None) -> bool:
    """True when a quadric face's trim is exactly its parametric-rectangle UV box, so the scalar
    parameters describe it: one line-bounded wire whose (u, v) polygon area equals the box area."""
    umin, umax, vmin, vmax = uv_bounds
    box_area = abs((umax - umin) * (vmax - vmin))
    if box_area <= _EXTRACT_TOL:
        return False
    wires = list(_face_wire_edges(face)) if wires is None else wires
    if len(wires) != 1:
        return False
    _wire, _is_outer, edges = wires[0]
    points = []
    for edge, _start_vertex in edges:
        curve2d, first, last = BRep_Tool.CurveOnSurface(edge, face)
        if curve2d is None or Geom2dAdaptor_Curve(curve2d).GetType() != GeomAbs_Line:
            return False
        param = last if edge.Orientation() == TopAbs_REVERSED else first
        p = curve2d.Value(param)
        points.append((p.X(), p.Y()))
    area = 0.0
    n = len(points)
    for i in range(n):
        u0, v0 = points[i]
        u1, v1 = points[(i + 1) % n]
        area += u0 * v1 - u1 * v0
    return abs(0.5 * area - box_area) <= 1e-6 * box_area


def extract_cylindrical_face(face, scale_to_cm: float, wires=None) -> Tuple[Optional[dict], Optional[str]]:
    """Cylindrical face -> a 'cylinder' surface record; U = azimuth, V = height along the axis."""
    adaptor = BRepAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Cylinder:
        return None, "not a cylinder"
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    cyl = adaptor.Cylinder()
    ax3 = cyl.Position()
    s = scale_to_cm
    center = _xyz(ax3.Location(), s)
    axis = _xyz(cyl.Axis().Direction())
    ref_u = _xyz(ax3.XDirection())
    radius = cyl.Radius() * s
    height_min, height_max = vmin * s, vmax * s
    if height_max - height_min <= _EXTRACT_TOL:
        return None, "cylindrical face has a degenerate height range"
    phi_start, phi_sweep = _quadric_phi_range(ax3, umin, umax)
    inner_wall = face.Orientation() == TopAbs_REVERSED
    params = list(center) + list(axis) + list(ref_u) + [radius, height_min, height_max, phi_start, phi_sweep]
    record = {"type": "cylinder", "inner_wall": inner_wall, "params": params}
    wires = list(_face_wire_edges(face)) if wires is None else wires
    if _quadric_trim_fills_uv_box(face, (umin, umax, vmin, vmax), wires):
        return record, None  # trim is exactly the parametric rectangle: the scalar params suffice
    phi_of_u = (lambda u: u) if ax3.Direct() else (lambda u: -u)
    # affine (u, v) -> (phi[rad], h[cm]); OCC V is the height along the axis
    trim, reason = _quadric_trim_wire(face, lambda u, v: (phi_of_u(u), v * s), wires)
    if trim is None:
        return None, reason
    record["wires"] = trim
    return record, None


def extract_conical_face(face, scale_to_cm: float, wires=None) -> Tuple[Optional[dict], Optional[str]]:
    """Conical face -> a 'cone' surface record; r(v) = RefRadius + v sin(a), h(v) = v cos(a)."""
    adaptor = BRepAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Cone:
        return None, "not a cone"
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    cone = adaptor.Cone()
    ax3 = cone.Position()
    s = scale_to_cm
    half = cone.SemiAngle()
    ref_radius = cone.RefRadius()
    cos_a, sin_a = math.cos(half), math.sin(half)
    h_lo, r_lo = vmin * cos_a, ref_radius + vmin * sin_a
    h_hi, r_hi = vmax * cos_a, ref_radius + vmax * sin_a
    if h_lo > h_hi:
        h_lo, h_hi, r_lo, r_hi = h_hi, h_lo, r_hi, r_lo
    if r_lo < -_EXTRACT_TOL or r_hi < -_EXTRACT_TOL:
        return None, "conical trim produces a negative radius"
    r_lo, r_hi = max(0.0, r_lo), max(0.0, r_hi)
    if max(r_lo, r_hi) <= _EXTRACT_TOL:
        return None, "conical face has degenerate radii"
    if (h_hi - h_lo) * s <= _EXTRACT_TOL:
        return None, "conical face has a degenerate height range"
    center = _xyz(ax3.Location(), s)
    axis = _xyz(cone.Axis().Direction())
    ref_u = _xyz(ax3.XDirection())
    phi_start, phi_sweep = _quadric_phi_range(ax3, umin, umax)
    inner_wall = face.Orientation() == TopAbs_REVERSED
    params = (list(center) + list(axis) + list(ref_u) +
              [r_lo * s, r_hi * s, h_lo * s, h_hi * s, phi_start, phi_sweep])
    record = {"type": "cone", "inner_wall": inner_wall, "params": params}
    wires = list(_face_wire_edges(face)) if wires is None else wires
    if _quadric_trim_fills_uv_box(face, (umin, umax, vmin, vmax), wires):
        return record, None  # trim is exactly the parametric rectangle: the scalar params suffice
    # OCC V (ruling-line distance) maps to the C++ axial height h = v cos(alpha), in cm
    phi_of_u = (lambda u: u) if ax3.Direct() else (lambda u: -u)
    trim, reason = _quadric_trim_wire(face, lambda u, v: (phi_of_u(u), v * cos_a * s), wires)
    if trim is None:
        return None, reason
    record["wires"] = trim
    return record, None


def extract_spherical_face(face, scale_to_cm: float, wires=None) -> Tuple[Optional[dict], Optional[str]]:
    """Spherical face -> a 'sphere' surface record; the C++ polar angle is theta = pi/2 - v."""
    adaptor = BRepAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Sphere:
        return None, "not a sphere"
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    sph = adaptor.Sphere()
    ax3 = sph.Position()
    s = scale_to_cm
    center = _xyz(ax3.Location(), s)
    polar_axis = _xyz(ax3.Direction())
    ref_u = _xyz(ax3.XDirection())
    radius = sph.Radius() * s
    theta_min = 0.5 * math.pi - vmax
    theta_max = 0.5 * math.pi - vmin
    if theta_max - theta_min <= _EXTRACT_TOL:
        return None, "spherical face has a degenerate polar range"
    phi_start, phi_sweep = _quadric_phi_range(ax3, umin, umax)
    params = list(center) + list(polar_axis) + list(ref_u) + [radius, theta_min, theta_max, phi_start, phi_sweep]
    inner_wall = face.Orientation() == TopAbs_REVERSED
    record = {"type": "sphere", "inner_wall": inner_wall, "params": params}
    wires = list(_face_wire_edges(face)) if wires is None else wires
    if _quadric_trim_fills_uv_box(face, (umin, umax, vmin, vmax), wires):
        return record, None  # trim is exactly the parametric rectangle: the scalar params suffice
    # OCC V (latitude) maps to the C++ polar angle theta = pi/2 - v (rad)
    phi_of_u = (lambda u: u) if ax3.Direct() else (lambda u: -u)
    trim, reason = _quadric_trim_wire(face, lambda u, v: (phi_of_u(u), 0.5 * math.pi - v), wires)
    if trim is None:
        return None, reason
    record["wires"] = trim
    return record, None


def extract_toroidal_face(face, scale_to_cm: float, wires=None) -> Tuple[Optional[dict], Optional[str]]:
    """Toroidal face -> a 'torus' surface record.

    U is the ring phi (mirrored for a left-handed ax3) and V the tube phi.
    """
    adaptor = BRepAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Torus:
        return None, "not a torus"
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    tor = adaptor.Torus()
    ax3 = tor.Position()
    s = scale_to_cm
    major_radius = tor.MajorRadius() * s
    minor_radius = tor.MinorRadius() * s
    if minor_radius <= _EXTRACT_TOL or major_radius <= _EXTRACT_TOL:
        return None, "toroidal face has degenerate radii"
    center = _xyz(ax3.Location(), s)
    axis = _xyz(ax3.Direction())
    ref_u = _xyz(ax3.XDirection())
    phi_start, phi_sweep = _quadric_phi_range(ax3, umin, umax)
    two_pi = 2.0 * math.pi
    tube_sweep = vmax - vmin
    if tube_sweep <= 0.0:
        tube_sweep += two_pi
    tube_sweep = min(tube_sweep, two_pi)
    tube_start = vmin
    inner_wall = face.Orientation() == TopAbs_REVERSED
    params = (list(center) + list(axis) + list(ref_u) +
              [major_radius, minor_radius, phi_start, phi_sweep, tube_start, tube_sweep])
    record = {"type": "torus", "inner_wall": inner_wall, "params": params}
    wires = list(_face_wire_edges(face)) if wires is None else wires
    if _quadric_trim_fills_uv_box(face, (umin, umax, vmin, vmax), wires):
        return record, None  # trim is exactly the parametric rectangle: the scalar params suffice
    # affine (u, v) -> (phiRing[rad], phiTube[rad]); OCC V is the tube angle, unchanged by the frame
    phi_of_u = (lambda u: u) if ax3.Direct() else (lambda u: -u)
    trim, reason = _quadric_trim_wire(face, lambda u, v: (phi_of_u(u), v), wires)
    if trim is None:
        return None, reason
    record["wires"] = trim
    return record, None


# -------------------------------
# Canonical-form recognition: recover the exact analytic model behind a stored NURBS
# -------------------------------
# Model selection, not fitting: only a machine-precision fit is accepted, so an almost-cylinder
# stays free-form. Used for faces whose stored type has no direct extractor.

_RECOGNIZE_TOL_EXACT = 1.e-9


def _recognize_analytic_surface(adaptor, uv_bounds) -> Optional[dict]:
    """The exact plane/sphere/cylinder/cone behind a face, or None; lengths in native CAD units.

    Proposals are scored by their measured gap over the sample diagonal only; the fewest-parameter
    model below _RECOGNIZE_TOL_EXACT wins."""
    umin, umax, vmin, vmax = uv_bounds
    P, N = _sample_surface_for_recognition(adaptor, umin, umax, vmin, vmax)
    if P is None:
        return None
    scale = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    if scale < 1e-12:
        return None

    def score(kind, model):
        """The one criterion: the achieved gap, relative to the patch's own size."""
        try:
            gap = _analytic_surface_gap(kind, model, P)
        except (ValueError, FloatingPointError):
            return float("inf")
        return gap / scale if math.isfinite(gap) else float("inf")

    best = ("freeform", float("inf"), {})
    for kind, model in _analytic_surface_proposals(P, N):
        res = score(kind, model)
        if kind == "plane":
            if res < _RECOGNIZE_TOL_EXACT:
                # Parsimony: an exact plane wins outright.
                out = {"kind": "plane", "residual": res, "P": P, "N": N}
                out.update(model)
                out["gap"] = res * scale
                out["gap_relative"] = res
                return out
            continue
        if res < best[1]:
            best = (kind, res, model)

    kind, res, extra = best
    if res >= _RECOGNIZE_TOL_EXACT:
        return None
    out = {"kind": kind, "residual": res, "P": P, "N": N}
    out.update(extra)
    out["gap"] = res * scale
    out["gap_relative"] = res
    return out


# -------------------------------
# Self-test for the recognition path (`--self-test`)
# -------------------------------
# Every positive control has a negative one; all are built in-process from OCC primitives.

class _Checks:
    """A self-test block's counters, and its one printed line per check."""

    def __init__(self):
        self.checks = 0
        self.failures = 0

    def report(self, ok: bool, label: str, detail: str = ""):
        self.checks += 1
        if not ok:
            self.failures += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")


def _self_test_faces_of(shape) -> List[object]:
    out = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        out.append(topods.Face(explorer.Current()))
        explorer.Next()
    return out


def run_recognition_self_test() -> int:
    """Assert the canonical-form recognizer against models whose answer is known in closed form.

    Returns the number of failures; prints one line per check.
    """
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_MakeFace
    from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder,
                                      BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus)
    from OCC.Core.gp import (gp_Ax2, gp_Ax3, gp_Cone, gp_Cylinder, gp_Dir, gp_Pln, gp_Sphere)

    tally = _Checks()
    report = tally.report
    accepted_gaps = []

    def recognize(face):
        adaptor = BRepAdaptor_Surface(face)
        try:
            uv_bounds = breptools.UVBounds(face)
        except Exception:
            return None
        rec = _recognize_analytic_surface(adaptor, uv_bounds)
        if rec is not None:
            accepted_gaps.append((rec["kind"], rec["gap_relative"]))
        return rec

    def nurbs(shape):
        return BRepBuilderAPI_NurbsConvert(shape, True).Shape()

    def expect(face, want: Optional[str], label: str):
        rec = recognize(face)
        got = rec["kind"] if rec else None
        detail = (f"got {got}" if rec is None or want is None else
                  f"got {got}, gap {rec['gap_relative']:.2e} of the patch diagonal")
        if rec is not None and want is not None and got == want:
            detail = f"gap {rec['gap_relative']:.2e} of the patch diagonal"
        report(got == want, label, detail)
        return rec

    print("Canonical-form recognition self-test")
    print(" positive controls: a quadric written as NURBS must be recovered")
    # BRepBuilderAPI_NurbsConvert turns each analytic face into the rational B-spline a CAD
    # exporter would have written -- the exporter artefact this whole path exists for, built here.
    frame = gp_Ax3(gp_Pnt(1.0, -2.0, 3.0), gp_Dir(0.3, 0.4, 0.866), gp_Dir(0.866, 0.0, -0.3))
    for label, surface, want in (
            ("cylinder", gp_Cylinder(frame, 5.0), "cylinder"),
            ("cone", gp_Cone(frame, 0.4, 2.0), "cone"),
            ("sphere", gp_Sphere(frame, 7.0), "sphere"),
            ("plane", gp_Pln(frame), "plane")):
        if label == "plane":
            native = BRepBuilderAPI_MakeFace(surface, -5.0, 5.0, -3.0, 3.0).Shape()
        elif label == "sphere":
            native = BRepBuilderAPI_MakeFace(surface, 0.2, 2.4, -0.9, 0.9).Shape()
        else:
            native = BRepBuilderAPI_MakeFace(surface, 0.2, 2.4, 1.0, 9.0).Shape()
        faces = _self_test_faces_of(nurbs(native))
        report(len(faces) == 1, f"NURBS-converted {label} patch is one face", f"{len(faces)} found")
        if faces:
            expect(faces[0], want, f"NURBS-encoded {label} is recognized as a {want}")

    print(" negative controls: a genuinely free-form surface must be declined")
    expect(_self_test_bezier_patch(
        lambda s, t: (10 * s - 5, 10 * t - 5, (10 * s - 5) * (10 * t - 5) / 10.0), 6, 6),
        None, "free-form saddle is not recognized as any quadric")
    expect(_self_test_bezier_patch(
        lambda s, t: (20 * s - 10, 0.5 * t, 0.02 * (20 * s - 10) ** 2 + 0.3 * (20 * s - 10) * t), 6, 6),
        None, "narrow free-form ridge is not recognized as any quadric")
    for face in _self_test_faces_of(
            nurbs(BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 10.0, 1.0).Shape())):
        expect(face, None, "NURBS-encoded torus is declined (no torus model -- known limitation)")

    print(" the ALICE3 cone over-acceptance: a swept non-circular profile")
    for bulge in (1.0e-3, 1.0e-2):
        for taper in (1.0e-4, 1.0e-6, 1.0e-8):
            expect(_self_test_tapered_near_circle(bulge, taper), None,
                   f"swept non-circular profile (bulge {bulge:.0e}, taper {taper:.0e}) is declined")

    print(" the invariant: every accepted recognition is within the declared tolerance")
    worst = max(accepted_gaps, key=lambda kv: kv[1], default=("-", 0.0))
    report(all(gap < _RECOGNIZE_TOL_EXACT for _kind, gap in accepted_gaps),
           "every accepted face's MEASURED gap is below the acceptance tolerance",
           f"worst {worst[0]} at {worst[1]:.2e} against {_RECOGNIZE_TOL_EXACT:.0e}")

    print(f"\n{tally.checks} checks, {tally.failures} failure(s)")
    return tally.failures


def run_placement_self_test() -> int:
    """Assert the placed-primitive emission and the COMPOSITION ORDER in `geom.C`.

    Points are classified by navigating the assembly and compared with OCCT in the part frame;
    three negative controls (transposed rotation, reversed product, dropped placement) must move
    the count. Returns the number of failures; needs PyROOT and pythonOCC.
    """
    tally = _Checks()
    report = tally.report

    print("\nPlaced-primitive emission and geom.C composition order")
    try:
        import ROOT
    except Exception as exc:                                         # noqa: BLE001
        print(f"  [FAIL] PyROOT is not importable in this interpreter ({exc}); the placement "
              "checks cannot run. Use the O2 environment.")
        print("\n1 checks, 1 failure(s)")
        return 1
    ROOT.gROOT.SetBatch(True)

    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
    from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

    from cadsupport import emit as csg_emit, primitives as prim          # noqa: E402

    # --- the specimen: a tube SEGMENT, rotated and translated off every coordinate axis --------
    axis = gp_Ax2(gp_Pnt(0, 0, -5), gp_Dir(0, 0, 1))
    wedge = BRepPrimAPI_MakeCylinder(axis, 2.0, 10.0, math.radians(75.0)).Shape()
    bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -6), gp_Dir(0, 0, 1)), 1.0, 12.0).Shape()
    seg = BRepAlgoAPI_Cut(wedge, bore).Shape()
    spin = gp_Trsf()
    spin.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 2, 3)), 0.9)
    shift = gp_Trsf()
    shift.SetTranslation(gp_Vec(3.0, -4.0, 5.0))
    placed = BRepBuilderAPI_Transform(seg, shift.Multiplied(spin), True).Shape()

    record = csg_emit.process_solid(placed, "selftest-placed-tubeseg")
    if not record["accepted"]:
        report(False, "a rotated, translated tube segment is recognised and accepted",
               f"{record['reason']}")
        print(f"\n{tally.checks} checks, {tally.failures} failure(s)")
        return tally.failures
    report(True, "a rotated, translated tube segment is recognised and accepted",
           f"{record['recogniser']}: {record['description']}")

    shape, placement = prim.build_root(record["candidate"], "selftest")
    report(shape.ClassName() == "TGeoTubeSeg" and placement is not None,
           "it emits a TGeoTubeSeg with a placement, not a TGeoCompositeShape",
           f"{shape.ClassName()}, placement {'present' if placement else 'absent'}")

    # --- the win: an analytic Capacity() again, checked against OCCT's own volume -------------
    props = GProp_GProps()
    brepgprop.VolumeProperties(placed, props)
    occ_volume = props.Mass()
    rel = abs(shape.Capacity() - occ_volume) / occ_volume
    report(rel < 1.0e-12, "its Capacity() is analytic and agrees with the OCCT volume",
           f"ROOT {shape.Capacity():.12g} vs OCCT {occ_volume:.12g}, rel {rel:.2e}")
    # ... and the same comparison must fail on a shape that is 1% too fat.
    fat = dict(record["candidate"]["leaves"][0]["params"])
    fat["rmax"] *= 1.01
    fat_cand = prim.candidate("primitive", [prim.leaf(
        "TGeoTubeSeg", fat, record["candidate"]["leaves"][0]["frame"])], "selftest-negative")
    fat_shape, _ = prim.build_root(fat_cand, "selftest_fat")
    rel_fat = abs(fat_shape.Capacity() - occ_volume) / occ_volume
    report(rel_fat > 1.0e-3, "the same capacity comparison does reject a 1% wrong radius",
           f"rel {rel_fat:.2e}")

    # --- the composition order, decided by navigation ------------------------------------------
    # A deliberately non-symmetric part placement, as emit_placement_cpp writes for an AddNode.
    part_rot = ROOT.TGeoRotation("selftest_partrot", 37.0, 24.0, 61.0)
    ROOT.SetOwnership(part_rot, False)
    part_placement = ROOT.TGeoCombiTrans(-2.0, 7.0, 1.5, part_rot)
    ROOT.SetOwnership(part_placement, False)
    shape_placement = prim.root_placement_matrix(placement, "selftest_shapeplace")

    def compose(order):
        """The node matrix under a given composition rule."""
        if order == "part*shape":
            m = ROOT.TGeoHMatrix(part_placement)
            m.Multiply(shape_placement)
        elif order == "shape*part":
            m = ROOT.TGeoHMatrix(shape_placement)
            m.Multiply(part_placement)
        elif order == "part*shapeT":
            t = [[placement[r][c] for r in range(3)] + [placement[c][3]] for c in range(3)]
            m = ROOT.TGeoHMatrix(part_placement)
            m.Multiply(prim.root_placement_matrix(t, "selftest_shapeplaceT"))
        else:  # the placement dropped on the floor -- the bug this test is really for
            m = ROOT.TGeoHMatrix(part_placement)
        return m

    # Probes in the assembly frame, with OCCT's verdict after undoing the part placement only;
    # drawn over the padded part box, so about a third are inside.
    classifier = BRepClass3d_SolidClassifier(placed)
    tolerance = max(csg_emit.model_tolerance_cm(placed), 1.0e-9)
    bnd = Bnd_Box()
    brepbndlib.Add(placed, bnd)
    bnd.SetGap(0.0)
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = bnd.Get()
    bpad = 0.1 * max(bxmax - bxmin, bymax - bymin, bzmax - bzmin)
    rng = random.Random(4242)
    probes = []
    master = array("d", [0.0, 0.0, 0.0])
    for _ in range(3000):
        part_point = (rng.uniform(bxmin - bpad, bxmax + bpad),
                      rng.uniform(bymin - bpad, bymax + bpad),
                      rng.uniform(bzmin - bpad, bzmax + bpad))
        classifier.Perform(gp_Pnt(*part_point), tolerance)
        state = classifier.State()
        if state == TopAbs_ON:
            continue
        part_placement.LocalToMaster(array("d", list(part_point)), master)
        probes.append(((master[0], master[1], master[2]), state == TopAbs_IN))
    n_inside = sum(1 for _p, inside in probes if inside)
    print(f"        ({len(probes)} probes, {n_inside} of them inside the CAD body)")

    def _keep(obj):
        """Everything below is registered with the TGeoManager, which frees it. Handing ownership
        to Python as well is a double free -- the same rule cadsupport/primitives.py follows."""
        ROOT.SetOwnership(obj, False)
        return obj

    _keep(shape)
    _keep(fat_shape)

    def disagreements(order):
        # A fresh manager per variant. Constructing one DELETES the previous geometry, which is
        # why nothing created here may be owned by Python as well.
        manager = _keep(ROOT.TGeoManager(f"selftest_{order}", "placement composition self-test"))
        vacuum = _keep(ROOT.TGeoMaterial("Vacuum", 0., 0., 0.))
        medium = _keep(ROOT.TGeoMedium("Vacuum", 1, vacuum))
        world = _keep(ROOT.TGeoVolume("TOP", _keep(ROOT.TGeoBBox("selftestWorld", 40., 40., 40.)),
                                      medium))
        # A fresh copy of the shape per manager, for the same reason.
        local_shape, _ = prim.build_root(record["candidate"], f"selftest_{order}_shape")
        part = _keep(ROOT.TGeoVolume("PART", _keep(local_shape), medium))
        world.AddNode(part, 1, _keep(compose(order)))
        manager.SetTopVolume(world)
        manager.CloseGeometry()
        bad = 0
        for p, want in probes:
            node = manager.FindNode(p[0], p[1], p[2])
            inside = node is not None and node.GetVolume().GetName() == "PART"
            if inside != want:
                bad += 1
        return bad, len(probes)

    bad_ok, scored = disagreements("part*shape")
    report(bad_ok == 0 and scored > 500,
           "geom.C's node matrix partPlacement * shapePlacement puts the solid where the CAD "
           "body is", f"{bad_ok} disagreement(s) over {scored} navigated points")
    for order, label in (("shape*part", "the reversed product"),
                         ("part*shapeT", "a transposed shape rotation"),
                         ("part-only", "dropping the shape placement")):
        bad, _n = disagreements(order)
        report(bad > 0, f"{label} does move the count", f"{bad} disagreement(s)")

    print(f"\n{tally.checks} checks, {tally.failures} failure(s)")
    return tally.failures


# -------------------------------
# Self-test for the planar trim vocabulary (`--self-test`, third block)
# -------------------------------
# An oblique plane cuts a cylinder on an ellipse, stored exactly as a rational B-spline; the
# deviation is measured both ways, and the instrument must be able to report a large one.

def _self_test_oblique_cut_cylinder(radius: float = 1.2, height: float = 5.0,
                                    tilt_deg: float = 60.0, lift: float = 2.5):
    """The `oblique_cut_cyl` ladder fixture, built in-process: a cylinder cut by a plane inclined
    to its axis. Returns the solid. Everything is already in cm (scale_to_cm = 1)."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Trsf, gp_Vec

    cyl = BRepPrimAPI_MakeCylinder(radius, height).Shape()
    knife = BRepPrimAPI_MakeBox(gp_Pnt(-20.0, -20.0, 0.0), 40.0, 40.0, 40.0).Shape()
    rot = gp_Trsf()
    rot.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.radians(tilt_deg))
    move = gp_Trsf()
    move.SetTranslation(gp_Vec(0.0, 0.0, lift))
    knife = BRepBuilderAPI_Transform(knife, move * rot, True).Shape()
    return BRepAlgoAPI_Cut(cyl, knife).Shape()


def _self_test_conic_bounded_plane(conic, t0: float, t1: float):
    """A planar face bounded by one conic arc from `t0` to `t1` plus the chord closing it."""
    from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                         BRepBuilderAPI_MakeWire)
    arc = BRepBuilderAPI_MakeEdge(conic, t0, t1).Edge()
    chord = BRepBuilderAPI_MakeEdge(conic.Value(t1), conic.Value(t0)).Edge()
    wire = BRepBuilderAPI_MakeWire(arc, chord).Wire()
    return BRepBuilderAPI_MakeFace(wire, True).Face()


def _self_test_rebuild_2d_curve(seg):
    """Rebuild a sidecar wire segment as an OCC `Geom2d_Curve`, so the curve the *sidecar* carries
    can be measured against the CAD edge instead of being argued about."""
    from OCC.Core.Geom2d import Geom2d_BSplineCurve
    from OCC.Core.gp import gp_Pnt2d
    from OCC.Core.TColgp import TColgp_Array1OfPnt2d
    from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger

    if seg["curve"] != "bspline":
        return None
    p = seg["params"]
    degree, n_poles = int(p[0]), int(p[1])
    poles = TColgp_Array1OfPnt2d(1, n_poles)
    for i in range(n_poles):
        poles.SetValue(i + 1, gp_Pnt2d(p[2 + 2 * i], p[3 + 2 * i]))
    weights = TColStd_Array1OfReal(1, n_poles)
    for i in range(n_poles):
        weights.SetValue(i + 1, p[2 + 2 * n_poles + i])
    flat = p[2 + 3 * n_poles:]
    distinct = []
    for k in flat:
        if not distinct or abs(k - distinct[-1][0]) > 1e-12:
            distinct.append([k, 1])
        else:
            distinct[-1][1] += 1
    knots = TColStd_Array1OfReal(1, len(distinct))
    mults = TColStd_Array1OfInteger(1, len(distinct))
    for i, (k, m) in enumerate(distinct):
        knots.SetValue(i + 1, k)
        mults.SetValue(i + 1, m)
    return Geom2d_BSplineCurve(poles, weights, knots, mults, degree)


def _self_test_trim_deviation(face, record, scale_to_cm: float = 1.0, n: int = 257):
    """Largest distance, in cm, between the CAD and the stored boundary curves, measured both ways.

    Returns (max_deviation_cm, patch_diagonal_cm) or (None, None) if a segment cannot be rebuilt."""
    from OCC.Core.Geom2dAPI import Geom2dAPI_ProjectPointOnCurve
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCC.Core.gp import gp_Pnt2d

    origin_cm = record["params"][0:3]
    axis_u = record["params"][3:6]
    axis_v = record["params"][6:9]
    project = _planar_projector(origin_cm, axis_u, axis_v, scale_to_cm)

    def unproject(u, v):
        return gp_Pnt(*[origin_cm[i] + u * axis_u[i] + v * axis_v[i] for i in range(3)])

    def distance_to(proj, endpoints, point):
        """Distance from `point` to a curve, falling back to the endpoints (an upper bound)."""
        best = min(point.Distance(e) for e in endpoints)
        if proj.NbPoints() > 0:
            best = min(best, proj.LowerDistance())
        return best

    segs = [s for w in record["wires"] for s in w["edges"]]
    edges = [e for _w, _o, es in _face_wire_edges(face) for e, _v in es]
    if len(segs) != len(edges):
        return None, None
    worst = 0.0
    points = []
    for seg, edge in zip(segs, edges):
        curve3d, first, last = BRep_Tool.Curve(edge)
        if curve3d is None:
            return None, None
        lo, hi = (first, last) if first <= last else (last, first)
        cad = [curve3d.Value(float(t)) for t in np.linspace(lo, hi, n)]
        points.extend([(p.X() * scale_to_cm, p.Y() * scale_to_cm, p.Z() * scale_to_cm) for p in cad])
        if seg["curve"] == "line":
            u0, v0, u1, v1 = seg["params"]
            for p in cad:
                u, v = project(p)
                du, dv = u - u0, v - v0
                lu, lv = u1 - u0, v1 - v0
                l2 = lu * lu + lv * lv
                t = 0.0 if l2 <= 0.0 else min(1.0, max(0.0, (du * lu + dv * lv) / l2))
                worst = max(worst, math.hypot(du - t * lu, dv - t * lv))
            stored = [unproject(u0 + (u1 - u0) * t, v0 + (v1 - v0) * t)
                      for t in np.linspace(0.0, 1.0, n)]
        elif seg["curve"] == "arc":
            cu, cv, r, a0, sweep = seg["params"]
            stored = [unproject(cu + r * math.cos(a0 + sweep * t), cv + r * math.sin(a0 + sweep * t))
                      for t in np.linspace(0.0, 1.0, n)]
        else:
            curve2d = _self_test_rebuild_2d_curve(seg)
            if curve2d is None:
                return None, None
            t0, t1 = curve2d.FirstParameter(), curve2d.LastParameter()
            stored = []
            for t in np.linspace(t0, t1, n):
                q = curve2d.Value(float(t))
                stored.append(unproject(q.X(), q.Y()))
            ends2d = [curve2d.Value(t0), curve2d.Value(t1)]
            for p in cad:
                u, v = project(p)
                here = gp_Pnt2d(u, v)
                worst = max(worst, distance_to(Geom2dAPI_ProjectPointOnCurve(here, curve2d),
                                               ends2d, here))
        # the reverse direction: every stored sample back onto the CAD 3D curve
        ends3d = [curve3d.Value(float(lo)), curve3d.Value(float(hi))]
        for q in stored:
            here = gp_Pnt(q.X() / scale_to_cm, q.Y() / scale_to_cm, q.Z() / scale_to_cm)
            worst = max(worst, scale_to_cm *
                        distance_to(GeomAPI_ProjectPointOnCurve(here, curve3d), ends3d, here))
    arr = np.asarray(points)
    diagonal = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0))) if len(arr) else 0.0
    return worst, diagonal


def run_planar_trim_self_test() -> int:
    """Assert the planar face's trim-curve vocabulary: an ellipse boundary is carried EXACTLY, and
    a boundary that is not a conic we can write exactly is still declined."""
    from OCC.Core.Geom import Geom_Ellipse, Geom_Hyperbola, Geom_Parabola
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Elips, gp_Hypr, gp_Parab

    tally = _Checks()
    report = tally.report

    print("\nPlanar trim vocabulary: the ellipse boundary")

    solid = _self_test_oblique_cut_cylinder()
    cut_face = None
    for face in _self_test_faces_of(solid):
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() != GeomAbs_Plane:
            continue
        kinds = set()
        for _w, _o, es in _face_wire_edges(face):
            for e, _v in es:
                kinds.add(CURVE_TYPE_NAME.get(BRepAdaptor_Curve(e).GetType(), "unknown"))
        if "ellipse" in kinds:
            cut_face = face
    report(cut_face is not None,
           "the oblique cut of a cylinder really does produce an ellipse-bounded planar face",
           "found" if cut_face is not None else "no ellipse boundary edge -- fixture is wrong")

    if cut_face is not None:
        record, reason = extract_planar_face(cut_face, 1.0)
        report(record is not None, "an oblique planar cut of a cylinder is accepted",
               "accepted" if record else f"declined: {reason}")
        if record is not None:
            segs = [s for w in record["wires"] for s in w["edges"]]
            kinds = sorted({s["curve"] for s in segs})
            report(kinds == ["bspline"],
                   "the ellipse is stored as a B-spline segment", f"segments: {kinds}")
            spreads = []
            for s in segs:
                if s["curve"] != "bspline":
                    continue
                n_poles = int(s["params"][1])
                w = s["params"][2 + 2 * n_poles: 2 + 3 * n_poles]
                spreads.append(max(w) - min(w))
            rational = any(spread > 1e-12 for spread in spreads)
            report(rational,
                   "it is a RATIONAL B-spline -- the exact conic form, not a polynomial fit",
                   f"weight spread {max(spreads):.3f}" if spreads else "no bspline segment")
            dev, diag = _self_test_trim_deviation(cut_face, record)
            report(dev is not None and dev < 1.0e-9,
                   "the stored trim reproduces the CAD boundary at machine precision",
                   f"max deviation {dev:.2e} cm = {dev / diag:.2e} patch diagonals"
                   if dev is not None else "could not be measured")

    # A partial ellipse arc: the ExcavatorArm/Bucket shape of the problem, not the fixture's closed one.
    frame = gp_Ax2(gp_Pnt(0.3, -0.2, 1.1), gp_Dir(0.3, 0.4, 0.866), gp_Dir(0.866, 0.0, -0.3))
    ell_face = _self_test_conic_bounded_plane(Geom_Ellipse(gp_Elips(frame, 2.4, 1.2)), 0.35, 2.6)
    record, reason = extract_planar_face(ell_face, 1.0)
    report(record is not None, "an ellipse ARC boundary (the Bucket case) is accepted",
           "accepted" if record else f"declined: {reason}")
    if record is not None:
        dev, diag = _self_test_trim_deviation(ell_face, record)
        report(dev is not None and dev < 1.0e-9,
               "the ellipse arc's stored trim reproduces the CAD boundary at machine precision",
               f"max deviation {dev:.2e} cm = {dev / diag:.2e} patch diagonals"
               if dev is not None else "could not be measured")

    print(" the deviation instrument must be able to return a large number")
    if record is not None:
        # A circular arc with the ellipse's endpoints and centre: the instrument must see it.
        import copy
        wrong = copy.deepcopy(record)
        for w in wrong["wires"]:
            for s in w["edges"]:
                if s["curve"] == "bspline":
                    n_poles = int(s["params"][1])
                    for i in range(n_poles):
                        s["params"][2 + 2 * i] *= 0.5   # squash the major axis: a different conic
        bad_dev, bad_diag = _self_test_trim_deviation(ell_face, wrong)
        report(bad_dev is not None and bad_dev > 1.0e-3,
               "a deliberately wrong conic is caught by the same measurement",
               f"max deviation {bad_dev:.2e} cm = {bad_dev / bad_diag:.2e} patch diagonals"
               if bad_dev is not None else "could not be measured")

    print(" negative controls: a boundary that is not an exactly-writable conic is still declined")
    hyp_face = _self_test_conic_bounded_plane(Geom_Hyperbola(gp_Hypr(frame, 2.0, 1.0)), 0.2, 0.9)
    record, reason = extract_planar_face(hyp_face, 1.0)
    report(record is None and reason is not None and "hyperbola" in reason,
           "a hyperbola boundary edge is declined", reason if record is None else "ACCEPTED")
    par_face = _self_test_conic_bounded_plane(Geom_Parabola(gp_Parab(frame, 1.5)), -1.4, 1.4)
    record, reason = extract_planar_face(par_face, 1.0)
    report(record is None and reason is not None and "parabola" in reason,
           "a parabola boundary edge is declined", reason if record is None else "ACCEPTED")

    print(f"\n{tally.checks} checks, {tally.failures} failure(s)")
    return tally.failures


# -------------------------------
# Self-test for coincident placements (`--self-test`, fourth block)
# -------------------------------

def _self_test_shape_tool():
    """A fresh, empty in-memory XCAF document and its shape tool, for pathological fixtures."""
    doc = TDocStd_Document("selftest-placements")
    return doc, XCAFDoc_DocumentTool.ShapeTool(doc.Main())


def _self_test_shift(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> gp_Trsf:
    trsf = gp_Trsf()
    if (dx, dy, dz) != (0.0, 0.0, 0.0):
        trsf.SetTranslation(gp_Vec(dx, dy, dz))
    return trsf


def _self_test_leaf(shape_tool, side: float):
    return shape_tool.AddShape(BRepPrimAPI_MakeBox(side, side, side).Shape(), False)


def _self_test_assembly(shape_tool, components):
    """`components` is a sequence of (child label, gp_Trsf)."""
    label = shape_tool.NewShape()
    for child, trsf in components:
        shape_tool.AddComponent(label, child, TopLoc_Location(trsf))
    return label


def _self_test_convert(shape_tool):
    """Run the production traversal over an in-memory assembly and report what it placed.

    Returns (report, leaf occurrences), where the occurrences are (definition, world transform
    signature) pairs -- measured by walking the emitted graph, not read back out of the rule.
    """
    reset_graph()
    report = expand_free_shapes(shape_tool, meshparam=None, scale_to_cm=1.0)
    leaves = [occ for occ in enumerate_occurrences(placements, top_defs)
              if occ[0] in logical_volumes]
    return report, leaves


def run_duplicate_placement_self_test() -> int:
    """Assert that one definition at one world transform is placed exactly ONCE, and that one
    definition at two different world transforms is still placed twice (the negative control).

    Returns the number of failures; prints one line per check.
    """
    tally = _Checks()
    report = tally.report

    print("\nCoincident placements: one definition, one world transform, one placement")

    # --- 1. the ALICE3 shape: a root whose FIRST child contains its own siblings ---------------
    _doc, st = _self_test_shape_tool()
    leaves = [_self_test_leaf(st, 1.0 + i) for i in range(3)]
    subs = [_self_test_assembly(st, [(leaf, _self_test_shift(dx=10.0 * i))])
            for i, leaf in enumerate(leaves)]
    detector = _self_test_assembly(st, [(sub, gp_Trsf()) for sub in subs])
    # The root lists the detector AND, at the identity beside it, the detector's own three
    # children -- entity for entity what CAD_noETA.stp's root does.
    _self_test_assembly(st, [(detector, gp_Trsf())] + [(sub, gp_Trsf()) for sub in subs])
    st.UpdateAssemblies()
    rep, occ = _self_test_convert(st)

    report(rep["declared_leaf_placements"] == 6 and rep["declared_multiplicity"] == {2: 3},
           "the fixture really does declare the ALICE3 defect: 3 solids, each declared twice at "
           "the same place",
           f"{rep['declared_leaf_placements']} declared, multiplicity "
           f"{rep['declared_multiplicity']}")
    report(len(occ) == 3, "it converts to 3 leaf placements, not 6", f"{len(occ)} placed")
    report(len(set(occ)) == 3 and len(occ) == len(set(occ)),
           "and no two of them share a definition and a world transform",
           f"{len(set(occ))} distinct (definition, world transform) pair(s)")
    report(rep["n_suppressed_by_rule"]["root-containment"] == 3,
           "the root-containment rule is what fires, and it drops exactly the 3 root edges",
           f"{rep['n_suppressed_by_rule']}")

    # --- 2. THE negative control: legitimate instancing at two DIFFERENT transforms ------------
    # A rule keyed on the definition alone would fail here.
    _doc, st = _self_test_shape_tool()
    leaf = _self_test_leaf(st, 2.0)
    module = _self_test_assembly(st, [(leaf, gp_Trsf())])
    _self_test_assembly(st, [(module, _self_test_shift(dx=0.0)),
                             (module, _self_test_shift(dx=100.0))])
    st.UpdateAssemblies()
    rep, occ = _self_test_convert(st)
    report(len(occ) == 2, "one sub-assembly instanced twice at DIFFERENT transforms still gets "
                          "two placements", f"{len(occ)} placed")
    report(len(set(sig for _lid, sig in occ)) == 2,
           "and they are at two different world transforms, as the CAD says",
           f"{len(set(sig for _lid, sig in occ))} distinct world transform(s)")
    report(sum(rep["n_suppressed_by_rule"].values()) == 0,
           "nothing is suppressed there", f"{rep['n_suppressed_by_rule']}")

    # ... and the same model with a THIRD, coincident instance bolted on must lose exactly one.
    _doc, st = _self_test_shape_tool()
    leaf = _self_test_leaf(st, 2.0)
    module = _self_test_assembly(st, [(leaf, gp_Trsf())])
    _self_test_assembly(st, [(module, _self_test_shift(dx=0.0)),
                             (module, _self_test_shift(dx=100.0)),
                             (module, _self_test_shift(dx=100.0))])
    st.UpdateAssemblies()
    rep, occ = _self_test_convert(st)
    report(len(occ) == 2 and len(set(occ)) == 2 and rep["declared_leaf_placements"] == 3,
           "a third instance that coincides with the second is the one that goes",
           f"{rep['declared_leaf_placements']} declared -> {len(occ)} placed at "
           f"{len(set(occ))} distinct transform(s)")

    # --- 3. the same definition at the same transform down two different assembly paths --------
    _doc, st = _self_test_shape_tool()
    shared_leaf = _self_test_leaf(st, 3.0)
    own_left, own_right = _self_test_leaf(st, 4.0), _self_test_leaf(st, 5.0)
    shared = _self_test_assembly(st, [(shared_leaf, gp_Trsf())])
    at = _self_test_shift(dz=7.0)
    left = _self_test_assembly(st, [(shared, at), (own_left, _self_test_shift(dx=20.0))])
    right = _self_test_assembly(st, [(shared, at), (own_right, _self_test_shift(dx=40.0))])
    _self_test_assembly(st, [(left, gp_Trsf()), (right, gp_Trsf())])
    st.UpdateAssemblies()
    rep, occ = _self_test_convert(st)
    report(rep["declared_leaf_placements"] == 4 and len(occ) == 3,
           "the same sub-assembly at the same transform down two assembly paths is placed once",
           f"{rep['declared_leaf_placements']} declared -> {len(occ)} placed")
    report(len(set(occ)) == len(occ),
           "and the two paths' own, distinct parts both survive",
           f"{len(set(occ))} distinct (definition, world transform) pair(s)")
    report(rep["n_suppressed_by_rule"]["coincident-occurrence"] == 1
           and rep["n_suppressed_by_rule"]["root-containment"] == 0,
           "here it is the defensive rule that fires, not the structural one",
           f"{rep['n_suppressed_by_rule']}")

    # --- 4. the invariant on a real corpus, not only on a fixture ------------------------------
    # ExcavatorArm must never move: 13 solids, 13 distinct signatures.
    excavator_arm = _Path(__file__).resolve().parent.parent / "examples" / "ExcavatorArm.step"
    if not excavator_arm.exists():
        report(False, "the count invariant holds on a real corpus (ExcavatorArm.step)",
               f"missing corpus: {excavator_arm}")
    else:
        extract_graph(str(excavator_arm), meshparam=None, scale_to_cm=0.1)
        occ = [o for o in enumerate_occurrences(placements, top_defs) if o[0] in logical_volumes]
        report(len(occ) == 13 and len(set(occ)) == 13,
               "the count invariant holds on a real corpus: ExcavatorArm.step has 13 placed solids in "
               "and 13 out", f"{len(occ)} placed, {len(set(occ))} distinct")

    print(f"\n{tally.checks} checks, {tally.failures} failure(s)")
    return tally.failures


def run_in_field_media_self_test() -> int:
    """Assert what `--in-field` writes, and that without it no SetParam is written.

    Returns the number of failures; prints one line per check.
    """
    tally = _Checks()
    report = tally.report

    print("\nMedium parameters under --in-field")

    mat = ResolvedMaterial(
        bom_name="Silicon", nist_name="G4_Si", score=1.0, note="self-test",
        rho_used_g_cm3=2.33,
        elements=[{"symbol": "Si", "Z": 14, "A_g_mol": 28.0853614555, "mass_fraction": 1.0}],
        radlen_cm=9.3660702922, intlen_cm=45.6603073704)
    used = {"Silicon": mat}

    off, _ = emit_materials_cpp(used, in_field=None)
    report("SetParam" not in off,
           "without --in-field no SetParam is written (the negative control)",
           "" if "SetParam" not in off else "emitter changed behaviour for existing modules")

    on, _ = emit_materials_cpp(used, in_field=(2.0, 10.0))
    report("cadFieldTrackingParams(cad_ifield, cad_fieldm);" in on,
           "--in-field queries the LIVE field instead of asserting a pair")
    report("med_Silicon->SetParam(1, cad_ifield);" in on,
           "ifield is the queried variable, not a literal")
    report("med_Silicon->SetParam(2, cad_fieldm);" in on,
           "fieldm is the queried variable, not a literal")
    report("int   cad_ifield = 2;" in on and "float cad_fieldm = 10;" in on,
           "the seed is only what applies when no field is loaded")
    report("med_Default->SetParam(1, cad_ifield);" in on,
           "the Default medium is not left field-free either")
    for slot, key in enumerate(MEDIUM_PARAM_ORDER):
        if key in ("ifield", "fieldm"):
            continue
        report(f"med_Silicon->SetParam({slot}, 0);" in on,
               f"step control {key} stays 0 (the transport default)")
    report(on.count("SetParam") == 2 * len(MEDIUM_PARAM_ORDER),
           "all eight parameters are written for each of the two media",
           f"found {on.count('SetParam')}")
    _pre_on, _pre_off = emit_cpp_prelude(in_field=True), emit_cpp_prelude(in_field=False)
    report('#include "Field/MagneticField.h"' in _pre_on and "#include \"TVirtualMC.h\"" in _pre_on,
           "the prelude pulls the two headers Cling parses standalone")
    report("static void cadFieldTrackingParams(int& mode, float& maxfield)" in _pre_on,
           "and defines the query helper")
    report("DetectorsBase/Detector.h" not in _pre_on,
           "and NOT Detector.h, whose FairDetector payload segfaults a bare root -l session")
    report("cadFieldTrackingParams" not in _pre_off,
           "none of it appears without --in-field (the negative control)")

    # The exported driver: CheckOverlaps must be opt-in. Emitting the whole macro needs a CAD
    # model, so this asserts on the emitter's own source, which is where the default lives.
    import inspect as _inspect
    _src = _inspect.getsource(emit_root_macro)
    report("if (checkOverlaps) { gGeoManager->CheckOverlaps(); }" in _src,
           "the emitted build_and_export runs CheckOverlaps only on request")
    report("bool checkOverlaps=false" in _src,
           "and its default is off -- it cost ~15 min on oTOF's 62 628 placements")

    custom, _ = emit_materials_cpp(used, in_field=(1.0, 5.5))
    report("int   cad_ifield = 1;" in custom and "float cad_fieldm = 5.5;" in custom,
           "IFIELD,FIELDM overrides seed the query")

    print(f"\n{tally.checks - tally.failures}/{tally.checks} in-field media checks passed")
    return tally.failures


def run_bom_token_self_test() -> int:
    """Assert that BOM tokenisation strips the "EN AW" alloy prefix and nothing inside a word."""
    tally = _Checks()
    print("\nBOM material tokens")
    for text, want in (("Tungsten", ["tungsten", "w"]), ("EN AW-6082", ["6082"])):
        got = _norm_tokens(text)
        tally.report(got == want, f"_norm_tokens({text!r}) == {want}", str(got))
    return tally.failures


def run_multibody_leaf_self_test() -> int:
    """Assert that one XCAF leaf label carrying several solid bodies becomes several volumes, and
    that a single-body leaf keeps its bare label entry as definition key.

    Returns the number of failures; prints one line per check.
    """
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound

    tally = _Checks()
    report = tally.report

    def compound_of(*shapes):
        comp = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(comp)
        for s in shapes:
            builder.Add(comp, s)
        return comp

    print("\nMulti-body leaf labels: one label, one body each")

    # --- 1. a leaf label holding two boxes, instanced twice -----------------------------------
    _doc, st = _self_test_shape_tool()
    two_bodies = compound_of(BRepPrimAPI_MakeBox(gp_Pnt(0., 0., 0.), 1., 1., 1.).Shape(),
                             BRepPrimAPI_MakeBox(gp_Pnt(5., 0., 0.), 1., 1., 1.).Shape())
    part = st.AddShape(two_bodies, False)
    module = _self_test_assembly(st, [(part, gp_Trsf())])
    _self_test_assembly(st, [(module, _self_test_shift(dx=0.0)),
                             (module, _self_test_shift(dx=100.0))])
    st.UpdateAssemblies()
    rep, occ = _self_test_convert(st)
    report(len(logical_volumes) == 2,
           "a leaf label carrying two solid bodies becomes two logical volumes, not one",
           f"{len(logical_volumes)} logical volume(s)")
    report(len(occ) == 4 and len(set(occ)) == 4,
           "and instancing that label twice places four bodies, all at distinct transforms",
           f"{rep['declared_leaf_placements']} declared -> {len(occ)} placed, "
           f"{len(set(occ))} distinct")
    report(sum(rep["n_suppressed_by_rule"].values()) == 0,
           "the two bodies of one label never look like a coincident duplicate",
           f"{rep['n_suppressed_by_rule']}")

    # --- 2. the control: a single-body leaf keeps its bare label entry as the definition key ---
    _doc, st = _self_test_shape_tool()
    one_body = st.AddShape(BRepPrimAPI_MakeBox(2., 2., 2.).Shape(), False)
    _self_test_assembly(st, [(one_body, gp_Trsf())])
    st.UpdateAssemblies()
    _rep, occ = _self_test_convert(st)
    keys = list(logical_volumes)
    report(len(keys) == 1 and "#b" not in keys[0] and keys[0] == label_id(one_body),
           "a single-body leaf is untouched: one volume, keyed on the bare label entry",
           f"{keys}")
    report(len(occ) == 1, "and it is placed exactly once", f"{len(occ)} placed")

    # --- 3. a leaf label with no geometry at all is skipped, not crashed on ---------------------
    _doc, st = _self_test_shape_tool()
    empty = st.AddShape(compound_of(), False)
    good = st.AddShape(BRepPrimAPI_MakeBox(3., 3., 3.).Shape(), False)
    _self_test_assembly(st, [(empty, gp_Trsf()), (good, _self_test_shift(dx=10.0))])
    st.UpdateAssemblies()
    ok_empty = True
    detail = ""
    try:
        _rep, occ = _self_test_convert(st)
    except Exception as exc:          # the old failure mode: Bnd_Box is void
        ok_empty = False
        occ = []
        detail = f"{type(exc).__name__}: {exc}"
    report(ok_empty and len(logical_volumes) == 1 and len(occ) == 1,
           "an empty leaf label is dropped with a warning and its siblings still convert",
           detail or f"{len(logical_volumes)} volume(s), {len(occ)} placed")

    print(f"\n{tally.checks} checks, {tally.failures} failure(s)")
    return tally.failures


def _recognized_inner_wall(face, rec) -> Optional[bool]:
    """Decide, by measurement, which side of a RECOGNIZED quadric is outside the solid.

    On a NURBS-encoded quadric the orientation flag says nothing about the axis, so the face's own
    outward normal is compared with the quadric's radial direction at every sample. Returns None
    when the samples do not decide.
    """
    samples = rec.get("P")
    normals = rec.get("N")
    if samples is None or normals is None or len(samples) == 0:
        return None
    sign = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
    kind = rec["kind"]
    if kind in ("cylinder", "cone"):
        axis = np.asarray(rec["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
    votes = 0
    for point, normal in zip(samples, normals):
        outward = np.asarray(normal, dtype=float) * sign
        if kind == "cylinder":
            radial = point - rec["origin"]
            radial = radial - np.dot(radial, axis) * axis
        elif kind == "sphere":
            radial = point - rec["centre"]
        elif kind == "cone":
            relative = point - rec["apex"]
            radial = relative - np.dot(relative, axis) * axis
            # The cone's outward normal tilts out of the radial direction by the half angle; only
            # its sign relative to the radial direction matters here, and that tilt cannot flip it.
        else:
            return None
        length = np.linalg.norm(radial)
        if length < 1e-12:
            continue  # on the axis: this sample says nothing
        votes += 1 if float(np.dot(outward, radial / length)) > 0.0 else -1
    if votes == 0:
        return None
    return votes < 0


def _arbitrary_orthonormal_frame(axis):
    """One arbitrary orthonormal in-plane vector for an axis with no natural reference direction
    (a full/partial sphere has no preferred polar reference)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, axis) * axis
    return e1 / np.linalg.norm(e1)


def _recognized_quadric_wire_block(face, project):
    """Build line-only trim wires in the recognized (phi, other) domain from the edges' 3D curves.

    phi is unwrapped continuously over all samples of a wire; an accepted edge is iso in phi or in
    `other`, and a degenerate edge takes the incoming phi. Returns (wires, (phiStart, phiSweep,
    otherLo, otherHi), None), or (None, None, reason).
    """
    wires_edges = list(_face_wire_edges(face))
    if not wires_edges:
        return None, None, "recognized quadric face has no wires"

    n_samples = 9
    per_wire = []  # (is_outer, [ [ (phi_raw, other), ... ] or None (degenerate) per edge ], [ other_at_degenerate_vertex or None ])
    all_other = []
    for _wire, is_outer, edges in wires_edges:
        if len(edges) < 3:
            return None, None, "recognized quadric trim wire has fewer than 3 edges"
        edge_samples = []
        degenerate_other = []
        for edge, start_vertex in edges:
            if BRep_Tool.Degenerated(edge):
                _phi, other = project(BRep_Tool.Pnt(start_vertex))
                edge_samples.append(None)
                degenerate_other.append(other)
                all_other.append(other)
                continue
            degenerate_other.append(None)
            try:
                curve3d, first, last = BRep_Tool.Curve(edge)
            except Exception:
                curve3d = None
            if curve3d is None:
                return None, None, "recognized quadric boundary edge has no 3D curve"
            reversed_edge = edge.Orientation() == TopAbs_REVERSED
            samples = []
            for k in range(n_samples):
                tau = k / (n_samples - 1.0)
                t = (1.0 - tau) if reversed_edge else tau
                phi, other = project(curve3d.Value(first + t * (last - first)))
                samples.append((phi, other))
                all_other.append(other)
            edge_samples.append(samples)
        per_wire.append((is_outer, edge_samples, degenerate_other))
    tol_other = 1e-6 * max(1.0, max(all_other) - min(all_other))
    tol_phi = 1e-7

    wires_out: List[dict] = []
    outer_window = None
    for is_outer, edge_samples, degenerate_other in per_wire:
        n = len(edge_samples)
        unwrapped_edges = []
        prev_phi = None
        for samples, deg_other in zip(edge_samples, degenerate_other):
            if samples is None:
                # degenerate point: carry the running phi through unchanged (see docstring)
                if prev_phi is None:
                    prev_phi = 0.0
                unwrapped_edges.append([prev_phi] * n_samples)
                continue
            u_edge = []
            for phi_raw, _other in samples:
                if prev_phi is None:
                    phi_u = phi_raw
                else:
                    d = phi_raw - prev_phi
                    d -= 2.0 * math.pi * math.floor((d + math.pi) / (2.0 * math.pi))
                    phi_u = prev_phi + d
                u_edge.append(phi_u)
                prev_phi = phi_u
            unwrapped_edges.append(u_edge)

        starts = []
        all_phi_u, all_other_w = [], []
        for i, samples in enumerate(edge_samples):
            phis_u = unwrapped_edges[i]
            if samples is None:
                others = [degenerate_other[i]] * n_samples
            else:
                others = [o for _p, o in samples]
            all_phi_u.extend(phis_u)
            all_other_w.extend(others)
            is_iso_other = (max(others) - min(others)) <= tol_other
            is_iso_phi = (max(phis_u) - min(phis_u)) <= tol_phi
            if not (is_iso_other or is_iso_phi):
                return None, None, "recognized quadric boundary edge is not axis-aligned in (phi, h/theta)"
            starts.append((phis_u[0], others[0]))
        if is_outer:
            outer_window = (min(all_phi_u), max(all_phi_u) - min(all_phi_u), min(all_other_w), max(all_other_w))

        seg_edges = []
        for i in range(n):
            u0, v0 = starts[i]
            u1, v1 = starts[(i + 1) % n]
            seg_edges.append({"curve": "line", "params": [u0, v0, u1, v1]})
        wires_out.append({"role": "outer" if is_outer else "inner", "edges": seg_edges})

    n_outer = sum(1 for w in wires_out if w["role"] == "outer")
    if n_outer != 1:
        return None, None, f"recognized quadric face has {n_outer} outer trim wires (expected exactly 1)"
    return wires_out, outer_window, None


_NOT_RECOGNIZED_YET = object()


def recognize_and_extract_face(face, scale_to_cm: float,
                               rec=_NOT_RECOGNIZED_YET) -> Tuple[Optional[dict], Optional[str]]:
    """Canonical-form pre-pass: extract a face whose stored surface has no direct extractor through
    the exact plane/sphere/cylinder/cone behind it; (None, None) when it is not recognizable.
    `rec` is the recognizer's result when the surface report already computed it.
    """
    adaptor = BRepAdaptor_Surface(face)
    try:
        uv_bounds = breptools.UVBounds(face)
    except Exception:
        return None, None
    if rec is _NOT_RECOGNIZED_YET:
        rec = _recognize_analytic_surface(adaptor, uv_bounds)
    if rec is None:
        return None, None
    kind = rec["kind"]
    s = scale_to_cm
    # Which side is outside is measured on the face, falling back to the orientation flag.
    inner_wall = face.Orientation() == TopAbs_REVERSED
    measured_inner_wall = _recognized_inner_wall(face, rec)
    if measured_inner_wall is not None:
        inner_wall = measured_inner_wall

    if kind == "plane":
        normal = rec["normal"]
        e1 = _arbitrary_orthonormal_frame(normal)
        outward_sign = -1.0 if inner_wall else 1.0
        e2 = np.cross(normal, e1) * outward_sign  # axisU x axisV must equal the outward normal
        origin_cm = (rec["point"] * s).tolist()
        record, reason = extract_planar_face(face, s, frame_override=(origin_cm, e1.tolist(), e2.tolist()))
        if record is None:
            return None, f"recognized as plane but {reason}"
        record["recognized"] = {"kind": "plane", "residual": rec["residual"]}
        return record, None

    if kind == "cylinder":
        axis = rec["axis"] / np.linalg.norm(rec["axis"])
        refu = rec["refu"] - np.dot(rec["refu"], axis) * axis
        refu = refu / np.linalg.norm(refu)
        e2 = np.cross(axis, refu)
        origin_native = rec["origin"]

        def project(pnt):
            rel = np.array([pnt.X(), pnt.Y(), pnt.Z()]) - origin_native
            phi = math.atan2(np.dot(rel, e2), np.dot(rel, refu))
            return phi, float(np.dot(rel, axis)) * s

        wires, window, reason = _recognized_quadric_wire_block(face, project)
        if wires is None:
            return None, f"recognized as cylinder but {reason}"
        phi_start, phi_sweep, h_lo, h_hi = window
        if phi_sweep <= 0.0 or phi_sweep > 2.0 * math.pi + 1e-9:
            return None, "recognized cylinder trim wraps more than a full turn in phi"
        params = ((origin_native * s).tolist() + axis.tolist() + refu.tolist() +
                  [rec["radius"] * s, h_lo, h_hi, phi_start, phi_sweep])
        record = {"type": "cylinder", "inner_wall": inner_wall, "params": params, "wires": wires,
                  "recognized": {"kind": "cylinder", "residual": rec["residual"]}}
        return record, None

    if kind == "cone":
        axis = rec["axis"] / np.linalg.norm(rec["axis"])
        refu = rec["refu"] - np.dot(rec["refu"], axis) * axis
        refu = refu / np.linalg.norm(refu)
        e2 = np.cross(axis, refu)
        apex_native = rec["apex"]
        tan_half = math.tan(rec["half_angle"])

        def project(pnt):
            rel = np.array([pnt.X(), pnt.Y(), pnt.Z()]) - apex_native
            phi = math.atan2(np.dot(rel, e2), np.dot(rel, refu))
            return phi, float(np.dot(rel, axis)) * s

        wires, window, reason = _recognized_quadric_wire_block(face, project)
        if wires is None:
            return None, f"recognized as cone but {reason}"
        phi_start, phi_sweep, h_lo, h_hi = window
        if phi_sweep <= 0.0 or phi_sweep > 2.0 * math.pi + 1e-9:
            return None, "recognized cone trim wraps more than a full turn in phi"
        h_lo = max(0.0, h_lo)
        h_hi = max(h_lo, h_hi)
        params = ((apex_native * s).tolist() + axis.tolist() + refu.tolist() +
                  [h_lo * tan_half, h_hi * tan_half, h_lo, h_hi, phi_start, phi_sweep])
        record = {"type": "cone", "inner_wall": inner_wall, "params": params, "wires": wires,
                  "recognized": {"kind": "cone", "residual": rec["residual"]}}
        return record, None

    if kind == "sphere":
        centre_native = rec["centre"]
        # A sphere has no natural polar axis; any orthonormal frame is a valid (self-consistent)
        # (phi, theta) parametrization for this face.
        polar_axis = np.array([0.0, 0.0, 1.0])
        refu = _arbitrary_orthonormal_frame(polar_axis)
        e2 = np.cross(polar_axis, refu)

        def project(pnt):
            rel = (np.array([pnt.X(), pnt.Y(), pnt.Z()]) - centre_native) / rec["radius"]
            theta = math.acos(max(-1.0, min(1.0, float(np.dot(rel, polar_axis)))))
            phi = math.atan2(float(np.dot(rel, e2)), float(np.dot(rel, refu)))
            return phi, theta

        wires, window, reason = _recognized_quadric_wire_block(face, project)
        if wires is None:
            return None, f"recognized as sphere but {reason}"
        phi_start, phi_sweep, theta_lo, theta_hi = window
        if phi_sweep <= 0.0 or phi_sweep > 2.0 * math.pi + 1e-9:
            return None, "recognized sphere trim wraps more than a full turn in phi"
        params = ((centre_native * s).tolist() + polar_axis.tolist() + refu.tolist() +
                  [rec["radius"] * s, theta_lo, theta_hi, phi_start, phi_sweep])
        record = {"type": "sphere", "inner_wall": inner_wall, "params": params, "wires": wires,
                  "recognized": {"kind": "sphere", "residual": rec["residual"]}}
        return record, None

    return None, None


# Face extractors dispatched by analytic surface type.
_FACE_EXTRACTORS = {
    "plane": extract_planar_face,
    "cylinder": extract_cylindrical_face,
    "cone": extract_conical_face,
    "sphere": extract_spherical_face,
    "torus": extract_toroidal_face,
}


def extract_surfaces_for_shape(shape, scale_to_cm: float, recognize_surfaces: bool = True,
                               recognition=None,
                               lid=None) -> Tuple[Optional[List[dict]], List[str], int]:
    """Attempt to extract every face of a leaf solid into exact sidecar surface records.

    Returns (surfaces, [], nModelEdges), or (None, reasons, 0) when any face is unsupported, so an
    emitted sidecar describes all faces. `recognition` holds the surface report's results by
    (`lid`, face index).
    """
    surfaces: List[dict] = []
    reasons: List[str] = []
    n_faces = 0
    edge_map, edge_id = build_edge_table(shape)
    for index, face in enumerate(TopologyExplorer(shape).faces()):
        n_faces += 1
        adaptor = BRepAdaptor_Surface(face)
        surf_type = SURFACE_TYPE_NAME.get(adaptor.GetType(), "unknown")
        extractor = _FACE_EXTRACTORS.get(surf_type)
        wires = None
        if extractor is None:
            record, reason = None, f"{surf_type} face extraction not implemented yet"
        else:
            wires = list(_face_wire_edges(face))
            record, reason = extractor(face, scale_to_cm, wires=wires)
        if record is None and recognize_surfaces:
            known = (recognition.get((lid, index), _NOT_RECOGNIZED_YET) if recognition is not None
                     else _NOT_RECOGNIZED_YET)
            rec_record, rec_reason = recognize_and_extract_face(face, scale_to_cm, rec=known)
            if rec_record is not None:
                record, reason = rec_record, None
            elif rec_reason is not None:
                reason = f"{reason}; recognition attempted: {rec_reason}"
        if record is None:
            reasons.append(reason or f"{surf_type} face not supported")
        else:
            record["edge_refs"] = face_boundary_edge_refs(face, edge_id,
                                                         anchored=bool(record.get("wires")),
                                                         wires=wires)
            surfaces.append(record)
    if n_faces == 0:
        return None, ["shape has no faces"], 0
    if reasons:
        return None, reasons, 0
    return surfaces, [], edge_map.Size()


# -------------------------------
# BOM / material mapping
# -------------------------------

@dataclass(frozen=True)
class BomEntry:
    part_number: str
    revision: str
    name: str
    mass_value: float  # as in CSV
    material: str

    @property
    def part_number_key(self) -> str:
        return (self.part_number or "").strip()

    @property
    def name_key(self) -> str:
        return (self.name or "").strip()


def _to_float(s: str) -> Optional[float]:
    try:
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def read_bom_csv(csv_path: str) -> List[BomEntry]:
    """
    Reads a BOM CSV in the format provided by design team.

    We look for rows whose first column is 'CAD' and second is 'Mechanical/Part'.
    Columns (0-based):
      0 CAD
      1 type
      2 part number
      3 revision
      4 name/description
      5 mass
      6 material
    """
    entries: List[BomEntry] = []
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 7:
                continue
            if row[0].strip() != "CAD":
                continue
            if row[1].strip() != "Mechanical/Part":
                continue

            part_no = (row[2] or "").strip()
            rev = (row[3] or "").strip()
            name = (row[4] or "").strip()
            mass = _to_float(row[5])
            mat = (row[6] or "").strip()

            if not (part_no or name):
                continue
            if mass is None:
                mass = float("nan")
            if not mat:
                mat = "Default"

            entries.append(BomEntry(part_no, rev, name, float(mass), mat))
    return entries



def normalize_material_name(mat: str) -> str:
    """
    Normalizes a BOM material string for matching / caching.

    Note: We keep the *original* string for ROOT object names; this is only used
    internally for robust matching and dictionary keys.
    """
    mat = (mat or "Default").strip()
    mat = re.sub(r"\s+", " ", mat)
    return mat


def _norm_tokens(s: str) -> List[str]:
    s = (s or "").lower()
    # common grade/format noise
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\ben[\s-]*aw\b", " ", s)
    s = re.sub(r"\b(en|aw)\b", " ", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    toks = s.split(" ")

    # small synonym normalization
    syn = {
        "alu": "al",
        "aluminium": "aluminum",
        "silicium": "silicon",
        "inox": "stainless",
        "ss": "stainless",
        "cu": "copper",
        "fe": "iron",
        "ptfe": "teflon",
        "ti": "titanium",
        "be": "beryllium",
    }

    # Expand common element symbols to names and vice-versa so that e.g. "G4_Si" can match "silicon".
    elem_alias = {
        "h": "hydrogen", "he": "helium", "c": "carbon", "n": "nitrogen", "o": "oxygen",
        "al": "aluminum", "si": "silicon", "fe": "iron", "cu": "copper", "be": "beryllium",
        "mg": "magnesium", "mn": "manganese", "cr": "chromium", "ni": "nickel", "zn": "zinc",
        "ti": "titanium", "w": "tungsten", "pb": "lead", "sn": "tin",
    }
    name_to_sym = {v: k for k, v in elem_alias.items()}

    out: List[str] = []
    for t in toks:
        t2 = syn.get(t, t)
        out.append(t2)
        if t2 in elem_alias:
            out.append(elem_alias[t2])
        if t2 in name_to_sym:
            out.append(name_to_sym[t2])

    # de-dup while preserving order
    seen = set()
    out2: List[str] = []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            out2.append(t)
    return out2


def _density_score(rho_part: Optional[float], rho_ref: Optional[float]) -> float:
    if rho_part is None or rho_ref is None or not (rho_part > 0.0) or not (rho_ref > 0.0):
        return 0.0
    # symmetric score in log-space; 1.0 is perfect match
    d = abs(math.log(rho_ref / rho_part))
    return 1.0 / (1.0 + d)


def _token_score(tokens_a: List[str], tokens_b: List[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    sa = set(tokens_a)
    sb = set(tokens_b)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def load_g4_nist_db(json_path: str) -> Dict[str, dict]:
    """
    Loads a JSON dump created by the 'nist_export_all' tool.
    Returns a dict: nist_name -> material record.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mats = data.get("materials", {})
    if not isinstance(mats, dict) or not mats:
        raise RuntimeError(f"G4 NIST DB JSON seems empty or malformed: {json_path}")
    return mats

# Minimal periodic table for parsing custom alloys not present in NIST.
# Values: Z (atomic number), A (g/mol)
_ELEMENT_TABLE = {
    "H": (1, 1.00794),
    "C": (6, 12.0107),
    "N": (7, 14.0067),
    "O": (8, 15.9994),
    "Al": (13, 26.9815385),
    "Si": (14, 28.0855),
    "Fe": (26, 55.845),
    "Cu": (29, 63.546),
    "Be": (4, 9.0121831),
    "Mg": (12, 24.305),
    "Mn": (25, 54.938044),
    "Cr": (24, 51.9961),
    "Ni": (28, 58.6934),
    "Zn": (30, 65.38),
    "Ti": (22, 47.867),
    "W": (74, 183.84),
    "Pb": (82, 207.2),
    "Sn": (50, 118.71),
}


@dataclass
class ResolvedMaterial:
    bom_name: str
    nist_name: Optional[str]          # e.g. "G4_Al"
    score: float
    rho_used_g_cm3: Optional[float]   # density used in ROOT definition
    radlen_cm: Optional[float]
    intlen_cm: Optional[float]
    elements: Optional[List[dict]]    # list of {symbol,Z,A_g_mol,mass_fraction}
    note: str                         # for comments in geom.C (warnings/FIXME)

@dataclass
class MatMatchConfig:
    # Minimum combined score to accept a match.
    min_score: float = 0.35
    # If (best - second_best) < ambiguity_delta, treat as ambiguous/unresolved.
    ambiguity_delta: float = 0.05
    # Weights for the combined score = w_token * token_score + w_density * density_score
    w_token: float = 0.75
    w_density: float = 0.25
    # Optional hard filter on density proximity (in log-space). If <=0, disabled.
    # Example: max_log_density_diff=0.8 means accept within exp(0.8)~2.2x in either direction.
    max_log_density_diff: float = 0.0
    # Penalize compound matches (oxide/dioxide/carbide/...) when BOM doesn't mention those tokens.
    compound_penalty: float = 0.25


def resolve_bom_material(
    bom_material: str,
    rho_part_g_cm3: Optional[float],
    g4db: Optional[Dict[str, dict]],
    cfg: MatMatchConfig,
) -> ResolvedMaterial:
    """
    Resolves an arbitrary BOM material string to a Geant4 NIST material name using:
      - exact key match (BOM already uses e.g. "G4_Al")
      - token overlap scoring on names
      - density proximity scoring (if rho_part_g_cm3 available)

    If unresolved/ambiguous, tries to parse element symbols from the BOM string (e.g. "Cu Be")
    and emits a placeholder mixture (equal mass fractions) annotated with FIXME.
    """
    raw_bom_material = (bom_material or "").strip()
    bom_material = normalize_material_name(bom_material)

    if not g4db:
        return ResolvedMaterial(
            bom_name=bom_material,
            nist_name=None,
            score=0.0,
            rho_used_g_cm3=rho_part_g_cm3,
            radlen_cm=None,
            intlen_cm=None,
            elements=None,
            note="FIXME: No Geant4 NIST DB provided; using dummy material.",
        )

    # Trivial: BOM already provides an exact Geant4 material key
    if bom_material in g4db:
        rec = g4db[bom_material]
        rho_ref = rec.get("density_g_cm3")
        # Use NIST density for emission; CAD-derived density is used only for matching.
        rho_used = rho_ref

        rad = rec.get("radlen_cm")
        itl = rec.get("intlen_cm")

        return ResolvedMaterial(
            bom_name=bom_material,
            nist_name=bom_material,
            score=1.0,
            rho_used_g_cm3=rho_used,
            radlen_cm=rad,
            intlen_cm=itl,
            elements=rec.get("elements", []),
            note="Resolved by exact Geant4 NIST name from BOM.",
        )

    bom_toks = _norm_tokens(bom_material)
    if not bom_toks:
        return ResolvedMaterial(
            bom_name=bom_material,
            nist_name=None,
            score=0.0,
            rho_used_g_cm3=rho_part_g_cm3,
            radlen_cm=None,
            intlen_cm=None,
            elements=None,
            note="FIXME: Empty/unknown BOM material string; using dummy material.",
        )

    def _build_custom_from_elements(note_prefix: str) -> Optional[ResolvedMaterial]:
        s = raw_bom_material
        if not s:
            return None

        symbols = set(re.findall(r"\b([A-Z][a-z]?)\b", s))
        name_to_symbol = {
            "aluminum": "Al", "aluminium": "Al", "silicon": "Si", "iron": "Fe", "copper": "Cu",
            "beryllium": "Be", "magnesium": "Mg", "manganese": "Mn", "chromium": "Cr", "nickel": "Ni",
            "zinc": "Zn", "titanium": "Ti", "tungsten": "W", "lead": "Pb", "tin": "Sn",
        }
        for t in bom_toks:
            if t in name_to_symbol:
                symbols.add(name_to_symbol[t])

        symbols = [sym for sym in sorted(symbols) if sym in _ELEMENT_TABLE]
        if not symbols:
            return None

        frac = 1.0 / float(len(symbols))
        elems: List[dict] = []
        for sym in symbols:
            Z, A = _ELEMENT_TABLE[sym]
            elems.append({"symbol": sym, "Z": Z, "A_g_mol": A, "mass_fraction": frac})

        return ResolvedMaterial(
            bom_name=bom_material,
            nist_name=None,
            score=0.0,
            rho_used_g_cm3=rho_part_g_cm3,
            radlen_cm=None,
            intlen_cm=None,
            elements=elems,
            note=f"FIXME: {note_prefix} No suitable Geant4 NIST material. Emitting placeholder mixture from parsed elements {symbols} with equal mass fractions; please adjust fractions/material.",
        )

    best = (None, -1.0, 0.0, 0.0)   # (nist_name, score, dens_score, token_score)
    second = (None, -1.0, 0.0, 0.0)

    bom_has_compound = any(t in bom_toks for t in (
        "oxide", "dioxide", "carbide", "nitride", "fluoride", "chloride",
        "sulfate", "phosphate", "glass", "dioxyde"
    ))

    for nist_name, rec in g4db.items():
        nist_toks = _norm_tokens(nist_name)
        ts = _token_score(bom_toks, nist_toks)
        if ts <= 0.0:
            continue

        ds = _density_score(rho_part_g_cm3, rec.get("density_g_cm3"))

        # Optional hard density filter
        if cfg.max_log_density_diff and cfg.max_log_density_diff > 0.0 and rho_part_g_cm3 and rec.get("density_g_cm3"):
            try:
                if abs(math.log(float(rec.get("density_g_cm3")) / float(rho_part_g_cm3))) > cfg.max_log_density_diff:
                    continue
            except Exception:
                pass

        nist_has_compound = any(t in nist_toks for t in (
            "oxide", "dioxide", "carbide", "nitride", "fluoride", "chloride",
            "sulfate", "phosphate", "glass", "dioxyde"
        ))
        compound_pen = cfg.compound_penalty if (nist_has_compound and not bom_has_compound) else 0.0

        score = cfg.w_token * ts + cfg.w_density * ds - compound_pen

        if score > best[1]:
            second = best
            best = (nist_name, score, ds, ts)
        elif score > second[1]:
            second = (nist_name, score, ds, ts)

    nist_best, score_best, ds_best, ts_best = best
    nist_second, score_second, _, _ = second

    if nist_best is None or score_best < cfg.min_score:
        custom = _build_custom_from_elements("Could not resolve with enough confidence.")
        if custom is not None:
            return custom
        return ResolvedMaterial(
            bom_name=bom_material,
            nist_name=None,
            score=float(score_best if score_best > 0 else 0.0),
            rho_used_g_cm3=rho_part_g_cm3,
            radlen_cm=None,
            intlen_cm=None,
            elements=None,
            note="FIXME: Could not resolve BOM material to a Geant4 NIST material with enough confidence; using dummy material.",
        )

    if score_second > 0 and (score_best - score_second) < cfg.ambiguity_delta:
        custom = _build_custom_from_elements(
            f"Ambiguous material match (best '{nist_best}' score={score_best:.3f}, second '{nist_second}' score={score_second:.3f})."
        )
        if custom is not None:
            return custom
        return ResolvedMaterial(
            bom_name=bom_material,
            nist_name=None,
            score=float(score_best),
            rho_used_g_cm3=rho_part_g_cm3,
            radlen_cm=None,
            intlen_cm=None,
            elements=None,
            note=f"FIXME: Ambiguous material match (best '{nist_best}' score={score_best:.3f}, second '{nist_second}' score={score_second:.3f}); using dummy material.",
        )

    rec = g4db[nist_best]
    rho_ref = rec.get("density_g_cm3")
    # Use NIST density for emission; CAD-derived density is used only for matching.
    rho_used = rho_ref

    rad = rec.get("radlen_cm")
    itl = rec.get("intlen_cm")

    return ResolvedMaterial(
        bom_name=bom_material,
        nist_name=nist_best,
        score=float(score_best),
        rho_used_g_cm3=rho_used,
        radlen_cm=rad,
        intlen_cm=itl,
        elements=rec.get("elements", []),
        note=f"Resolved to '{nist_best}' (token={ts_best:.3f}, density={ds_best:.3f}, score={score_best:.3f}).",
    )


def build_volume_to_material_map(
    bom_entries: List[BomEntry],
    def_names: Dict[str, str],
) -> Dict[str, BomEntry]:
    """
    Builds a mapping def_lid -> BomEntry by matching the XCAF display name to:
      - exact part_number match
      - exact description/name match
      - substring match on part_number within the XCAF name

    This is heuristic; if nothing matches we keep no assignment for that volume.
    """
    # lookup tables
    by_part: Dict[str, BomEntry] = {}
    by_name: Dict[str, BomEntry] = {}
    for e in bom_entries:
        if e.part_number_key:
            by_part[e.part_number_key] = e
        if e.name_key and e.name_key not in by_name:
            by_name[e.name_key] = e

    out: Dict[str, BomEntry] = {}
    for lid, disp in def_names.items():
        key = (disp or "").strip()
        if not key:
            continue

        # 1) exact part number
        if key in by_part:
            out[lid] = by_part[key]
            continue
        # 2) exact name/description
        if key in by_name:
            out[lid] = by_name[key]
            continue
        # 3) substring match on any part number
        for pn, e in by_part.items():
            if pn and pn in key:
                out[lid] = e
                break
    return out


# -------------------------------
# C++ emission helpers
# -------------------------------

def trsf_to_tgeo(trsf: gp_Trsf, name: str, scale_to_cm: float) -> str:
    m = trsf.GetRotation().GetMatrix()
    t = trsf.TranslationPart()
    return f"""
  Double_t {name}_m[9] = {{
    {m.Value(1,1)}, {m.Value(1,2)}, {m.Value(1,3)},
    {m.Value(2,1)}, {m.Value(2,2)}, {m.Value(2,3)},
    {m.Value(3,1)}, {m.Value(3,2)}, {m.Value(3,3)}
  }};
  TGeoRotation *{name}_rot = new TGeoRotation();
  {name}_rot->SetMatrix({name}_m);
  TGeoCombiTrans *{name} = new TGeoCombiTrans({t.X()*scale_to_cm}, {t.Y()*scale_to_cm}, {t.Z()*scale_to_cm}, {name}_rot);
"""


def emit_cpp_prelude(exact_surfaces: bool = False, csg_shapes: bool = False,
                     flat_csg_shapes: bool = False, o2_tessellated: bool = False,
                     in_field: bool = False) -> str:
    prelude = """#include <TGeoManager.h>
#include <TFile.h>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

static void LoadFacets(const std::string& file, TGeoTessellated* solid, bool check=false)
{
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open facet file: " + file);

  uint32_t nTri = 0;
  in.read(reinterpret_cast<char*>(&nTri), sizeof(nTri));
  if (!in) throw std::runtime_error("Bad facet header in: " + file);

  for (uint32_t i=0;i<nTri;i++) {
    float v[9];
    in.read(reinterpret_cast<char*>(v), sizeof(v));
    if (!in) throw std::runtime_error("Unexpected EOF in: " + file);

    solid->AddFacet(TGeoTessellated::Vertex_t(v[0],v[1],v[2]),
                    TGeoTessellated::Vertex_t(v[3],v[4],v[5]),
                    TGeoTessellated::Vertex_t(v[6],v[7],v[8]));
  }
  solid->CloseShape(check, true);
}
"""
    if in_field:
        # --in-field queries the live field through headers Cling parses standalone.
        prelude += """#include "Field/MagneticField.h"
#include "TVirtualMC.h"

// The live field's integration mode and maximum, exactly as
// o2::base::Detector::initFieldTrackingParams computes them. Values passed in are the fallback
// used when no field is loaded.
static void cadFieldTrackingParams(int& mode, float& maxfield)
{
  auto vmc = TVirtualMC::GetMC();
  if (!vmc) {
    return;
  }
  if (auto* fld = dynamic_cast<o2::field::MagneticField*>(vmc->GetMagField())) {
    mode = fld->Integral();
    maxfield = fld->Max();
  }
}
"""

    if csg_shapes:
        # TGeoHMatrix comes in through TGeoManager.h today, but the CSG loader names it directly
        # and must not depend on that.
        prelude += "#include <TGeoMatrix.h>\n"
        prelude += import_csg_hook().CPP_LOADER
    if flat_csg_shapes:
        prelude += import_csg_hook().FLAT_CPP_PRELUDE
    if not exact_surfaces and not o2_tessellated:
        return prelude

    # The navigable solids need libO2CADSupport; headers are included, never declared by prototype.
    prelude += """
// --- navigable O2 solid support (requires the ALICE O2 environment) ---
R__ADD_INCLUDE_PATH($O2_ROOT/include)
R__LOAD_LIBRARY(libO2CADSupport)
"""
    if o2_tessellated:
        # O2Tessellated navigates the facets; ROOT's TGeoTessellated only navigates as its bbox.
        prelude += """#include "DetectorsBase/O2Tessellated.h"
#include "CADSupport/O2SurfaceSolidIO.h"

static void LoadFacetsO2(const std::string& file, o2::base::O2Tessellated* solid, bool check=false)
{
  if (!o2::cad::LoadFacetSolid(file, *solid)) {
    throw std::runtime_error("Cannot load facet sidecar: " + file);
  }
  solid->CloseShape(check, true, false);
}
"""
    if not exact_surfaces:
        return prelude
    prelude += """#include "CADSupport/O2BVHSurfaceSolid.h"
// The loader comes from its own public header, NOT from a hand-rolled prototype.
// o2::cad::loadCADGeometryHook JITs this macro inside a unique namespace and hoists
// only lines beginning with '#' to global scope, so a `namespace o2 { namespace cad {`
// block here becomes `<wrapper>::o2::cad` and shadows the real one -- every later
// `o2::cad::O2BVHSurfaceSolid` then fails to resolve and the whole module silently
// does not load. An #include is hoisted, so it declares the right symbol.
#include "CADSupport/O2SurfaceSolidIO.h"

static void LoadSurfaces(const std::string& file, o2::cad::O2BVHSurfaceSolid* solid, bool check=false)
{
  if (!o2::cad::LoadSurfaceSolid(file, *solid)) {
    throw std::runtime_error("Cannot load surface sidecar: " + file);
  }
  solid->CloseShape(check);
  if (check && (!solid->IsClosed() || !solid->IsOrientationConsistent())) {
    throw std::runtime_error("Surface solid not closed/orientation-consistent: " + file);
  }
}
"""
    return prelude


def emit_media_sidecar_cpp(sidecar: dict) -> Tuple[str, Dict[str, str]]:
    """Emit the media of a TGeo -> STEP writer sidecar verbatim, field for field.

    Returns the C++ block and a map from medium name to its C++ variable.
    """
    cpp: List[str] = []
    cpp.append("  // Media rebuilt verbatim from the TGeo -> STEP media sidecar.")
    cpp.append("  // Default stays as the fallback for a part the sidecar does not name.")
    cpp.append("  TGeoMaterial *mat_Default = new TGeoMaterial(\"Default\", 0., 0., 0.);")
    cpp.append("  TGeoMedium   *med_Default = new TGeoMedium(\"Default\", 1, mat_Default);")
    cpp.append("")

    medium_var: Dict[str, str] = {"Default": "med_Default"}
    order = list(sidecar.get("mediumParamOrder") or
                 ("isvol", "ifield", "fieldm", "tmaxfd",
                  "stemax", "deemax", "epsil", "stmin"))

    for name in sorted(sidecar.get("media", {})):
        rec = sidecar["media"][name]
        mat = rec["material"]
        safe = sanitize_cpp_name(name)
        mvar, medvar = f"mat_{safe}", f"med_{safe}"

        if mat.get("isMixture"):
            els = mat.get("elements", [])
            cpp.append(f"  TGeoMixture *{mvar} = new TGeoMixture(\"{mat['name']}\", "
                       f"{len(els)}, {mat['density']:.17g});")
            for el in els:
                cpp.append(f"  {mvar}->AddElement({el['A']:.17g}, {el['Z']:.17g}, "
                           f"{el['W']:.17g});")
        else:
            cpp.append(f"  TGeoMaterial *{mvar} = new TGeoMaterial(\"{mat['name']}\", "
                       f"{mat['A']:.17g}, {mat['Z']:.17g}, {mat['density']:.17g});")

        # No SetRadLen: ROOT recomputes it from the recipe, which is carried exactly.
        cpp.append(f"  TGeoMedium *{medvar} = new TGeoMedium(\"{name}\", {int(rec['id'])}, "
                   f"{mvar});")
        for i, key in enumerate(order):
            cpp.append(f"  {medvar}->SetParam({i}, {float(rec['params'][key]):.17g});"
                       f"   // {key}")
        cpp.append("")
        medium_var[name] = medvar

    return "\n".join(cpp), medium_var


# The eight Geant medium parameters, in the order TGeoMedium stores them.
MEDIUM_PARAM_ORDER = ("isvol", "ifield", "fieldm", "tmaxfd",
                      "stemax", "deemax", "epsil", "stmin")

# The --in-field seed: Detector::initFieldTrackingParams's own fallback; step controls stay 0.
IN_FIELD_SEED = (2, 10.0)


def _in_field_setparams(medvar: str) -> List[str]:
    """The eight SetParam lines for one medium under `--in-field`.

    ifield and fieldm come from the live-field query; the step controls stay 0 (transport default).
    """
    out: List[str] = []
    for i, key in enumerate(MEDIUM_PARAM_ORDER):
        if key == "ifield":
            out.append(f"  {medvar}->SetParam({i}, cad_ifield);   // ifield, from the live field")
        elif key == "fieldm":
            out.append(f"  {medvar}->SetParam({i}, cad_fieldm);   // fieldm, from the live field")
        else:
            out.append(f"  {medvar}->SetParam({i}, 0);   // {key} (transport default)")
    return out


def emit_materials_cpp(
    used_materials: Dict[str, ResolvedMaterial],
    in_field: Optional[Tuple[float, float]] = None,
    # key: BOM material string as used in CSV after normalization
) -> Tuple[str, Dict[str, str]]:
    """
    Emits C++ code defining TGeoMaterial/TGeoMixture + TGeoMedium for all used materials.

    - A resolved Geant4 NIST material becomes a mixture, with RadLen/IntLen when available.
    - An unresolved one becomes a dummy material with FIXME comments.
    - With `in_field` the eight medium parameters are written, ifield and fieldm from the live field.
    """
    cpp: List[str] = []
    cpp.append("  // Default material/medium (placeholder; can be replaced later)")
    cpp.append("  TGeoMaterial *mat_Default = new TGeoMaterial(\"Default\", 0., 0., 0.);")
    cpp.append("  TGeoMedium   *med_Default = new TGeoMedium(\"Default\", 1, mat_Default);")
    if in_field is not None:
        cpp.append("")
        cpp.append("  // Field tracking parameters, taken from the LIVE field: the same query")
        cpp.append("  // o2::base::Detector::initFieldTrackingParams makes, so a CAD module is not")
        cpp.append("  // treated differently from a hand-written detector. The seeds below are what")
        cpp.append("  // that function itself falls back to when no field is loaded.")
        cpp.append(f"  int   cad_ifield = {int(in_field[0])};")
        cpp.append(f"  float cad_fieldm = {float(in_field[1]):.17g};")
        cpp.append("  cadFieldTrackingParams(cad_ifield, cad_fieldm);")
        cpp.append("")
        cpp.extend(_in_field_setparams("med_Default"))
    cpp.append("")

    emitted_el: Dict[str, str] = {}

    def _emit_element(el: dict) -> str:
        sym = el.get("symbol", "X")
        Z = int(el.get("Z", 0))
        A = float(el.get("A_g_mol", 0.0))
        if sym in emitted_el:
            return emitted_el[sym]
        safe = sanitize_cpp_name(sym)
        var = f"el_{safe}"
        cpp.append(f"  TGeoElement *{var} = new TGeoElement(\"{sym}\", \"{sym}\", {Z}, {A:.10g});")
        emitted_el[sym] = var
        return var

    medium_var: Dict[str, str] = {"Default": "med_Default"}
    next_id = 2

    for bom_mat in sorted(used_materials.keys(), key=lambda s: s.lower()):
        rm = used_materials[bom_mat]
        safe = sanitize_cpp_name(bom_mat)
        base = safe
        k = 2
        while f"med_{safe}" in medium_var.values():
            safe = f"{base}_{k}"
            k += 1

        rho = rm.rho_used_g_cm3 if (rm.rho_used_g_cm3 and rm.rho_used_g_cm3 > 0.0) else 0.0

        cpp.append(f"  // BOM material: {rm.bom_name}")
        cpp.append(f"  // {rm.note}")

        if rm.elements:
            elems = rm.elements
            if len(elems) == 1 and abs(float(elems[0].get('mass_fraction', 1.0)) - 1.0) < 1e-6:
                el = elems[0]
                A = float(el.get("A_g_mol", 0.0))
                Z = float(el.get("Z", 0))
                cpp.append(f"  TGeoMaterial *mat_{safe} = new TGeoMaterial(\"{bom_mat}\", {A:.10g}, {Z:.10g}, {rho:.10g});")
            else:
                cpp.append(f"  TGeoMixture  *mat_{safe} = new TGeoMixture(\"{bom_mat}\", {len(elems)}, {rho:.10g});")
                for el in elems:
                    elvar = _emit_element(el)
                    w = float(el.get("mass_fraction", 0.0))
                    cpp.append(f"  mat_{safe}->AddElement({elvar}, {w:.10g});")

            if rm.radlen_cm is not None and rm.intlen_cm is not None:
                cpp.append(f"  mat_{safe}->SetRadLen({float(rm.radlen_cm):.10g}, {float(rm.intlen_cm):.10g});")
            elif rm.radlen_cm is not None:
                cpp.append(f"  mat_{safe}->SetRadLen({float(rm.radlen_cm):.10g});")
        else:
            cpp.append("  // FIXME: Unresolved material. Replace with a proper TGeoMaterial/TGeoMixture.")
            cpp.append(f"  TGeoMaterial *mat_{safe} = new TGeoMaterial(\"{bom_mat}\", 0., 0., {rho:.10g});")

        cpp.append(f"  TGeoMedium   *med_{safe} = new TGeoMedium(\"{bom_mat}\", {next_id}, mat_{safe});")
        if in_field is not None:
            cpp.extend(_in_field_setparams(f"med_{safe}"))
        cpp.append("")
        medium_var[bom_mat] = f"med_{safe}"
        next_id += 1

    return "\n".join(cpp), medium_var




def emit_tessellated_cpp(lid: str, vol_display_name: str, facet_abspath: str, ntriangles: int, medium_var: str,
                         solid_class: str = "o2::base::O2Tessellated") -> str:
    """Emit the tessellated fallback for one leaf solid.

    ``solid_class`` defaults to O2Tessellated; ROOT's TGeoTessellated navigates as its bbox.
    """
    safe = sanitize_cpp_name(lid)
    shape_name = vol_display_name if vol_display_name else lid

    if ntriangles <= 0:
        out = []
        out.append(f'  TGeoBBox *solid_{safe} = new TGeoBBox("{shape_name}", 0.001, 0.001, 0.001);')
        out.append(f'  TGeoVolume *vol_{safe} = new TGeoVolume("{shape_name}", solid_{safe}, {medium_var});')
        return "\n".join(out)

    loader = "LoadFacetsO2" if solid_class != "TGeoTessellated" else "LoadFacets"
    out = []
    out.append(f'  {solid_class} *solid_{safe} = new {solid_class}("{shape_name}", {ntriangles});')
    out.append(f'  {loader}("{facet_abspath}", solid_{safe}, check);')
    out.append(f'  TGeoVolume *vol_{safe} = new TGeoVolume("{shape_name}", solid_{safe}, {medium_var});')
    return "\n".join(out)


def emit_surface_solid_cpp(lid: str, vol_display_name: str, surface_abspath: str, medium_var: str) -> str:
    """Exact-surface counterpart of emit_tessellated_cpp: the volume gets an
    O2BVHSurfaceSolid filled from a surface sidecar file. Requires
    emit_cpp_prelude(exact_surfaces=True)."""
    safe = sanitize_cpp_name(lid)
    shape_name = vol_display_name if vol_display_name else lid

    out = []
    out.append(f'  auto *solid_{safe} = new o2::cad::O2BVHSurfaceSolid("{shape_name}");')
    out.append(f'  LoadSurfaces("{surface_abspath}", solid_{safe}, check);')
    out.append(f'  TGeoVolume *vol_{safe} = new TGeoVolume("{shape_name}", solid_{safe}, {medium_var});')
    return "\n".join(out)


def emit_assembly_cpp(lid: str, asm_display_name: str) -> str:
    safe = sanitize_cpp_name(lid)
    name = asm_display_name if asm_display_name else lid
    return f'  TGeoVolumeAssembly *asm_{safe} = new TGeoVolumeAssembly("{name}");'


# -------------------------------
# CAD clipping helpers
# -------------------------------

def make_clip_box_shape(clip_box: ClipBox):
    return BRepPrimAPI_MakeBox(
        gp_Pnt(clip_box.xmin, clip_box.ymin, clip_box.zmin),
        gp_Pnt(clip_box.xmax, clip_box.ymax, clip_box.zmax),
    ).Shape()


def _compose_trsf(parent_to_world: gp_Trsf, local_to_parent: gp_Trsf) -> gp_Trsf:
    return parent_to_world.Multiplied(local_to_parent)


def _shape_is_empty(shape) -> bool:
    if shape is None:
        return True
    try:
        if shape.IsNull():
            return True
    except Exception:
        pass
    try:
        for _ in TopologyExplorer(shape).faces():
            return False
        return True
    except Exception:
        return False


def _transformed_bbox(shape, trsf: gp_Trsf) -> Optional[Tuple[float, float, float, float, float, float]]:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    except Exception:
        return None

    points = []
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            for z in (zmin, zmax):
                p = gp_Pnt(x, y, z)
                p.Transform(trsf)
                points.append((p.X(), p.Y(), p.Z()))

    return (
        min(p[0] for p in points),
        min(p[1] for p in points),
        min(p[2] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
        max(p[2] for p in points),
    )


def _bbox_outside_clip_box(bbox: Tuple[float, float, float, float, float, float], clip_box: ClipBox) -> bool:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return (
        xmax < clip_box.xmin or xmin > clip_box.xmax or
        ymax < clip_box.ymin or ymin > clip_box.ymax or
        zmax < clip_box.zmin or zmin > clip_box.zmax
    )


def _bbox_inside_clip_box(bbox: Tuple[float, float, float, float, float, float], clip_box: ClipBox) -> bool:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return (
        xmin >= clip_box.xmin and xmax <= clip_box.xmax and
        ymin >= clip_box.ymin and ymax <= clip_box.ymax and
        zmin >= clip_box.zmin and zmax <= clip_box.zmax
    )


def _classify_shape_against_clip_box(shape, clip_box: ClipBox, local_to_world: gp_Trsf) -> Optional[str]:
    world_bbox = _transformed_bbox(shape, local_to_world)
    if world_bbox is None:
        return None
    if _bbox_outside_clip_box(world_bbox, clip_box):
        return "outside"
    if _bbox_inside_clip_box(world_bbox, clip_box):
        return "inside"
    return "overlap"


def clip_shape_to_box(shape, clip_box: ClipBox, clip_box_shape, local_to_world: gp_Trsf, lid: str):
    clip_state = _classify_shape_against_clip_box(shape, clip_box, local_to_world)
    if clip_state is None:
        return None
    if clip_state == "outside":
        return None
    if clip_state == "inside":
        return shape

    local_clip = BRepBuilderAPI_Transform(clip_box_shape, local_to_world.Inverted(), True).Shape()
    common = BRepAlgoAPI_Common(shape, local_clip)
    common.Build()
    if not common.IsDone():
        raise RuntimeError(f"Failed to clip CAD shape {lid} against --clip-box")

    clipped = common.Shape()
    if _shape_is_empty(clipped):
        return None
    return clipped


# -------------------------------
# Definition graph extraction
# -------------------------------

logical_volumes: Dict[str, list] = {}     # def_lid -> triangles
def_names: Dict[str, str] = {}           # def_lid -> human display name (may be "")
def_volume_source: Dict[str, object] = {}   # def_lid -> unclipped leaf shape, for the BOM volume
def_shapes: Dict[str, object] = {}       # def_lid -> (possibly clipped) TopoDS shape (leaf only)
assemblies = set()                       # def_lid
placements = []                          # (parent_def_lid, child_def_lid, gp_Trsf local)
top_defs = set()                         # top definition lids
visited_defs = set()                     # expanded defs


def reset_graph() -> None:
    """Clear the definition graph. One place, so `extract_graph` and the self-test agree."""
    global logical_volumes, def_names, def_volume_source, def_shapes, assemblies, placements, top_defs, visited_defs
    logical_volumes = {}
    def_names = {}
    def_volume_source = {}
    def_shapes = {}
    assemblies = set()
    placements = []
    top_defs = set()
    visited_defs = set()


def cpp_var_for_def(lid: str) -> str:
    safe = sanitize_cpp_name(lid)
    return f"asm_{safe}" if lid in assemblies else f"vol_{safe}"


def solid_bodies_of(shape) -> list:
    """The TopoDS_Solid bodies a shape carries, each with its own location already baked in."""
    if shape is None:
        return []
    try:
        if shape.IsNull():
            return []
    except Exception:
        return []
    out = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        out.append(topods.Solid(exp.Current()))
        exp.Next()
    return out


def _register_leaf_shape(def_key: str, shape, meshparam, scale_to_cm: float,
                         clip_enabled: bool, clip_box, clip_box_shape,
                         world_trsf, def_lid: str) -> bool:
    """Record one leaf logical volume: its unclipped shape, its shape and its triangles.

    Returns False when clipping removed the shape entirely, in which case nothing is recorded.
    """
    source = shape
    if clip_enabled:
        shape = clip_shape_to_box(shape, clip_box, clip_box_shape, world_trsf, def_lid)
        if shape is None:
            return False

    def_volume_source[def_key] = source
    def_shapes[def_key] = shape

    do_meshing = (meshparam is not None) and meshparam.get("do_meshing", None) is True
    logical_volumes[def_key] = (triangulate_CAD_solid(shape, meshparam=meshparam, scale_to_cm=scale_to_cm)
                                if do_meshing else triangulate_asbbox(shape, scale_to_cm=scale_to_cm))
    return True


def expand_definition(
    def_label: TDF_Label,
    shape_tool,
    meshparam=None,
    scale_to_cm: float = 1.0,
    clip_box: Optional[ClipBox] = None,
    clip_box_shape=None,
    clip_deduplicate: str = "intact",
    name_filter: Optional[NameFilter] = None,
    include_subtree: bool = False,
    world_trsf: Optional[gp_Trsf] = None,
    occ_path: str = "r1",
) -> Optional[str]:
    clip_enabled = clip_box_shape is not None
    if world_trsf is None:
        world_trsf = gp_Trsf()

    def_lid = label_id(def_label)
    nm = label_name(def_label)

    subtree_included = include_subtree
    if name_filter is not None:
        if name_filter.matches_exclude(def_lid, nm):
            return None
        if name_filter.has_include and name_filter.matches_include(def_lid, nm):
            subtree_included = True

    if clip_enabled and clip_box is not None:
        try:
            shape_for_clip = shape_tool.GetShape(def_label)
        except Exception:
            shape_for_clip = None
        if shape_for_clip is not None:
            clip_state = _classify_shape_against_clip_box(shape_for_clip, clip_box, world_trsf)
            if clip_state == "outside":
                return None
            if clip_state == "inside" and clip_deduplicate == "intact":
                return expand_definition(
                    def_label,
                    shape_tool,
                    meshparam=meshparam,
                    scale_to_cm=scale_to_cm,
                    clip_box=None,
                    clip_box_shape=None,
                    clip_deduplicate=clip_deduplicate,
                    name_filter=name_filter,
                    include_subtree=subtree_included,
                )

    def_key = f"{def_lid}@{occ_path}" if clip_enabled else def_lid
    if not clip_enabled and def_lid in visited_defs:
        return def_lid
    if not clip_enabled:
        visited_defs.add(def_lid)

    if nm and def_key not in def_names:
        def_names[def_key] = nm
    elif def_key not in def_names:
        def_names[def_key] = ""

    children = TDF_LabelSequence()
    shape_tool.GetComponents(def_label, children)
    has_children = children.Length() > 0

    if has_children or shape_tool.IsAssembly(def_label):
        assemblies.add(def_key)
        kept_children = 0

        for i in range(children.Length()):
            child = children.Value(i + 1)
            child_occ_path = f"{occ_path}_{i + 1}"
            if shape_tool.IsReference(child):
                referred = TDF_Label()
                shape_tool.GetReferredShape(child, referred)

                loc = shape_tool.GetLocation(child)
                trsf = loc.Transformation()
                if clip_enabled:
                    child_key = expand_definition(
                        referred,
                        shape_tool,
                        meshparam=meshparam,
                        scale_to_cm=scale_to_cm,
                        clip_box=clip_box,
                        clip_box_shape=clip_box_shape,
                        clip_deduplicate=clip_deduplicate,
                        name_filter=name_filter,
                        include_subtree=subtree_included,
                        world_trsf=_compose_trsf(world_trsf, trsf),
                        occ_path=child_occ_path,
                    )
                    if child_key is None:
                        continue
                    placements.append((def_key, child_key, trsf))
                else:
                    child_key = expand_definition(
                        referred,
                        shape_tool,
                        meshparam=meshparam,
                        scale_to_cm=scale_to_cm,
                        clip_deduplicate=clip_deduplicate,
                        name_filter=name_filter,
                        include_subtree=subtree_included,
                    )
                    if child_key is None:
                        continue
                    placements.append((def_key, child_key, trsf))
            else:
                trsf = gp_Trsf()
                if clip_enabled:
                    child_key = expand_definition(
                        child,
                        shape_tool,
                        meshparam=meshparam,
                        scale_to_cm=scale_to_cm,
                        clip_box=clip_box,
                        clip_box_shape=clip_box_shape,
                        clip_deduplicate=clip_deduplicate,
                        name_filter=name_filter,
                        include_subtree=subtree_included,
                        world_trsf=world_trsf,
                        occ_path=child_occ_path,
                    )
                    if child_key is None:
                        continue
                    placements.append((def_key, child_key, trsf))
                else:
                    child_key = expand_definition(
                        child,
                        shape_tool,
                        meshparam=meshparam,
                        scale_to_cm=scale_to_cm,
                        clip_deduplicate=clip_deduplicate,
                        name_filter=name_filter,
                        include_subtree=subtree_included,
                    )
                    if child_key is None:
                        continue
                    placements.append((def_key, child_key, trsf))
            kept_children += 1

        if (clip_enabled or (name_filter is not None and name_filter.has_include)) and kept_children == 0:
            assemblies.discard(def_key)
            return None
        return def_key

    if shape_tool.IsSimpleShape(def_label):
        if name_filter is not None and name_filter.has_include and not subtree_included:
            return None

        if def_key in logical_volumes or def_key in assemblies:
            return def_key

        shape = shape_tool.GetShape(def_label)
        bodies = solid_bodies_of(shape)

        # A leaf label may hold several bodies; each becomes a volume the label places once.
        if len(bodies) > 1:
            assemblies.add(def_key)
            kept_bodies = 0
            for i, body in enumerate(bodies):
                body_key = f"{def_key}#b{i + 1}"
                if body_key not in def_names:
                    def_names[body_key] = nm
                if _register_leaf_shape(body_key, body, meshparam, scale_to_cm,
                                        clip_enabled, clip_box, clip_box_shape,
                                        world_trsf, def_lid):
                    placements.append((def_key, body_key, gp_Trsf()))
                    kept_bodies += 1
            if kept_bodies == 0:
                assemblies.discard(def_key)
                return None
            return def_key

        if not bodies and _shape_is_empty(shape):
            print(f"WARNING: CAD leaf {def_lid} ('{nm}') carries no geometry at all "
                  f"(empty compound); skipping it.")
            return None

        if not _register_leaf_shape(def_key, shape, meshparam, scale_to_cm,
                                    clip_enabled, clip_box, clip_box_shape,
                                    world_trsf, def_lid):
            return None
        return def_key

    assemblies.add(def_key)
    return def_key


# -------------------------------
# Coincident placements: one definition, one world transform, ONE placement
# -------------------------------
#
# The rule keys on the (definition, world transform) pair only: two placements of a definition at
# different transforms are instancing, and both stay.

_PLACEMENT_SIG_DIGITS = 9


def trsf_signature(trsf: gp_Trsf, ndigits: int = _PLACEMENT_SIG_DIGITS) -> tuple:
    """A hashable stand-in for a world transform: the 12 matrix entries, rounded.

    Rounding can only cost a missed duplicate, which leaves geometry where the CAD put it; two
    distinct placements are never within 1e-9 model units.
    """
    return tuple(round(trsf.Value(r, c), ndigits) for r in range(1, 4) for c in range(1, 5))


_IDENTITY_TRSF_SIG = trsf_signature(gp_Trsf())


def _placement_children(placements_list) -> Dict[str, List[tuple]]:
    """parent def key -> [(edge index, child def key, local transform), ...], in emission order."""
    kids: Dict[str, List[tuple]] = {}
    for idx, (parent, child, trsf) in enumerate(placements_list):
        kids.setdefault(parent, []).append((idx, child, trsf))
    return kids


def enumerate_occurrences(placements_list, tops, suppressed=frozenset(), limit=8_000_000):
    """Every occurrence the geometry would contain, WITH multiplicity (independent of the dedup).

    Returns a list of (def_key, world transform signature) in depth-first order.
    """
    kids = _placement_children(placements_list)
    out: List[tuple] = []
    stack = [(top, gp_Trsf()) for top in sorted(tops, reverse=True)]
    while stack:
        key, world = stack.pop()
        out.append((key, trsf_signature(world)))
        if len(out) > limit:
            raise RuntimeError(
                f"assembly graph expands past {limit} occurrences; it is probably cyclic")
        for idx, child, trsf in reversed(kids.get(key, ())):
            if idx in suppressed:
                continue
            stack.append((child, _compose_trsf(world, trsf)))
    return out


def _walk_distinct_occurrences(kids, tops, suppressed):
    """Depth-first over DISTINCT (def_key, world signature) occurrences.

    Returns (seen, discoverer); marking at visit time keeps the deep structure, not a flat root.
    """
    seen: Dict[tuple, gp_Trsf] = {}
    discoverer: Dict[tuple, int] = {}
    stack = [(top, gp_Trsf(), _IDENTITY_TRSF_SIG, -1) for top in sorted(tops, reverse=True)]
    while stack:
        key, world, sig, via = stack.pop()
        if (key, sig) in seen:
            continue
        seen[(key, sig)] = world
        if via >= 0:
            discoverer[(key, sig)] = via
        for idx, child, trsf in reversed(kids.get(key, ())):
            if idx in suppressed:
                continue
            cworld = _compose_trsf(world, trsf)
            stack.append((child, cworld, trsf_signature(cworld), idx))
    return seen, discoverer


def _occurrences_below(kids, start_def, start_world, suppressed) -> set:
    """Every (def_key, world signature) placed strictly BELOW this occurrence."""
    visited = set()
    stack = [(start_def, start_world, True)]
    while stack:
        key, world, is_start = stack.pop()
        if not is_start:
            sig = trsf_signature(world)
            if (key, sig) in visited:
                continue
            visited.add((key, sig))
        for idx, child, trsf in kids.get(key, ()):
            if idx in suppressed:
                continue
            stack.append((child, _compose_trsf(world, trsf), False))
    return visited


def deduplicate_placements(placements_list, tops, leaf_keys):
    """Suppress the placement edges that would build one definition twice in the same place.

    Rule 1 drops a root child that a sibling root child already places at the same transform.
    Rule 2 drops an edge only when EVERY one of its occurrences coincides with another edge's; a
    partly coincident edge is reported and kept.

    Returns (kept placements, report dict, emitted leaf occurrences).
    """
    kids = _placement_children(placements_list)
    suppressed: set = set()
    by_rule: Dict[str, List[int]] = {"root-containment": [], "coincident-occurrence": []}

    # --- rule 1: a root child that another root child already contains -----------------------
    for top in sorted(tops):
        siblings = kids.get(top, ())
        holders: Dict[tuple, List[int]] = {}          # occurrence strictly below sibling p -> [p]
        for p, (_idx, child, trsf) in enumerate(siblings):
            for occ in _occurrences_below(kids, child, trsf, suppressed):
                holders.setdefault(occ, []).append(p)
        for jp, (jdx, jchild, jtrsf) in enumerate(siblings):
            owners = holders.get((jchild, trsf_signature(jtrsf)), ())
            if any(p != jp and siblings[p][0] not in suppressed for p in owners):
                suppressed.add(jdx)
                by_rule["root-containment"].append(jdx)

    # --- rule 2: whatever is left that is still coincident, to a fixed point ------------------
    partial: Dict[int, tuple] = {}
    for _ in range(64):
        seen, discoverer = _walk_distinct_occurrences(kids, tops, suppressed)
        total = [0] * len(placements_list)
        for (key, _sig), world in seen.items():
            for idx, _child, _trsf in kids.get(key, ()):
                if idx not in suppressed:
                    total[idx] += 1
        kept = [0] * len(placements_list)
        for idx in discoverer.values():
            kept[idx] += 1
        newly, partial = set(), {}
        for idx in range(len(placements_list)):
            if idx in suppressed or total[idx] == 0:
                continue
            if kept[idx] == 0:
                newly.add(idx)
            elif kept[idx] < total[idx]:
                partial[idx] = (kept[idx], total[idx])
        if not newly:
            break
        suppressed |= newly
        by_rule["coincident-occurrence"].extend(sorted(newly))
    else:                                                        # pragma: no cover - pathological
        raise RuntimeError("coincident-placement de-duplication did not converge")

    declared = [occ for occ in enumerate_occurrences(placements_list, tops) if occ[0] in leaf_keys]
    emitted = [occ for occ in enumerate_occurrences(placements_list, tops, suppressed)
               if occ[0] in leaf_keys]
    kept_placements = [p for idx, p in enumerate(placements_list) if idx not in suppressed]
    report = {
        "declared_leaf_placements": len(declared),
        "distinct_leaf_placements": len(set(declared)),
        "emitted_leaf_placements": len(emitted),
        "declared_multiplicity": dict(sorted(Counter(Counter(declared).values()).items())),
        "emitted_multiplicity": dict(sorted(Counter(Counter(emitted).values()).items())),
        "suppressed_edges": [(placements_list[i][0], placements_list[i][1], rule)
                             for rule, idxs in by_rule.items() for i in sorted(idxs)],
        "n_suppressed_by_rule": {rule: len(idxs) for rule, idxs in by_rule.items()},
        "partial_edges": [(placements_list[i][0], placements_list[i][1]) + v
                          for i, v in sorted(partial.items())],
    }
    return kept_placements, report, emitted


def report_duplicate_placements(report: dict, names: Optional[Dict[str, str]] = None) -> None:
    """Say it out loud, every run. A model that declares coincident duplicates is telling us
    something about the CAD, and silence here would hide the next one."""
    names = names or {}
    n_sup = sum(report["n_suppressed_by_rule"].values())
    declared, distinct = report["declared_leaf_placements"], report["distinct_leaf_placements"]
    if n_sup == 0 and declared == distinct:
        print(f"Placement check: {declared} leaf placement(s), all at distinct world transforms.")
        return

    print(f"WARNING: this CAD model DECLARES {declared - distinct} leaf solid placement(s) that "
          f"coincide exactly with another placement of the same solid.")
    print(f"  The assembly structure in the file says so -- these are not an artefact of this "
          f"traversal, which walks the STEP product structure edge for edge.")
    print(f"  Leaf placements: {declared} declared "
          f"(multiplicity {report['declared_multiplicity']}) -> {distinct} distinct.")
    print(f"  Suppressed {n_sup} placement edge(s) so that no definition is built twice at the "
          f"same world transform "
          f"({', '.join(f'{n} by {rule}' for rule, n in report['n_suppressed_by_rule'].items())}):")
    for parent, child, rule in report["suppressed_edges"]:
        pn, cn = names.get(parent, "") or parent, names.get(child, "") or child
        print(f"    dropped {pn} -> {cn}   [{rule}]")
    print(f"  Emitting {report['emitted_leaf_placements']} leaf placement(s) "
          f"(multiplicity {report['emitted_multiplicity']}).")
    for parent, child, kept, total in report["partial_edges"]:
        pn, cn = names.get(parent, "") or parent, names.get(child, "") or child
        print(f"  WARNING: {pn} -> {cn} is coincident for {total - kept} of its {total} instances "
              f"and NOT suppressed: the placement graph is keyed by definition, so dropping it "
              f"would delete the {kept} instance(s) that are needed.")


def verify_placement_invariant(placements_list, tops, leaf_keys, occurrences=None) -> dict:
    """The permanent check: leaf placements in == out, and no two share definition and transform.

    Raises rather than warns. `occurrences` is the leaf occurrence list when the caller has it.
    """
    occ = (occurrences if occurrences is not None else
           [o for o in enumerate_occurrences(placements_list, tops) if o[0] in leaf_keys])
    multiplicity = Counter(Counter(occ).values())
    if set(multiplicity) - {1}:
        worst = Counter(occ).most_common(1)[0]
        raise RuntimeError(
            f"placement invariant violated: {len(occ)} leaf placements hold only "
            f"{len(set(occ))} distinct (definition, world transform) pairs "
            f"(multiplicity {dict(sorted(multiplicity.items()))}); e.g. {worst[0][0]} is placed "
            f"{worst[1]} times at the same world matrix")
    return {"leaf_placements": len(occ), "multiplicity": dict(sorted(multiplicity.items()))}


def extract_graph(
    step_path: str,
    meshparam=None,
    scale_to_cm: float = 1.0,
    clip_box: Optional[ClipBox] = None,
    clip_deduplicate: str = "intact",
    name_filter: Optional[NameFilter] = None,
):
    reset_graph()
    doc, shape_tool = load_step_with_xcaf(step_path)
    expand_free_shapes(
        shape_tool,
        meshparam=meshparam,
        scale_to_cm=scale_to_cm,
        clip_box=clip_box,
        clip_deduplicate=clip_deduplicate,
        name_filter=name_filter,
    )
    return doc, shape_tool


def expand_free_shapes(
    shape_tool,
    meshparam=None,
    scale_to_cm: float = 1.0,
    clip_box: Optional[ClipBox] = None,
    clip_deduplicate: str = "intact",
    name_filter: Optional[NameFilter] = None,
):
    """Expand every XCAF free shape into the definition graph, then make the placements unique."""
    global placements
    clip_box_shape = make_clip_box_shape(clip_box) if clip_box is not None else None

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)

    for i in range(roots.Length()):
        root = roots.Value(i + 1)
        root_occ_path = f"r{i + 1}"
        if shape_tool.IsReference(root):
            ref = TDF_Label()
            shape_tool.GetReferredShape(root, ref)
            root = ref
        top = expand_definition(
            root,
            shape_tool,
            meshparam=meshparam,
            scale_to_cm=scale_to_cm,
            clip_box=clip_box,
            clip_box_shape=clip_box_shape,
            clip_deduplicate=clip_deduplicate,
            name_filter=name_filter,
            occ_path=root_occ_path,
        )
        if top is not None:
            top_defs.add(top)

    placements, dup_report, emitted = deduplicate_placements(placements, top_defs,
                                                             set(logical_volumes))
    report_duplicate_placements(dup_report, def_names)
    verify_placement_invariant(placements, top_defs, set(logical_volumes), emitted)
    return dup_report


# -------------------------------
# ROOT macro emission
# -------------------------------

def emit_nested_placement_cpp(body_def: str, child_def: str, trsf: gp_Trsf, copy_no: int,
                              scale_to_cm: float, csg_lids: Optional[set] = None) -> str:
    """One `AddNode` of a daughter INTO its mother's body volume, not beside it.

    A TGeo daughter takes precedence over its mother's solid, so this restores the source nesting
    with no boolean. A child at T in the assembly frame is at `P^-1 * T` in the body volume's frame.
    """
    body_cpp = cpp_var_for_def(body_def)
    child_cpp = cpp_var_for_def(child_def)
    tr_name = f"trn_{sanitize_cpp_name(body_def)}_{sanitize_cpp_name(child_def)}_{copy_no}"
    out = trsf_to_tgeo(trsf, tr_name, scale_to_cm)
    node_matrix = tr_name

    hook = import_csg_hook()
    if csg_lids and child_def in csg_lids:
        node_matrix = f"{tr_name}_placed"
        out += hook.emit_csg_composed_placement_cpp(
            tr_name, hook.csg_placement_var(child_def, sanitize_cpp_name), node_matrix) + "\n"
    if csg_lids and body_def in csg_lids:
        inv = f"{tr_name}_inbody"
        pvar = hook.csg_placement_var(body_def, sanitize_cpp_name)
        out += (f"  TGeoHMatrix *{inv} = new TGeoHMatrix({pvar}->Inverse());\n"
                f"  {inv}->Multiply({node_matrix});\n")
        node_matrix = inv
    return out + f"  {body_cpp}->AddNode({child_cpp}, {copy_no}, {node_matrix});\n"


def emit_placement_cpp(parent_def: str, child_def: str, trsf: gp_Trsf, copy_no: int, scale_to_cm: float,
                       csg_lids: Optional[set] = None) -> str:
    """One `AddNode`, with the child's own shape placement composed in when it has one.

    The node matrix is `partPlacement * shapePlacement`, emitted for every placement of the child.
    """
    parent_cpp = cpp_var_for_def(parent_def)
    child_cpp = cpp_var_for_def(child_def)
    tr_name = f"tr_{sanitize_cpp_name(parent_def)}_{sanitize_cpp_name(child_def)}_{copy_no}"
    out = trsf_to_tgeo(trsf, tr_name, scale_to_cm)
    node_matrix = tr_name
    if csg_lids and child_def in csg_lids:
        hook = import_csg_hook()
        node_matrix = f"{tr_name}_placed"
        out += hook.emit_csg_composed_placement_cpp(
            tr_name, hook.csg_placement_var(child_def, sanitize_cpp_name), node_matrix) + "\n"
    return out + f"  {parent_cpp}->AddNode({child_cpp}, {copy_no}, {node_matrix});\n"



def _compute_density_g_cm3(
    volume_cm3: float,
    mass_value: float,
    mass_unit: str,
) -> Tuple[Optional[float], str]:
    """
    Computes an effective part density from (mass, CAD volume).

    Returns (rho_g_cm3 or None, comment). If rho is None, caller should fall back
    to the Geant4 NIST density (if resolved) or to a dummy density.
    """
    if not volume_cm3 or volume_cm3 <= 0:
        return None, "no CAD volume available for density"

    if (mass_value is None) or (isinstance(mass_value, float) and math.isnan(mass_value)):
        return None, "no BOM mass available for density"

    mass_g = float(mass_value)
    mu = (mass_unit or "kg").lower()
    if mu == "kg":
        mass_g *= 1000.0
    elif mu == "g":
        pass
    else:
        # unknown unit: assume kg
        mass_g *= 1000.0

    rho = mass_g / float(volume_cm3)
    # Guard against obvious unit/volume issues
    if not (0.01 < rho < 50.0):
        return None, f"computed density {rho:.3g} g/cm3 rejected (unit mismatch?)"

    return rho, "density from BOM mass and CAD volume"


def emit_root_macro(
    step_path: str,
    out_folder: _Path,
    meshparam=None,
    step_unit: str = "auto",
    clip_box: Optional[ClipBox] = None,
    clip_deduplicate: str = "intact",
    name_filter: Optional[NameFilter] = None,
    materials_csv: Optional[str] = None,
    media_json: Optional[str] = None,
    in_field: Optional[Tuple[float, float]] = None,
    bom_mass_unit: str = "kg",
    g4_nist_json: Optional[str] = None,
    mat_cfg: Optional[MatMatchConfig] = None,
    surface_report: Optional[str] = None,
    exact_surfaces: str = "off",
    recognize_surfaces: str = "exact",
    dump_brep: bool = False,
    csg: str = "off",
    csg_report: Optional[str] = None,
    max_cells: Optional[int] = None,
    max_splits: Optional[int] = None,
    decompose_timeout: Optional[float] = None,
    mesh_solid: str = "o2",
):
    # exact_surfaces mode:
    #   off      : tessellated output only (default; leaves generated output unchanged).
    #   auto     : emit O2BVHSurfaceSolid for every leaf solid whose faces all extract
    #              exactly, tessellated fallback otherwise.
    #   required : like auto, but abort if any leaf solid cannot be represented exactly.
    #
    # dump_brep: also write brep_<VOLNAME>_<LID>.brep, scaled to cm, next to each surfaces_*.bin.
    if (step_unit or "auto").lower() == "auto":
        detected = detect_step_length_unit(step_path)
        scale_to_cm = step_unit_scale_to_cm(detected)
        print(f"Detected STEP length unit: {detected} (scale to cm = {scale_to_cm})")
    else:
        scale_to_cm = step_unit_scale_to_cm(step_unit)
        print(f"Using overridden STEP length unit: {step_unit} (scale to cm = {scale_to_cm})")

    if clip_box is not None:
        print(f"Clipping CAD geometry to STEP-coordinate bounding box: {clip_box.as_tuple()}")
        print(f"Clip deduplication mode: {clip_deduplicate}")

    if name_filter is not None and name_filter.active:
        print(f"CAD name filters: {len(name_filter.include)} include regex(es), {len(name_filter.exclude)} exclude regex(es)")

    extract_graph(
        step_path,
        meshparam=meshparam,
        scale_to_cm=scale_to_cm,
        clip_box=clip_box,
        clip_deduplicate=clip_deduplicate,
        name_filter=name_filter,
    )

    out_folder = out_folder.expanduser().resolve()
    out_folder.mkdir(parents=True, exist_ok=True)

    recognize_mode = (recognize_surfaces or "exact").lower()
    recognize_flag = recognize_mode == "exact"

    # --- optional exact-surface eligibility report (does not modify the emitted geometry) ---
    surface_report_data = None
    surface_report_path = None
    recognition: Dict[tuple, Optional[dict]] = {}   # (lid, face index) -> recognizer result
    if surface_report:
        surface_report_data = build_surface_report(step_path, scale_to_cm,
                                                   recognize_surfaces=recognize_flag,
                                                   recognition=recognition)
        surface_report_path = _Path(surface_report).expanduser().resolve()
        surface_report_path.parent.mkdir(parents=True, exist_ok=True)
        surface_report_path.write_text(json.dumps(surface_report_data, indent=1))
        summ = surface_report_data["summary"]
        print(f"Surface report: {summ['n_eligible']}/{summ['n_volumes']} logical volumes eligible "
              f"for exact O2BVHSurfaceSolid conversion")
        print(f"  face types: {summ['face_type_counts']}")
        if summ["recognized_surface_counts"]:
            print(f"  recognized (stored type is not the geometry): {summ['recognized_surface_counts']}"
                  f" recovered from stored {summ['recognized_stored_type_counts']}")
        if summ["fallback_reasons"]:
            top = sorted(summ["fallback_reasons"].items(), key=lambda kv: -kv[1])[:5]
            for reason, count in top:
                print(f"  fallback ({count}x): {reason}")
        print(f"Wrote surface report: {surface_report_path}")

    # --- exact-surface extraction (auto/required modes) ---
    exact_mode = (exact_surfaces or "off").lower()
    scaled_shapes: Dict[str, object] = {}  # def_lid -> the cm copy written for --dump-brep
    surface_files: Dict[str, str] = {}     # def_lid -> absolute path of its surfaces_*.bin
    if exact_mode != "off":
        brep_files: Dict[str, str] = {}  # def_lid -> absolute path of brep_*.brep (--dump-brep)
        failures: Dict[str, List[str]] = {}  # def_lid -> unsupported-face reasons
        extracted: Dict[str, int] = {}   # def_lid -> number of surface records written
        for lid, shape in def_shapes.items():
            surfaces, reasons, n_model_edges = extract_surfaces_for_shape(
                shape, scale_to_cm, recognize_surfaces=recognize_flag, recognition=recognition,
                lid=lid)
            if surfaces is None:
                failures[lid] = reasons
                continue
            extracted[lid] = len(surfaces)
            disp = def_names.get(lid, "")
            volname = sanitize_filename(disp) if disp else "vol"
            name_suffix = f"{volname}_{sanitize_filename(lid)}"
            fpath = (out_folder / f"surfaces_{name_suffix}.bin").resolve()
            write_surfaces_bin(fpath, surfaces, accept.model_tolerance_cm(shape) * scale_to_cm,
                               n_model_edges)
            surface_files[lid] = str(fpath)
            if dump_brep:
                bpath = (out_folder / f"brep_{name_suffix}.brep").resolve()
                scaled_shapes[lid] = write_brep_cm(bpath, shape, scale_to_cm)
                brep_files[lid] = str(bpath)
        if dump_brep:
            print(f"Wrote {len(brep_files)} reference BREP file(s) (brep_*.brep, scaled to cm)")
        n_leaf = len(def_shapes)
        print(f"Exact-surface extraction ({exact_mode}): {len(surface_files)}/{n_leaf} leaf solids "
              f"represented exactly, {len(failures)} fall back to tessellation")
        reason_counts: Dict[str, int] = {}
        if failures:
            # Aggregate reasons for a compact, useful report.
            for reasons in failures.values():
                for r in reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
            for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
                print(f"  fallback ({count} face(s)): {reason}")
            if exact_mode == "required":
                lines = [f"--exact-surfaces required: {len(failures)}/{n_leaf} leaf solid(s) cannot be "
                         f"represented exactly:"]
                for lid in sorted(failures):
                    name = def_names.get(lid, "") or lid
                    uniq = sorted(set(failures[lid]))
                    lines.append(f"  {name} [{lid}]: {'; '.join(uniq)}")
                raise ValueError("\n".join(lines))

        # `eligible` is a claim about surfaces only; `emitted` is what extraction actually did.
        if surface_report_data is not None:
            n_emitted_rescued = 0
            for lid, vol in surface_report_data["volumes"].items():
                emitted_here = lid in extracted
                vol["emitted"] = emitted_here
                if not emitted_here:
                    vol["extraction_reasons"] = sorted(set(failures.get(lid, [])))
                    # The extractor's verdict supersedes the classification pass's optimistic
                    # one: this is the reason the sidecar was actually not written.
                    vol["why_not_surface"] = (distill_reasons(failures.get(lid, []))
                                              or vol.get("why_not_surface")
                                              or "no sidecar was written")
                else:
                    # The part has a sidecar; whatever the classification pass guessed, there
                    # is no "why not".
                    vol["why_not_surface"] = None
                    if vol["recognized_counts"]:
                        n_emitted_rescued += 1
            summary = surface_report_data["summary"]
            summary["n_emitted"] = len(extracted)
            summary["n_emitted_carrying_recognized_faces"] = n_emitted_rescued
            summary["n_eligible_but_not_emitted"] = sum(
                1 for v in surface_report_data["volumes"].values()
                if v["eligible"] and not v["emitted"])
            summary["extraction_fallback_reasons"] = reason_counts if failures else {}
            surface_report_path.write_text(json.dumps(surface_report_data, indent=1))
            print(f"  emitted {summary['n_emitted']}/{summary['n_volumes']}; "
                  f"{summary['n_eligible_but_not_emitted']} surface-eligible solid(s) declined at "
                  f"extraction; {summary['n_emitted_carrying_recognized_faces']} emitted solid(s) "
                  f"carry recognized faces")

    # --- CSG recognition (--csg auto|required) -- the one CSG hook ------------------------
    # Only an accepted part is emitted as a native ROOT shape; every representation is still written.
    csg_mode = (csg or "off").lower()
    csg_files: Dict[str, str] = {}
    # flatcsg_*.bin per O2FlatCSG part, disjoint from csg_files.
    flat_files: Dict[str, str] = {}
    if csg_mode != "off":
        hook = import_csg_hook()
        # The budgets live as module constants in cadsupport/decompose.py and are read at call time by
        # cadsupport/recognise.py, so setting them here is enough and nothing has to be threaded.
        if any(v is not None for v in (max_cells, max_splits, decompose_timeout)):
            from cadsupport import decompose as _decomp
            if max_cells is not None:
                print(f"  cell budget raised: {_decomp.PART_MAX_CELLS} -> {max_cells}")
                _decomp.PART_MAX_CELLS = max_cells
            if max_splits is not None:
                print(f"  split budget raised: {_decomp.MAX_SPLITS} -> {max_splits}")
                _decomp.MAX_SPLITS = max_splits
            if decompose_timeout is not None:
                print(f"  decomposition timeout raised: {_decomp.TIMEOUT_S} -> "
                      f"{decompose_timeout} s")
                _decomp.TIMEOUT_S = decompose_timeout
        csg_files, flat_files, csg_records = hook.recognise_and_emit(
            def_shapes, def_names, scale_to_cm, out_folder, sanitize_filename, mode=csg_mode,
            scaled=scaled_shapes)
        csg_report_path = _Path(csg_report) if csg_report else (out_folder / "csg_report.json")
        # The lid -> sidecar mapping lets write_report compute tessellation exactness.
        csg_report_data = hook.write_report(csg_records, csg_report_path, dict(surface_files),
                                            set(logical_volumes))
        hook.print_tier_table(csg_report_data)
        print(f"Wrote CSG report: {csg_report_path}")

    # --- Geant4 NIST material DB (optional but recommended) ---
    g4db: Optional[Dict[str, dict]] = None
    if g4_nist_json:
        g4db = load_g4_nist_db(g4_nist_json)
        print(f"Loaded Geant4 NIST DB with {len(g4db)} materials from: {g4_nist_json}")
    else:
        print("No --g4-nist-json provided: unresolved materials will fall back to dummy ROOT materials.")
    mat_cfg = mat_cfg or MatMatchConfig()


    # --- BOM: map volumes to materials (heuristic) ---
    lid_to_bom: Dict[str, BomEntry] = {}
    if materials_csv:
        bom_entries = read_bom_csv(materials_csv)
        lid_to_bom = build_volume_to_material_map(bom_entries, def_names)
        print(f"Loaded {len(bom_entries)} BOM entries from: {materials_csv}")
        print(f"Matched {len(lid_to_bom)} CAD logical volumes to BOM entries (by name/part-number heuristics)")
    else:
        print("No --materials-csv provided: emitting Default medium for all logical volumes")

    # --- media sidecar: the exact media of the geometry this STEP came from ---
    media_sidecar: Optional[dict] = None
    if media_json:
        with open(media_json) as _fh:
            media_sidecar = json.load(_fh)
        print(f"Loaded media sidecar: {media_sidecar.get('nMedia')} media over "
              f"{media_sidecar.get('nParts')} parts from {media_json}")

    # --- facet files ---
    facet_files = {}  # def_lid -> absolute path string
    for lid, tris in logical_volumes.items():
        disp = def_names.get(lid, "")
        volname = sanitize_filename(disp) if disp else "vol"
        lidname = sanitize_filename(lid)
        fname = f"facets_{volname}_{lidname}.bin"
        fpath = (out_folder / fname).resolve()
        write_facets_bin(fpath, tris)
        facet_files[lid] = str(fpath).replace("\\", "\\\\")  # C++ string literal safety

    # --- which materials do we need to emit? ---
    
    # --- materials: collect unique BOM material strings actually used by leaf volumes ---
    # We resolve each unique BOM string to a Geant4 NIST material using string + density scoring.
    used_materials: Dict[str, ResolvedMaterial] = {}

    # Precompute one representative part density per BOM material (first good value wins)
    mat_to_rho: Dict[str, Optional[float]] = {}
    mat_to_rho_note: Dict[str, str] = {}

    for lid in logical_volumes.keys():
        if lid not in lid_to_bom:
            continue
        bom = lid_to_bom[lid]
        mat_name = normalize_material_name(bom.material)

        if mat_name not in mat_to_rho:
            rho_part, rho_note = _compute_density_g_cm3(
                _leaf_volume_cm3(lid, scale_to_cm),
                bom.mass_value,
                bom_mass_unit,
            )
            mat_to_rho[mat_name] = rho_part
            mat_to_rho_note[mat_name] = rho_note

    for mat_name in sorted(mat_to_rho.keys(), key=lambda s: s.lower()):
        rho_part = mat_to_rho.get(mat_name)
        rm = resolve_bom_material(mat_name, rho_part, g4db, mat_cfg)

        # Fold density provenance into the note for geom.C comments
        rm.note = f"{rm.note} (density: {mat_to_rho_note.get(mat_name, 'n/a')})"

        if rm.nist_name is None:
            print(f"WARNING: Unresolved/ambiguous material '{mat_name}'. See FIXME in generated geom.C.")

        used_materials[mat_name] = rm

    if media_sidecar is not None:
        materials_cpp, medium_var_map = emit_media_sidecar_cpp(media_sidecar)
    else:
        materials_cpp, medium_var_map = emit_materials_cpp(used_materials, in_field=in_field)

    # --- emit C++ macro ---
    if surface_files:
        print(f"Emitting {len(surface_files)}/{len(logical_volumes)} logical volumes as exact O2BVHSurfaceSolid "
              f"(macro requires the ALICE O2 environment)")

    # The tessellated fallback's shape class; "tgeo" navigates as bounding boxes.
    if mesh_solid not in ("o2", "tgeo"):
        raise ValueError(f"mesh_solid must be 'o2' or 'tgeo', got {mesh_solid!r}")
    tess_lids = [lid for lid in logical_volumes
                 if lid not in flat_files and lid not in csg_files and lid not in surface_files
                 and len(logical_volumes[lid]) > 0]
    solid_class = "o2::base::O2Tessellated" if mesh_solid == "o2" else "TGeoTessellated"
    if tess_lids:
        if mesh_solid == "o2":
            print(f"Emitting {len(tess_lids)}/{len(logical_volumes)} logical volumes as navigable "
                  f"o2::base::O2Tessellated (macro requires the ALICE O2 environment)")
        else:
            print(f"  [WARN] --mesh-solid tgeo: {len(tess_lids)}/{len(logical_volumes)} logical volume(s) "
                  f"are emitted as ROOT TGeoTessellated, which implements no navigation of its own and "
                  f"inherits Contains/DistFrom*/Safety from TGeoBBox. Every one of them will be navigated "
                  f"as its bounding box, filled. Use --mesh-solid o2 for a geometry meant to be traversed.")

    cpp: List[str] = []
    cpp.append(emit_cpp_prelude(exact_surfaces=bool(surface_files), csg_shapes=bool(csg_files),
                                flat_csg_shapes=bool(flat_files),
                                o2_tessellated=bool(tess_lids) and mesh_solid == "o2",
                                in_field=in_field is not None))

    _media_unresolved: List[Tuple[str, str]] = []   # part named a medium the sidecar lacks
    _media_unnamed: List[str] = []                  # part the sidecar does not name at all
    cpp.append("TGeoVolume* build(bool check=true) {")
    cpp.append('  if (!gGeoManager) { throw std::runtime_error("gGeoManager is null. Call build_and_export(), or create a TGeoManager yourself before calling build() directly: new TGeoManager(\\"geom\\",\\"geom\\");"); }')
    cpp.append(materials_cpp)

    for lid in logical_volumes.keys():
        ntriangles = len(logical_volumes[lid])

        # choose medium for this volume
        med = "med_Default"
        if media_sidecar is not None:
            # The sidecar keys on the emitted STEP part name, which is exactly the
            # display name the reader recovered for this definition.
            part = def_names.get(lid, "")
            medname = media_sidecar.get("parts", {}).get(part)
            if medname:
                med = medium_var_map.get(medname, "med_Default")
                if med == "med_Default":
                    _media_unresolved.append((part, medname))
            else:
                _media_unnamed.append(part)
        elif lid in lid_to_bom:
            mat_name = normalize_material_name(lid_to_bom[lid].material)
            med = medium_var_map.get(mat_name, "med_Default")

        # The cascade, in one place: CSG, else exact surfaces, else the tessellated fallback.
        if lid in flat_files:
            sidecar = str(_Path(flat_files[lid]).expanduser().resolve()).replace("\\", "\\\\")
            cpp.append(import_csg_hook().emit_flat_csg_shape_cpp(
                lid, def_names.get(lid, ""), sidecar, med, sanitize_cpp_name))
        elif lid in csg_files:
            shape_path = str(_Path(csg_files[lid]).expanduser().resolve()).replace("\\", "\\\\")
            cpp.append(import_csg_hook().emit_csg_shape_cpp(
                lid, def_names.get(lid, ""), shape_path, med, sanitize_cpp_name))
        elif lid in surface_files:
            sidecar = str(_Path(surface_files[lid]).expanduser().resolve()).replace("\\", "\\\\")
            cpp.append(emit_surface_solid_cpp(lid, def_names.get(lid, ""), sidecar, med))
        else:
            cpp.append(emit_tessellated_cpp(lid, def_names.get(lid, ""), facet_files[lid], ntriangles, med,
                                            solid_class=solid_class))

    if media_sidecar is not None:
        nvol = len(logical_volumes)
        nresolved = nvol - len(_media_unnamed) - len(_media_unresolved)
        print(f"Media from sidecar: {nresolved}/{nvol} volumes carry their source medium")
        if _media_unnamed:
            print(f"  [WARN] {len(_media_unnamed)} volume(s) are not named by the sidecar "
                  f"and fall back to Default (transparent): {_media_unnamed[:5]}")
        if _media_unresolved:
            print(f"  [WARN] {len(_media_unresolved)} volume(s) name a medium the sidecar "
                  f"does not define: {_media_unresolved[:5]}")

    for lid in sorted(assemblies):
        cpp.append(emit_assembly_cpp(lid, def_names.get(lid, "")))

    csg_lids = set(csg_files)

    # Which emitted part is the body of which assembly, from the writer's sidecar.
    # Keyed by def id here, because that is what the placement edges carry.
    body_of = {}          # assembly def id -> its body def id
    if media_sidecar is not None:
        name_to_lid = {}
        for _lid, _nm in def_names.items():
            if _nm:
                name_to_lid.setdefault(_nm, []).append(_lid)
        # A completely carved mother must NOT be nested: carving or nesting, one per mother.
        carved_complete = media_sidecar.get("carvedComplete") or {}
        _skipped_carved = 0
        for bodyname, asmname in (media_sidecar.get("bodyOfAssembly") or {}).items():
            if carved_complete.get(asmname):
                _skipped_carved += 1
                continue
            blids, alids = name_to_lid.get(bodyname, []), name_to_lid.get(asmname, [])
            if len(blids) == 1 and len(alids) == 1:
                body_of[alids[0]] = blids[0]
            elif blids and alids:
                # An ambiguous name would nest a mother's daughters into the wrong
                # body, so refuse rather than guess.
                print(f"  [WARN] not nesting {asmname}: {len(alids)} definition(s) "
                      f"of that name and {len(blids)} of {bodyname}")

    if media_sidecar is not None and (media_sidecar.get("carvedComplete") or {}):
        print(f"Carving: {_skipped_carved} mother(s) were carved completely and are left "
              f"flat; {len(body_of)} were not and keep their nesting")
    _nested = 0
    for idx, (parent, child, trsf) in enumerate(placements, start=1):
        body = body_of.get(parent)
        if body is not None and child != body:
            cpp.append(emit_nested_placement_cpp(body, child, trsf, idx, scale_to_cm, csg_lids))
            _nested += 1
        else:
            cpp.append(emit_placement_cpp(parent, child, trsf, idx, scale_to_cm, csg_lids))
    if media_sidecar is not None:
        print(f"Mother nesting: {_nested} of {len(placements)} placement(s) go inside "
              f"their mother's body volume ({len(body_of)} assembly/assemblies with a body)")

    # A top-level CSG volume gets a one-node assembly to carry its shape placement.
    placed_tops = sorted(lid for lid in top_defs if lid in csg_lids)
    if len(top_defs) == 1 and not placed_tops:
        top = next(iter(top_defs))
        cpp.append(f"  return {cpp_var_for_def(top)};")
    else:
        hook = import_csg_hook() if placed_tops else None
        cpp.append('  TGeoVolumeAssembly *asm_WORLD = new TGeoVolumeAssembly("WORLD");')
        for i, node in enumerate(sorted(top_defs), start=1):
            if node in csg_lids:
                cpp.append(f"  asm_WORLD->AddNode({cpp_var_for_def(node)}, {i}, "
                           f"{hook.csg_placement_var(node, sanitize_cpp_name)});")
            else:
                cpp.append(f"  asm_WORLD->AddNode({cpp_var_for_def(node)}, {i});")
        cpp.append("  return asm_WORLD;")

    cpp.append("}")

    # The build_and_export driver; CheckOverlaps runs only with checkOverlaps=true.
    cpp.append('void build_and_export(const char* out_root = "geom.root", bool check=true,')
    cpp.append('                      bool checkOverlaps=false) {')
    cpp.append('  if (!gGeoManager) { new TGeoManager("geom","geom"); }')
    cpp.append('  TGeoVolume* top = build(check);')
    cpp.append('  gGeoManager->SetTopVolume(top);')
    cpp.append('  gGeoManager->CloseGeometry();')
    cpp.append('  if (checkOverlaps) { gGeoManager->CheckOverlaps(); }')
    cpp.append('  gGeoManager->Export(out_root);')
    cpp.append('}')

    # exports a function to get get hold of the builder function in ALICE O2
    cpp.append('std::function<TGeoVolume*()> get_builder_hook_checked() {')
    cpp.append('  return []() { return build(true); };')
    cpp.append('}')
    # exports a function to get get hold of the builder function in ALICE O2
    cpp.append('std::function<TGeoVolume*()> get_builder_hook_unchecked() {')
    cpp.append('  return []() { return build(false); };')
    cpp.append('}')

    return "\n".join(cpp)


# -------------------------------
# Geometry Tree printing (debug)
# -------------------------------

def traverse_print(label, shape_tool, depth=0):
    indent = "  " * depth
    name = label.GetLabelName()
    entry = label_id(label)
    print(f"{indent}- {name}  =>[{entry}]") 

    if shape_tool.IsReference(label):
        ref_label = TDF_Label()
        shape_tool.GetReferredShape(label, ref_label)
        traverse_print(ref_label, shape_tool, depth + 1)
        return

    children = TDF_LabelSequence()
    shape_tool.GetComponents(label, children)
    if children.Length() > 0 or shape_tool.IsAssembly(label):
        for i in range(children.Length()):
            traverse_print(children.Value(i + 1), shape_tool, depth + 1)
        return

    if shape_tool.IsSimpleShape(label):
        shape = shape_tool.GetShape(label)
        print(f"{indent}  [LogicalShape id={id(shape)}]")


def print_geom(step_file):
    print(f"Printing GEOM hierarchy for {step_file}")
    doc, shape_tool = load_step_with_xcaf(step_file)
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    for i in range(roots.Length()):
        traverse_print(roots.Value(i + 1), shape_tool)


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert STEP/XCAF to ROOT TGeo macro, facets in per-volume binary files.")
    ap.add_argument("step", nargs="?", help="Input STEP file (omit with --self-test)")
    ap.add_argument("-o", "--out", default="geom.C", help="Output ROOT macro file name (default: geom.C)")
    ap.add_argument("--output-folder", default="./", help="Output folder for macro + facet files")
    ap.add_argument("--mesh", action="store_true", help="Use full BRepMesh triangulation instead of bounding boxes")
    ap.add_argument("--print-tree", action="store_true", help="Just prints the geometry tree")
    ap.add_argument("--mesh-prec", type=float, default=0.1, help="meshing precision. lower --> slower")
    ap.add_argument("--in-field", nargs="?", const="2,10", default=None, metavar="IFIELD,FIELDM",
                    help="Treat this module as sitting in the magnetic field: write the eight Geant "
                         "medium parameters, with ifield and fieldm taken from the live field. "
                         "IFIELD,FIELDM applies when no field is loaded (default 2,10); step "
                         "control stays at the transport default. BOM/NIST material route only.")
    ap.add_argument("--step-unit", default="auto", choices=["auto", "mm", "cm", "m", "in", "ft"], help="STEP length unit override (default: auto-detect); TGeo expects cm")
    ap.add_argument("--clip-box", nargs=6, type=float, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"), default=None, help="Clip CAD geometry to this axis-aligned bounding box before meshing (coordinates in STEP file units, before conversion to cm)")
    ap.add_argument("--clip-deduplicate", default="intact", choices=["none", "intact"], help="When clipping, reuse original logical definitions for subtrees fully inside the clip box (default: intact); use 'none' for one volume per surviving occurrence")
    ap.add_argument("--include-name", action="append", default=[], help="Only convert CAD labels whose XCAF name or label entry matches this regex; may be repeated. Matching an assembly includes its subtree.")
    ap.add_argument("--exclude-name", action="append", default=[], help="Skip CAD labels/subtrees whose XCAF name or label entry matches this regex; may be repeated.")
    ap.add_argument("--name-filter-case-sensitive", action="store_true", help="Make --include-name/--exclude-name matching case-sensitive (default: case-insensitive)")
    ap.add_argument("--surface-report", default=None, metavar="PATH", help="Write a JSON report classifying each face by analytic surface type and each logical volume by exact O2BVHSurfaceSolid conversion eligibility. Does not change the generated geometry output.")
    ap.add_argument("--mesh-solid", default="o2", choices=["o2", "tgeo"], help="Shape class for the tessellated fallback. 'o2' (default): o2::base::O2Tessellated, which navigates and needs the O2 environment. 'tgeo': ROOT's TGeoTessellated, which navigates as its bounding box; only for a macro that must load outside O2.")
    ap.add_argument("--exact-surfaces", default="off", choices=["off", "auto", "required"], help="Emit exact O2BVHSurfaceSolid shapes, each with a surfaces_*.bin sidecar. 'off' (default): tessellated only. 'auto': exact where every face extracts, tessellated otherwise. 'required': fail if any leaf solid cannot be exact.")
    ap.add_argument("--dump-brep", action="store_true", help="With --exact-surfaces auto|required, also write brep_<VOLNAME>_<LID>.brep (the leaf solid in cm) next to each surfaces_*.bin, for the OCCT reference oracle.")
    ap.add_argument("--csg", default="off", choices=["off", "auto", "required"], help="Emit leaf solids recognised as native ROOT CSG shapes as shape_<VOLNAME>_<LID>.root, when OCCT's symmetric-difference volume against the CAD solid is inside the model tolerance. 'off' (default). 'auto': the per-part cascade CSG -> exact surfaces -> tessellated. 'required': fail if any leaf solid is not CSG. The evidence goes to csg_<VOLNAME>_<LID>.json and csg_report.json.")
    ap.add_argument("--max-cells", type=int, default=None, metavar="N",
                    help="Raise the decomposition's per-part cell budget (default 64), so deeper "
                         "boolean parts can ship as O2FlatCSG.")
    ap.add_argument("--max-splits", type=int, default=None, metavar="N",
                    help="Raise the decomposition's split budget (default 256). A raised cell "
                         "budget usually needs this too, since every cell costs a split.")
    ap.add_argument("--decompose-timeout", type=float, default=None, metavar="S",
                    help="Raise the per-part decomposition timeout in seconds (default 60).")
    ap.add_argument("--csg-report", default=None, metavar="PATH", help="Where to write the per-part CSG cascade report (default: csg_report.json in the output folder).")
    ap.add_argument("--recognize-surfaces", default="exact", choices=["exact", "off"], help="Recover the exact plane/sphere/cylinder/cone behind a stored bspline/bezier/revolution/extrusion face. 'exact' (default): only a fit at machine precision. 'off': keep such faces tessellated. Applies to --surface-report and --exact-surfaces.")

    # BOM / material support
    ap.add_argument("--materials-csv", default=None, help="BOM CSV file providing material + mass per part (optional)")
    ap.add_argument("--media-json", default=None,
                    help="Media sidecar written by O2_TGeoToCAD.py --media-json; rebuilds the "
                         "source media verbatim and takes precedence over --materials-csv.")
    ap.add_argument("--bom-mass-unit", default="kg", choices=["kg", "g"], help="Unit of the BOM mass column (default: kg)")
    ap.add_argument("--g4-nist-json", default=None, help="Path to Geant4 NIST DB JSON dump (from nist_export_all). Enables TGeoMixture emission + RadLen/IntLen.")


    # Material matching scoring knobs (only used if --g4-nist-json is provided)
    ap.add_argument("--mat-min-score", type=float, default=0.35, help="Minimum combined score to accept a G4 NIST material match (default: 0.35)")
    ap.add_argument("--mat-ambiguity-delta", type=float, default=0.05, help="If best-second < delta, treat match as ambiguous/unresolved (default: 0.05)")
    ap.add_argument("--mat-w-token", type=float, default=0.75, help="Weight for token/name similarity score (default: 0.75)")
    ap.add_argument("--mat-w-density", type=float, default=0.25, help="Weight for density proximity score (default: 0.25)")
    ap.add_argument("--mat-max-log-density-diff", type=float, default=0.0, help="Optional hard density filter in log-space (0 disables). Example 0.8 ~ within 2.2x (default: 0.0)")
    ap.add_argument("--mat-compound-penalty", type=float, default=0.25, help="Penalty for matching to oxides/carbides/etc. when BOM doesn't mention them (default: 0.25)")

    ap.add_argument("--self-test", action="store_true", help="Run the converter self-tests (no STEP file needed) and exit non-zero on any failure.")

    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if (run_recognition_self_test() + run_placement_self_test()
                       + run_planar_trim_self_test()
                       + run_duplicate_placement_self_test()
                       + run_multibody_leaf_self_test()
                       + run_in_field_media_self_test()
                       + run_bom_token_self_test()) else 0)
    if args.step is None:
        ap.error("the following arguments are required: step (or pass --self-test)")

    step_path = str(_Path(args.step).expanduser().resolve())
    if args.print_tree:
        print_geom(step_path)
        return

    out_folder = _Path(args.output_folder)

    clip_box = None
    if args.clip_box is not None:
        try:
            clip_box = ClipBox.from_values(args.clip_box)
        except ValueError as exc:
            ap.error(str(exc))

    in_field = None
    if args.in_field is not None:
        try:
            parts = [float(x) for x in str(args.in_field).split(",")]
        except ValueError:
            parts = []
        if len(parts) != 2:
            ap.error("--in-field takes IFIELD,FIELDM (e.g. --in-field 2,10) or no value at all")
        in_field = (parts[0], parts[1])
        print(f"--in-field: media take ifield/fieldm from the live field at build time "
              f"(seed {in_field[0]:g},{in_field[1]:g} if none is loaded); "
              "step control left at the transport default")

    name_filter = None
    if args.include_name or args.exclude_name:
        try:
            name_filter = NameFilter.from_patterns(
                args.include_name,
                args.exclude_name,
                case_sensitive=args.name_filter_case_sensitive,
            )
        except re.error as exc:
            ap.error(f"Invalid CAD name filter regex: {exc}")

    meshparam = {"do_meshing": args.mesh, "lin_defl": args.mesh_prec, "ang_defl": args.mesh_prec}


    mat_cfg = MatMatchConfig(
    min_score=args.mat_min_score,
    ambiguity_delta=args.mat_ambiguity_delta,
    w_token=args.mat_w_token,
    w_density=args.mat_w_density,
    max_log_density_diff=args.mat_max_log_density_diff,
    compound_penalty=args.mat_compound_penalty,
    )

    out_folder = out_folder.expanduser().resolve()
    out_folder.mkdir(parents=True, exist_ok=True)

    out_macro = (out_folder / _Path(args.out).name).resolve()
    code = emit_root_macro(
        step_path,
        out_folder,
        meshparam=meshparam,
        step_unit=args.step_unit,
        clip_box=clip_box,
        clip_deduplicate=args.clip_deduplicate,
        name_filter=name_filter,
        materials_csv=args.materials_csv,
        media_json=args.media_json,
        in_field=in_field,
        bom_mass_unit=args.bom_mass_unit,
        g4_nist_json=args.g4_nist_json,
        mat_cfg=mat_cfg,
        surface_report=args.surface_report,
        exact_surfaces=args.exact_surfaces,
        recognize_surfaces=args.recognize_surfaces,
        dump_brep=args.dump_brep,
        csg=args.csg,
        csg_report=args.csg_report,
        max_cells=args.max_cells,
        max_splits=args.max_splits,
        decompose_timeout=args.decompose_timeout,
        mesh_solid=args.mesh_solid,
    )
    out_macro.write_text(code)

    print(f"Wrote ROOT macro: {out_macro}")
    print(f"Wrote facet files into: {out_folder}")
    print("In ROOT you can do:")
    print(f"  root -l {out_macro}")
    print('  build_and_export("geom.root");')


if __name__ == "__main__":
    main()
