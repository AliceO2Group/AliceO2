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

"""Topology helpers of the CSG recogniser: edge convexity, material side, volume and bounding box.

The recognition census tool built on them is `validation/csgCensus.py`.
"""

import math

from cadsupport.occ_env import ensure_occ  # noqa: E402

ensure_occ()

from OCC.Core.BRep import BRep_Tool  # noqa: E402
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface  # noqa: E402
from OCC.Core.BRepBndLib import brepbndlib  # noqa: E402
from OCC.Core.BRepGProp import brepgprop  # noqa: E402
from OCC.Core.BRepLProp import BRepLProp_SLProps  # noqa: E402
from OCC.Core.Bnd import Bnd_Box  # noqa: E402
from OCC.Core.GProp import GProp_GProps  # noqa: E402
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED  # noqa: E402
from OCC.Core.TopExp import TopExp_Explorer, topexp  # noqa: E402
from OCC.Core.TopTools import (TopTools_IndexedDataMapOfShapeListOfShape,  # noqa: E402
                               TopTools_IndexedMapOfShape,
                               TopTools_ListIteratorOfListOfShape)
from OCC.Core.TopoDS import topods  # noqa: E402
from OCC.Core.gp import gp_Pnt, gp_Vec  # noqa: E402
from cadsupport.analytic import SURFACE_TYPE_NAME  # noqa: E402,F401  (read as census.SURFACE_TYPE_NAME)
from cadsupport.primitives import _cross, _dot, _norm, _sub  # noqa: E402

# Below this |n1 x n2| an edge is tangential and its dihedral has no sign.
TANGENTIAL_SIN = 1.0e-6

# Below this a concave/mixed verdict is labelled untrustworthy (a blend seam), never changed.
NEAR_TANGENTIAL_SIN = 1.0e-3


# --------------------------------------------------------------------------------------------
# small vector helpers (gp_Dir/gp_Pnt are awkward to compare directly)
# --------------------------------------------------------------------------------------------

def _xyz(p):
    return (p.X(), p.Y(), p.Z())


def halfspace_side(face, ad, stype):
    """Which side of its own carrier the material is on: `interior` or `exterior`.

    Decided geometrically from the face's own normal, not from the ORIENTATION flag alone.
    """
    if stype == "plane":
        return "interior"
    if stype not in ("cylinder", "cone", "sphere", "torus"):
        return None
    if stype == "sphere":
        sp = ad.Sphere()
        carrier = {"kind": "sphere", "p": _xyz(sp.Location())}
    elif stype == "torus":
        to = ad.Torus()
        ax = to.Axis()
        carrier = {"kind": "torus", "p": _xyz(ax.Location()), "d": _xyz(ax.Direction()),
                   "r": to.MajorRadius()}
    else:
        ax = ad.Cylinder().Axis() if stype == "cylinder" else ad.Cone().Axis()
        carrier = {"kind": stype, "p": _xyz(ax.Location()), "d": _xyz(ax.Direction())}
    return halfspace_side_of(face, ad, carrier)


def halfspace_side_of(face, ad, carrier):
    """`halfspace_side` for a carrier given as parameters, such as a canonicalised B-spline face."""
    kind = carrier["kind"]
    if kind == "plane":
        return "interior"
    if kind not in ("cylinder", "cone", "sphere", "torus"):
        return None
    u = 0.5 * (ad.FirstUParameter() + ad.LastUParameter())
    v = 0.5 * (ad.FirstVParameter() + ad.LastVParameter())
    if not all(math.isfinite(x) for x in (u, v)):
        return None
    n = _face_normal(face, u, v, ad)
    if n is None:
        return None
    try:
        p = _xyz(ad.Value(u, v))
    except Exception:
        return None
    out = _outward_of_carrier(carrier, p)
    if out is None or _norm(out) < 1e-30:
        return None
    return "interior" if _dot(n, out) > 0.0 else "exterior"


def _outward_of_carrier(carrier, p):
    """The direction pointing out of `carrier` at `p` (radial for a cylinder or cone), or None."""
    kind = carrier["kind"]
    if kind == "sphere":
        return _sub(p, carrier["p"])
    loc, d = carrier["p"], carrier["d"]
    rel = _sub(p, loc)
    radial = _sub(rel, tuple(c * _dot(rel, d) for c in d))
    if kind != "torus":
        return radial
    rl = _norm(radial)
    if rl < 1e-30:
        return None
    centre = tuple(loc[i] + radial[i] / rl * carrier["r"] for i in range(3))
    return _sub(p, centre)


# --------------------------------------------------------------------------------------------
# edge convexity
# --------------------------------------------------------------------------------------------

class FaceEdgeOrientations:
    """Per-face map from edge to the orientation(s) it occurs with, built once per face.

    Rescanning a face's wires for every edge would be quadratic in its edge count.
    """

    def __init__(self, solid):
        self._emap = TopTools_IndexedMapOfShape()
        topexp.MapShapes(solid, TopAbs_EDGE, self._emap)
        self._fmap = TopTools_IndexedMapOfShape()
        topexp.MapShapes(solid, TopAbs_FACE, self._fmap)
        self._cache = {}
        self._adaptors = {}

    def adaptor(self, face):
        """The face's `BRepAdaptor_Surface(face, True)`, built once per face."""
        fi = self._fmap.FindIndex(face)
        ad = self._adaptors.get(fi) if fi else None
        if ad is None:
            ad = BRepAdaptor_Surface(face, True)
            if fi:
                self._adaptors[fi] = ad
        return ad

    def get(self, edge, face):
        fi = self._fmap.FindIndex(face)
        table = self._cache.get(fi)
        if table is None:
            table = {}
            exp = TopExp_Explorer(face, TopAbs_EDGE)
            while exp.More():
                e = topods.Edge(exp.Current())
                table.setdefault(self._emap.FindIndex(e), []).append(e.Orientation())
                exp.Next()
            self._cache[fi] = table
        return table.get(self._emap.FindIndex(edge), [])


def _face_normal(face, u, v, adaptor=None):
    try:
        ad = BRepAdaptor_Surface(face, True) if adaptor is None else adaptor
        props = BRepLProp_SLProps(ad, u, v, 1, 1.0e-9)
        if not props.IsNormalDefined():
            return None
        n = _xyz(props.Normal())
    except Exception:
        return None
    if _norm(n) < 1e-30:
        return None
    if face.Orientation() == TopAbs_REVERSED:
        n = (-n[0], -n[1], -n[2])
    return n


def _pcurve(edge, face):
    """`(2D curve, first, last)` of an edge on a face, or None."""
    res = BRep_Tool.CurveOnSurface(edge, face)
    if res is None:
        return None
    c2d, f, l = res[0], res[1], res[2]
    if c2d is None:
        return None
    return c2d, f, l


def _uv_at(pcurve, t):
    """The (u, v) of a pcurve at edge parameter `t`, clamped to its range."""
    if pcurve is None:
        return None
    c2d, f, l = pcurve
    p = c2d.Value(min(max(t, f), l))
    return p.X(), p.Y()


def edge_dihedral(edge, f1, f2, orients, samples=3):
    """Classify the dihedral along an edge shared by two faces.

    (n1 x n2) . t >= 0 is convex, with t the edge tangent oriented along f1's traversal and n
    the *outward* normals (face orientation applied). A curved edge can change character, so it
    is sampled and the verdict is `mixed` when it does.
    """
    try:
        curve = BRepAdaptor_Curve(edge)
        first, last = curve.FirstParameter(), curve.LastParameter()
    except Exception:
        return "error", 0.0
    if not (math.isfinite(first) and math.isfinite(last)) or last <= first:
        return "error", 0.0

    o1 = orients.get(edge, f1)
    if len(o1) != 1:
        return "seam", 0.0
    sign1 = -1.0 if o1[0] == TopAbs_REVERSED else 1.0

    verdicts = set()
    max_sin = 0.0
    p, d1 = gp_Pnt(), gp_Vec()
    pcurves = None
    for i in range(samples):
        frac = (i + 1.0) / (samples + 1.0)
        t = first + frac * (last - first)
        try:
            curve.D1(t, p, d1)
        except Exception:
            continue
        tangent = (d1.X() * sign1, d1.Y() * sign1, d1.Z() * sign1)
        tl = _norm(tangent)
        if tl < 1e-30:
            continue
        tangent = tuple(c / tl for c in tangent)

        if pcurves is None:
            pcurves = (_pcurve(edge, f1), _pcurve(edge, f2))
        uv1 = _uv_at(pcurves[0], t)
        uv2 = _uv_at(pcurves[1], t)
        if uv1 is None or uv2 is None:
            continue
        n1 = _face_normal(f1, uv1[0], uv1[1], orients.adaptor(f1))
        n2 = _face_normal(f2, uv2[0], uv2[1], orients.adaptor(f2))
        if n1 is None or n2 is None:
            continue
        x = _cross(n1, n2)
        s = _norm(x)
        max_sin = max(max_sin, s)
        if s <= TANGENTIAL_SIN:
            verdicts.add("tangential")
        else:
            verdicts.add("convex" if _dot(x, tangent) >= 0.0 else "concave")

    if not verdicts:
        return "error", max_sin
    if len(verdicts) == 1:
        return verdicts.pop(), max_sin
    verdicts.discard("tangential")
    if len(verdicts) == 1:
        return verdicts.pop(), max_sin
    return "mixed", max_sin



def shape_list(lst):
    """The shapes in a TopTools_ListOfShape, via its iterator (pythonOCC 7.9 has no __iter__)."""
    out = []
    it = TopTools_ListIteratorOfListOfShape(lst)
    while it.More():
        out.append(it.Value())
        it.Next()
    return out


def edge_census(solid):
    amap = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(solid, TopAbs_EDGE, TopAbs_FACE, amap)
    orients = FaceEdgeOrientations(solid)
    counts = {"edges": 0, "convex": 0, "concave": 0, "tangential": 0, "mixed": 0,
              "seam": 0, "nonManifold": 0, "boundary": 0, "degenerate": 0, "error": 0,
              # Concave/mixed verdicts on near-tangential edges, whose sign is noise.
              "concaveNearTangential": 0, "mixedNearTangential": 0}
    for i in range(1, amap.Size() + 1):
        edge = topods.Edge(amap.FindKey(i))
        counts["edges"] += 1
        if BRep_Tool.Degenerated(edge):
            counts["degenerate"] += 1          # a pole of a sphere/cone: no dihedral exists
            continue
        faces = shape_list(amap.FindFromIndex(i))
        distinct = []
        for f in faces:
            if not any(f.IsSame(g) for g in distinct):
                distinct.append(f)
        if len(distinct) == 1:
            counts["seam" if len(faces) > 1 else "boundary"] += 1
            continue
        if len(distinct) != 2:
            counts["nonManifold"] += 1
            continue
        verdict, max_sin = edge_dihedral(edge, topods.Face(distinct[0]),
                                         topods.Face(distinct[1]), orients)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict in ("concave", "mixed") and max_sin < NEAR_TANGENTIAL_SIN:
            counts[verdict + "NearTangential"] += 1
    return counts


def bounding_box(shape):
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    if box.IsVoid():
        return None
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return [xmin, ymin, zmin, xmax, ymax, zmax]


def volume_of(shape):
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()
