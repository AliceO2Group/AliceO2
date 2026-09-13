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

"""
O2_TGeoToCAD.py -- TGeo -> STEP (AP214) with XCAF assembly structure.

The inverse of `O2_CADtoTGeo.py`: it reads a ROOT geometry file (an `o2-sim`
`o2sim_geometry.root`, ideal or aligned), walks the `TGeoVolume` DAG, builds an
OCCT solid for every volume whose shape it can map, and writes one STEP file with
the assembly tree preserved.  The original TGeo is then an exact oracle for the
round trip TGeo -> STEP -> `O2_CADtoTGeo.py`.

The mapping
-----------
    TGeoVolume with no daughters      -> one XCAF simple shape (a definition)
    TGeoVolume with daughters         -> one XCAF assembly label, one component
                                         per TGeoNode referring to the daughter's
                                         definition, carrying the node's TGeoMatrix
    TGeoVolume with daughters AND its
      own (non-assembly) shape        -> the above, plus one extra component
                                         `<name>__body` holding the mother's own
                                         solid at the identity
    TGeoVolumeAssembly                -> a pure XCAF assembly, no solid

A logical volume is converted once and referenced from every node that places it.
Definitions are keyed on volume identity, never on the name, which TGeo does not
require to be unique; two volumes share a definition when they agree by value (the
same shape, and for a mother the same placed content).  A name covering several
definitions is emitted as `name`, `name#2`, ... and recorded as `nameDisambiguation`.

Mother solids are exported uncarved, so every part is the shape the TGeo author
wrote and compares directly with `TGeoShape::Capacity()`.  `--carve-mothers`
subtracts the placed daughters for a CAD-facing export.

Units: TGeo is cm, STEP is written in mm, so every length and translation is
scaled by 10.

Usage
-----
    O2_TGeoToCAD.py INPUT.root OUTPUT.step [options]
    O2_TGeoToCAD.py --self-test

    --report FILE        per-volume JSON report (default: <output>.report.json)
    --top VOLNAME        start from this volume instead of the TGeoManager top
    --include-name PAT   only convert volumes whose name matches this glob
                         (their ancestors are still emitted as assemblies)
    --no-mother-bodies   omit the `__body` component of volumes with daughters
    --skip-top-body      omit only the top volume's own solid (the `cave` box)
    --carve-mothers      subtract placed daughters from each mother solid
    --dedup-world        expand the tree per occurrence and drop any placement of a
                         volume that coincides exactly with another placement of the
                         same volume (see "coincident placements" below)
    --no-verify          skip the per-definition BRepGProp capacity check
    --no-step            build and report, but do not write the STEP
    --quiet

Coincident placements
---------------------
The default export reproduces a volume placed twice at the same world transform,
which `O2_CADtoTGeo.py` refuses.  `--dedup-world` expands the tree per occurrence
and drops every repeated (definition, world transform); the key is the definition,
as in `O2_CADtoTGeo.py`.

Reflections
-----------
A STEP placement is a proper rigid motion.  With Z = diag(1, 1, -1) and V^ = Z*V a
volume's mirrored prototype, a reflecting placement M of V is M*V = (M*Z)*V^ with
M*Z proper; the same identity one level down pushes a reflection through an
assembly to its leaves, so every volume has at most two prototypes.  Mirrored
solids use an exact `gp_Trsf`; `gp_GTrsf` is only for a genuine non-uniform scale,
which is baked.

Report
------
One record per definition with {name, emittedName, mirrored, shapeClass, converted,
reason, capacity_cm3, occVolume_cm3, relDev, sharedByVolumes, ...}, a summary keyed
by shape class, `nameDisambiguation` and `sharedDefinitionMaxRelDev`.  A declined
volume carries a machine-readable `reason`.
"""

import argparse
import fnmatch
import json
import math
import os
import sys
import time

# --------------------------------------------------------------------------
# OCCT
# --------------------------------------------------------------------------

from OCC.Core.gp import (
    gp_Pnt, gp_Dir, gp_Vec, gp_XYZ, gp_Ax1, gp_Ax2, gp_Trsf, gp_GTrsf, gp_Mat,
    gp_Elips, gp_Pln,
)
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeRevol,
    BRepPrimAPI_MakePrism, BRepPrimAPI_MakeHalfSpace,
)
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform, BRepBuilderAPI_GTransform,
    BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid,
)
from OCC.Core.BRepFill import brepfill
from OCC.Core.TopoDS import topods
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TDataStd import TDataStd_Name
from OCC.Core.TDF import TDF_LabelSequence, TDF_Label
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone

SCALE_TO_MM = 10.0          # TGeo cm -> STEP mm
BOOLEAN_VOLUME_TOL = 1e-4   # relative slack on the boolean volume invariant
EPS = 1e-12


class ShapeDeclined(Exception):
    """A TGeo shape this mapper does not (or could not) convert. The message is
    the machine-readable decline reason that lands in the report."""


# --------------------------------------------------------------------------
# small OCCT helpers
# --------------------------------------------------------------------------

def _moved(shape, trsf):
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _rotz(deg):
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(deg))
    return t


def _translate(dx, dy, dz):
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(float(dx), float(dy), float(dz)))
    return t


def _ax2(z0, phi1_deg):
    ph = math.radians(phi1_deg)
    return gp_Ax2(gp_Pnt(0.0, 0.0, float(z0)), gp_Dir(0, 0, 1),
                  gp_Dir(math.cos(ph), math.sin(ph), 0.0))


def solid_volume_mm3(shape):
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return abs(props.Mass())


def _has_solid(shape):
    if shape is None or shape.IsNull():
        return False
    return TopExp_Explorer(shape, TopAbs_SOLID).More()


def _check(shape, what):
    if shape is None or shape.IsNull():
        raise ShapeDeclined(f"{what}: OCCT returned a null shape")
    if not _has_solid(shape):
        raise ShapeDeclined(f"{what}: OCCT result contains no solid")
    return shape


def _dedupe_ring(pts, tol=1e-9):
    """Drop consecutive duplicates in a closed point ring, wrap included."""
    out = []
    for p in pts:
        if out and abs(p[0] - out[-1][0]) < tol and abs(p[1] - out[-1][1]) < tol:
            continue
        out.append(p)
    while len(out) > 1 and abs(out[0][0] - out[-1][0]) < tol and abs(out[0][1] - out[-1][1]) < tol:
        out.pop()
    return out


def _revolve_profile(pts_rz, phi1_deg, dphi_deg, what):
    """Revolve a closed (r, z) profile in the x>=0 half of the XZ plane about +Z.

    This is the exact route for every solid of revolution: one operation, no
    booleans, and rmin > 0 comes out as a real inner face rather than a cut.
    """
    pts = _dedupe_ring([(float(r), float(z)) for (r, z) in pts_rz])
    if len(pts) < 3:
        raise ShapeDeclined(f"{what}: degenerate r-z profile ({len(pts)} distinct points)")
    if min(p[0] for p in pts) < -1e-9:
        raise ShapeDeclined(f"{what}: negative radius in profile")
    poly = BRepBuilderAPI_MakePolygon()
    for (r, z) in pts:
        poly.Add(gp_Pnt(r, 0.0, z))
    poly.Close()
    if not poly.IsDone():
        raise ShapeDeclined(f"{what}: could not build the r-z profile wire")
    mf = BRepBuilderAPI_MakeFace(poly.Wire())
    if not mf.IsDone():
        raise ShapeDeclined(f"{what}: r-z profile is not a valid planar face")
    rev = BRepPrimAPI_MakeRevol(mf.Face(), gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                math.radians(dphi_deg))
    rev.Build()
    if not rev.IsDone():
        raise ShapeDeclined(f"{what}: revolution of the r-z profile failed")
    sh = rev.Shape()
    if abs(phi1_deg) > 1e-12:
        sh = _moved(sh, _rotz(phi1_deg))
    return _check(sh, what)


def _revolve_edges(elements, phi1_deg, dphi_deg, what):
    """Revolve a closed (r, z) profile made of line and arc elements about +Z.

    Elements are ("line", p1, p2) or ("arc", p1, pmid, p2), each point an (r, z)
    pair in the x >= 0 half of the XZ plane.  This is the sphere route: OCCT's
    BRepPrimAPI_MakeSphere cuts theta with *planes* (a spherical zone), while TGeo
    cuts it with *cones* through the centre (a spherical cone), so the primitive
    cannot be used for a theta-sectioned sphere at all.
    """
    def _p(rz):
        return gp_Pnt(float(rz[0]), 0.0, float(rz[1]))

    mw = BRepBuilderAPI_MakeWire()
    nedges = 0
    for e in elements:
        if e[0] == "line":
            p1, p2 = e[1], e[2]
            if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < 1e-9:
                continue
            mw.Add(BRepBuilderAPI_MakeEdge(_p(p1), _p(p2)).Edge())
        else:
            p1, pm, p2 = e[1], e[2], e[3]
            arc = GC_MakeArcOfCircle(_p(p1), _p(pm), _p(p2))
            if not arc.IsDone():
                raise ShapeDeclined(f"{what}: could not build a profile arc")
            mw.Add(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())
        nedges += 1
    if nedges < 2 or not mw.IsDone():
        raise ShapeDeclined(f"{what}: could not close the r-z profile wire")
    mf = BRepBuilderAPI_MakeFace(mw.Wire())
    if not mf.IsDone():
        raise ShapeDeclined(f"{what}: r-z profile is not a valid planar face")
    rev = BRepPrimAPI_MakeRevol(mf.Face(), gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                math.radians(dphi_deg))
    rev.Build()
    if not rev.IsDone():
        raise ShapeDeclined(f"{what}: revolution of the r-z profile failed")
    sh = rev.Shape()
    if abs(phi1_deg) > 1e-12:
        sh = _moved(sh, _rotz(phi1_deg))
    return _check(sh, what)


def _signed_volume(shape):
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


def _polygon_wire(pts, what):
    poly = BRepBuilderAPI_MakePolygon()
    for (x, y, z) in pts:
        poly.Add(gp_Pnt(float(x), float(y), float(z)))
    poly.Close()
    if not poly.IsDone():
        raise ShapeDeclined(f"{what}: could not build a polygon wire")
    return poly.Wire()


def _dedupe_ring3(pts, tol=1e-9):
    out = []
    for p in pts:
        if out and max(abs(p[i] - out[-1][i]) for i in range(3)) < tol:
            continue
        out.append(p)
    while len(out) > 1 and max(abs(out[0][i] - out[-1][i]) for i in range(3)) < tol:
        out.pop()
    return out


def _quad_face(b0, b1, t1, t0, what, tol=1e-7):
    """One lateral patch of a prism: a planar face when the four corners are
    coplanar (so the reverse converter sees a plane), else a ruled face."""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    pts = _dedupe_ring3([b0, b1, t1, t0])
    if len(pts) < 3:
        return None
    # Distinct points can be collinear (a TGeoPgon z-step); a zero Newell area means no face.
    nrm = [0.0, 0.0, 0.0]
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2])
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0])
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1])
    span = max(math.sqrt(sum((pp[i] - pts[0][i]) ** 2 for i in range(3))) for pp in pts[1:])
    if math.sqrt(sum(c * c for c in nrm)) <= tol * span * span:
        return None
    if len(pts) == 3:
        return BRepBuilderAPI_MakeFace(_polygon_wire(pts, what)).Face()
    n = cross(sub(b1, b0), sub(t0, b0))
    nn = math.sqrt(sum(c * c for c in n))
    scale = max(math.sqrt(sum(c * c for c in sub(b1, b0))),
                math.sqrt(sum(c * c for c in sub(t0, b0))), 1e-30)
    d = sub(t1, b0)
    off = abs(sum(n[i] * d[i] for i in range(3))) / nn if nn > 0 else 0.0
    if nn > 1e-24 and off <= tol * scale:
        mf = BRepBuilderAPI_MakeFace(_polygon_wire(pts, what))
        if mf.IsDone():
            return mf.Face()
    e1 = BRepBuilderAPI_MakeEdge(gp_Pnt(*b0), gp_Pnt(*b1)).Edge()
    e2 = BRepBuilderAPI_MakeEdge(gp_Pnt(*t0), gp_Pnt(*t1)).Edge()
    return brepfill.Face(e1, e2)


def _prism_from_rings(outer, inner=None, what="prism"):
    """Build a solid from a stack of closed sections by sewing explicit faces.

    `outer` (and the optional `inner`, which makes the caps annular) is a list of
    sections, each a list of (x, y, z) with the same vertex count and order.
    """
    outer = [_dedupe_ring3(r) for r in outer]
    if len(outer) < 2:
        raise ShapeDeclined(f"{what}: fewer than two sections")
    nv = len(outer[0])
    if nv < 3 or any(len(r) != nv for r in outer):
        raise ShapeDeclined(
            f"{what}: sections have {sorted(set(len(r) for r in outer))} distinct "
            "vertices; a prism needs the same count in every section")
    rings = [outer]
    if inner is not None:
        inner = [_dedupe_ring3(r) for r in inner]
        if any(len(r) != nv for r in inner):
            raise ShapeDeclined(f"{what}: inner sections do not match the outer count")
        rings.append(inner)

    faces = []
    for ring in rings:
        for k in range(len(ring) - 1):
            lo, hi = ring[k], ring[k + 1]
            for i in range(nv):
                j = (i + 1) % nv
                f = _quad_face(lo[i], lo[j], hi[j], hi[i], what)
                if f is not None:
                    faces.append(f)
    # caps, annular when there is an inner stack
    for idx in (0, -1):
        mf = BRepBuilderAPI_MakeFace(_polygon_wire(outer[idx], what))
        if inner is not None:
            mf.Add(topods.Wire(_polygon_wire(inner[idx], what).Reversed()))
        if not mf.IsDone():
            raise ShapeDeclined(f"{what}: could not build a cap face")
        faces.append(mf.Face())

    ext = max(abs(c) for r in outer for p in r for c in p) or 1.0
    sew = BRepBuilderAPI_Sewing(1e-7 * ext)
    for f in faces:
        sew.Add(f)
    sew.Perform()
    shell = sew.SewedShape()
    if shell is None or shell.IsNull():
        raise ShapeDeclined(f"{what}: sewing produced nothing")
    try:
        ms = BRepBuilderAPI_MakeSolid(topods.Shell(shell))
        ms.Build()
        solid = ms.Solid()
    except Exception as e:
        raise ShapeDeclined(f"{what}: faces did not sew into a closed shell ({e})")
    if _signed_volume(solid) < 0:
        solid = topods.Solid(solid.Reversed())
    return _check(solid, what)


def _unify(shape):
    """Merge co-planar / co-cylindrical neighbouring faces (the seams a fuse chain leaves)."""
    u = ShapeUpgrade_UnifySameDomain(shape)
    u.Build()
    return u.Shape()


def _run_boolean(op, a, b):
    algo = op(a, b)
    algo.Build()
    if not algo.IsDone():
        return None
    return algo.Shape()


def _boolean(op, a, b, what, lower=True):
    """A boolean with a volume invariant, because OCCT can fail silently.

        fuse    max(vA, vB) <= v <= vA + vB
        cut     vA - vB     <= v <= vA
        common  0           <= v <= min(vA, vB)

    `lower=False` drops the lower bound, for an unbounded half-space tool.  A violation
    is retried with the operands unified, then declined.  The band is loose on
    purpose: it catches a lost operand, not an accuracy error.
    """
    try:
        va, vb = solid_volume_mm3(a), solid_volume_mm3(b)
    except Exception:
        va = vb = None

    def bounds(v):
        if va is None:
            return True, ""
        tol = BOOLEAN_VOLUME_TOL * max(va, vb, 1.0)
        if op is BRepAlgoAPI_Fuse:
            lo, hi = max(va, vb) - tol, va + vb + tol
        elif op is BRepAlgoAPI_Cut:
            lo, hi = va - vb - tol, va + tol
        else:
            lo, hi = -tol, min(va, vb) + tol
        if not lower:
            lo = -tol
        return lo <= v <= hi, f"{v:.6g} outside [{lo:.6g}, {hi:.6g}] mm^3"

    sh = _run_boolean(op, a, b)
    if sh is None:
        raise ShapeDeclined(f"{what}: OCCT boolean did not complete")
    _check(sh, what)
    ok, msg = bounds(solid_volume_mm3(sh))
    if ok:
        return sh
    retry = _run_boolean(op, _unify(a), _unify(b))
    if retry is not None and _has_solid(retry):
        ok2, msg2 = bounds(solid_volume_mm3(retry))
        if ok2:
            return retry
        msg = f"{msg}; after unifying the operands {msg2}"
    raise ShapeDeclined(
        f"{what}: OCCT's boolean returned a volume the operands cannot give "
        f"({msg}); the operation failed silently")


# --------------------------------------------------------------------------
# TGeoMatrix -> OCCT transform
# --------------------------------------------------------------------------

def tgeo_matrix_components(m):
    """(3x3 row-major matrix including any TGeoScale, translation in mm)."""
    if m is None:
        return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [0., 0., 0.]
    r = m.GetRotationMatrix()
    s = m.GetScale()
    t = m.GetTranslation()
    rot = [[float(r[3 * i + j]) for j in range(3)] for i in range(3)]
    sc = [float(s[j]) for j in range(3)]
    mat = [[rot[i][j] * sc[j] for j in range(3)] for i in range(3)]
    tr = [float(t[i]) * SCALE_TO_MM for i in range(3)]
    return mat, tr


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


# Hand-written rotation constants are not exactly orthogonal: a matrix inside this band is
# snapped to the nearest rotation (and reported), one outside is refused as rigid and baked.
_ORTHO_TOL = 1e-6

# The relative volume band a baked isometry must preserve.
_ISOMETRY_TOL = 1e-6

# A rotation correction below this is double-precision noise, not reported.
_ORTHO_NOISE = 1e-12


def orthogonality_deviation(mat):
    """max |M^T M - I| over the nine entries: 0 for an exact rotation or mirror."""
    return max(abs(sum(mat[k][i] * mat[k][j] for k in range(3))
                   - (1.0 if i == j else 0.0))
               for i in range(3) for j in range(3))


def _inv3(m):
    d = _det3(m)
    if abs(d) < 1e-30:
        return None
    c = [[m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3]
          - m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3]
          for j in range(3)] for i in range(3)]
    return [[c[j][i] / d for j in range(3)] for i in range(3)]


def orthonormalise(mat):
    """(the nearest orthogonal matrix, how far the input was, how far it moved).

    Polar decomposition by Newton's iteration `R <- (R + R^-T)/2`, which converges
    to the orthogonal factor of `R` and preserves the sign of the determinant, so a
    reflection stays a reflection.  Snapping is what keeps the exactness downstream:
    `gp_Trsf` and every world transform composed from it are then built from a
    matrix that really is an isometry.
    """
    dev = orthogonality_deviation(mat)
    r = [row[:] for row in mat]
    for _ in range(8):
        inv = _inv3(r)
        if inv is None:
            return mat, dev, 0.0
        r = [[0.5 * (r[i][j] + inv[j][i]) for j in range(3)] for i in range(3)]
        if orthogonality_deviation(r) < 1e-15:
            break
    corr = max(abs(r[i][j] - mat[i][j]) for i in range(3) for j in range(3))
    return r, dev, corr


def _isometry_trsf(mat, tr, proper_only):
    """(gp_Trsf, orthogonality deviation, correction) or (None, deviation, 0.0).

    A `gp_Trsf` carries an improper orthogonal matrix perfectly well -- OCCT models
    it as a uniform scale of -1 -- and `BRepBuilderAPI_Transform` then moves the
    exact analytic carriers.  Only a genuinely non-uniform scale needs a `gp_GTrsf`.
    """
    dev = orthogonality_deviation(mat)
    if dev > _ORTHO_TOL:
        return None, dev, 0.0
    d = _det3(mat)
    if abs(abs(d) - 1.0) > _ORTHO_TOL or (proper_only and d < 0.0):
        return None, dev, 0.0
    mat, dev, corr = orthonormalise(mat)
    t = gp_Trsf()
    try:
        t.SetValues(mat[0][0], mat[0][1], mat[0][2], tr[0],
                    mat[1][0], mat[1][1], mat[1][2], tr[1],
                    mat[2][0], mat[2][1], mat[2][2], tr[2])
    except Exception:
        return None, dev, corr
    return t, dev, corr


def tgeo_matrix_to_isometry(m):
    """An exact gp_Trsf for any isometry of a TGeoMatrix, reflections included."""
    mat, tr = tgeo_matrix_components(m)
    return _isometry_trsf(mat, tr, proper_only=False)[0]


def tgeo_matrix_to_gtrsf(m):
    mat, tr = tgeo_matrix_components(m)
    g = gp_GTrsf()
    g.SetVectorialPart(gp_Mat(mat[0][0], mat[0][1], mat[0][2],
                              mat[1][0], mat[1][1], mat[1][2],
                              mat[2][0], mat[2][1], mat[2][2]))
    g.SetTranslationPart(gp_XYZ(tr[0], tr[1], tr[2]))
    return g


def apply_isometry(shape, t, what):
    """Apply an exact isometry.

    An isometry cannot change a volume, so this is priced against the volume it must preserve.
    """
    v0 = solid_volume_mm3(shape)
    algo = BRepBuilderAPI_Transform(shape, t, True)
    if not algo.IsDone():
        raise ShapeDeclined(f"{what}: BRepBuilderAPI_Transform failed")
    out = _check(algo.Shape(), what)
    v1 = solid_volume_mm3(out)
    if v0 > 0 and abs(v1 - v0) > _ISOMETRY_TOL * v0:
        raise ShapeDeclined(f"{what}: the isometry changed the volume by "
                            f"{abs(v1 - v0) / v0:.3e} relative")
    if _signed_volume(out) < 0:
        raise ShapeDeclined(f"{what}: the isometry left the solid inside out")
    return out


def zmirror_trsf():
    """The canonical reflection z -> -z, exactly."""
    t = gp_Trsf()
    t.SetMirror(gp_Ax2(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.)))
    return t


def _zmirror_left(mat, tr):
    """Z * M, with Z = diag(1, 1, -1)."""
    return ([mat[0][:], mat[1][:], [-v for v in mat[2]]], [tr[0], tr[1], -tr[2]])


def _zmirror_right(mat, tr):
    """M * Z."""
    return ([[mat[i][0], mat[i][1], -mat[i][2]] for i in range(3)], list(tr))


def child_location(parent_mirrored, mat, tr):
    """Where a daughter goes, and whether it is the daughter's mirrored prototype.

    Write Z = diag(1, 1, -1) and let V^ = Z*V be a volume's mirrored prototype.
    A reflecting placement M of V is then M*V = (M*Z)*(Z*V) = (M*Z)*V^, and M*Z is
    proper -- so a reflection never needs a general transform and never needs a
    solid to bake into: it becomes a rigid placement of the child's prototype.  The
    same identity applied to Z*M pushes the reflection through an assembly and down
    to its leaves, which is why a reflected subtree can be emitted at all.

    Every volume therefore has at most two prototypes, itself and Z*itself, shared
    by every reflected use of it.

    Returns (gp_Trsf or None, child_mirrored, location matrix, location
    translation, orthogonality deviation, orthonormalisation correction).
    """
    if parent_mirrored:
        mat, tr = _zmirror_left(mat, tr)
    mirrored = _det3(mat) < 0.0
    if mirrored:
        mat, tr = _zmirror_right(mat, tr)
    t, dev, corr = _isometry_trsf(mat, tr, proper_only=True)
    return t, mirrored, mat, tr, dev, corr


def mirror_solid_z(shape, what="mirrored copy"):
    """Reflect a solid through the z = 0 plane, exactly and carrier-preserving."""
    return apply_isometry(shape, zmirror_trsf(), what)


def apply_tgeo_matrix(shape, m, what):
    """Move `shape` by a TGeoMatrix: an isometry through `gp_Trsf`, a non-uniform scale through `gp_GTrsf`."""
    t = tgeo_matrix_to_isometry(m)
    if t is not None:
        if t.IsNegative():
            return apply_isometry(shape, t, what)
        return _moved(shape, t)
    g = tgeo_matrix_to_gtrsf(m)
    algo = BRepBuilderAPI_GTransform(shape, g, True)
    if not algo.IsDone():
        raise ShapeDeclined(f"{what}: could not apply a reflecting/scaling matrix")
    return _check(algo.Shape(), what)


# --------------------------------------------------------------------------
# volume and shape identity
# --------------------------------------------------------------------------

_ROOT = None


def _root():
    global _ROOT
    if _ROOT is None:
        import ROOT
        _ROOT = ROOT
    return _ROOT


def obj_id(o):
    """The address of a ROOT object; it identifies a `TGeoVolume`, whose name need not be unique."""
    return int(_root().addressof(o))


def _r(v):
    return round(float(v), 12)


def _rs(seq, n):
    return tuple(_r(seq[i]) for i in range(n))


def _sig_zprofile(sh):
    nz = int(sh.GetNz())
    return (nz, tuple((_r(sh.GetZ(i)), _r(sh.GetRmin(i)), _r(sh.GetRmax(i)))
                      for i in range(nz)))


def shape_signature(sh):
    """A value key for a `TGeoShape`: equal keys mean the same solid.

    A class not known by value, `TGeoCompositeShape` included, is keyed on its address.
    Classes match exactly, so a subclass (`TGeoGtra` under `TGeoTrap`) is never taken for its base.
    """
    cls = str(sh.ClassName())
    o = sh.GetOrigin()
    bbox = (_r(sh.GetDX()), _r(sh.GetDY()), _r(sh.GetDZ()),
            _r(o[0]), _r(o[1]), _r(o[2]))
    if cls == "TGeoBBox":
        return (cls, bbox)
    if cls == "TGeoTube":
        return (cls, bbox, _r(sh.GetRmin()), _r(sh.GetRmax()), _r(sh.GetDz()))
    if cls == "TGeoTubeSeg":
        return (cls, bbox, _r(sh.GetRmin()), _r(sh.GetRmax()), _r(sh.GetDz()),
                _r(sh.GetPhi1()), _r(sh.GetPhi2()))
    if cls == "TGeoCtub":
        return (cls, bbox, _r(sh.GetRmin()), _r(sh.GetRmax()), _r(sh.GetDz()),
                _r(sh.GetPhi1()), _r(sh.GetPhi2()),
                _rs(sh.GetNlow(), 3), _rs(sh.GetNhigh(), 3))
    if cls == "TGeoCone":
        return (cls, bbox, _r(sh.GetDz()), _r(sh.GetRmin1()), _r(sh.GetRmax1()),
                _r(sh.GetRmin2()), _r(sh.GetRmax2()))
    if cls == "TGeoConeSeg":
        return (cls, bbox, _r(sh.GetDz()), _r(sh.GetRmin1()), _r(sh.GetRmax1()),
                _r(sh.GetRmin2()), _r(sh.GetRmax2()),
                _r(sh.GetPhi1()), _r(sh.GetPhi2()))
    if cls == "TGeoPcon":
        return (cls, bbox, _r(sh.GetPhi1()), _r(sh.GetDphi()), _sig_zprofile(sh))
    if cls == "TGeoPgon":
        return (cls, bbox, _r(sh.GetPhi1()), _r(sh.GetDphi()), int(sh.GetNedges()),
                _sig_zprofile(sh))
    if cls == "TGeoSphere":
        return (cls, bbox, _r(sh.GetRmin()), _r(sh.GetRmax()),
                _r(sh.GetTheta1()), _r(sh.GetTheta2()),
                _r(sh.GetPhi1()), _r(sh.GetPhi2()))
    if cls == "TGeoTorus":
        return (cls, bbox, _r(sh.GetR()), _r(sh.GetRmin()), _r(sh.GetRmax()),
                _r(sh.GetPhi1()), _r(sh.GetDphi()))
    if cls == "TGeoEltu":
        return (cls, bbox, _r(sh.GetA()), _r(sh.GetB()), _r(sh.GetDz()))
    if cls == "TGeoTrd1":
        return (cls, bbox, _r(sh.GetDx1()), _r(sh.GetDx2()), _r(sh.GetDy()),
                _r(sh.GetDz()))
    if cls == "TGeoTrd2":
        return (cls, bbox, _r(sh.GetDx1()), _r(sh.GetDx2()), _r(sh.GetDy1()),
                _r(sh.GetDy2()), _r(sh.GetDz()))
    if cls in ("TGeoArb8", "TGeoTrap"):
        return (cls, bbox, _r(sh.GetDz()), _rs(sh.GetVertices(), 16))
    if cls == "TGeoXtru":
        nv, nz = int(sh.GetNvert()), int(sh.GetNz())
        return (cls, bbox, nv, tuple((_r(sh.GetX(i)), _r(sh.GetY(i))) for i in range(nv)),
                nz, tuple((_r(sh.GetZ(k)), _r(sh.GetXOffset(k)), _r(sh.GetYOffset(k)),
                           _r(sh.GetScale(k))) for k in range(nz)))
    if cls == "TGeoScaledShape":
        return (cls, bbox, _rs(sh.GetScale().GetScale(), 3),
                shape_signature(sh.GetShape()))
    return ("byAddress", cls, obj_id(sh))


# --------------------------------------------------------------------------
# shape converters -- all output mm, centred as TGeo centres them
# --------------------------------------------------------------------------

def _phi_span(phi1, phi2):
    d = float(phi2) - float(phi1)
    while d <= 0:
        d += 360.0
    return float(phi1), min(d, 360.0)


def conv_box(sh, s):
    dx, dy, dz = sh.GetDX() * s, sh.GetDY() * s, sh.GetDZ() * s
    ox, oy, oz = (sh.GetOrigin()[0] * s, sh.GetOrigin()[1] * s, sh.GetOrigin()[2] * s)
    if min(dx, dy, dz) <= 0:
        raise ShapeDeclined("TGeoBBox: a half-length is zero or negative")
    box = BRepPrimAPI_MakeBox(2 * dx, 2 * dy, 2 * dz).Shape()
    return _moved(box, _translate(ox - dx, oy - dy, oz - dz))


def _tube_like(rmin, rmax, dz, phi1, dphi, what):
    if rmax <= 0 or dz <= 0:
        raise ShapeDeclined(f"{what}: rmax or dz is zero")
    if rmin >= rmax:
        raise ShapeDeclined(f"{what}: rmin >= rmax")
    if rmin <= EPS:
        cyl = BRepPrimAPI_MakeCylinder(_ax2(-dz, phi1), rmax, 2 * dz, math.radians(dphi))
        cyl.Build()
        if not cyl.IsDone():
            raise ShapeDeclined(f"{what}: BRepPrimAPI_MakeCylinder failed")
        return _check(cyl.Shape(), what)
    return _revolve_profile([(rmin, -dz), (rmax, -dz), (rmax, dz), (rmin, dz)],
                            phi1, dphi, what)


def conv_tube(sh, s):
    return _tube_like(sh.GetRmin() * s, sh.GetRmax() * s, sh.GetDz() * s,
                      0.0, 360.0, "TGeoTube")


def conv_tubeseg(sh, s):
    phi1, dphi = _phi_span(sh.GetPhi1(), sh.GetPhi2())
    return _tube_like(sh.GetRmin() * s, sh.GetRmax() * s, sh.GetDz() * s,
                      phi1, dphi, "TGeoTubeSeg")


def _cone_like(rmin1, rmax1, rmin2, rmax2, dz, phi1, dphi, what):
    if dz <= 0:
        raise ShapeDeclined(f"{what}: dz is zero")
    if max(rmax1, rmax2) <= 0:
        raise ShapeDeclined(f"{what}: both outer radii are zero")
    if rmin1 <= EPS and rmin2 <= EPS:
        cone = BRepPrimAPI_MakeCone(_ax2(-dz, phi1), rmax1, rmax2, 2 * dz, math.radians(dphi))
        cone.Build()
        if not cone.IsDone():
            raise ShapeDeclined(f"{what}: BRepPrimAPI_MakeCone failed")
        return _check(cone.Shape(), what)
    return _revolve_profile([(rmin1, -dz), (rmax1, -dz), (rmax2, dz), (rmin2, dz)],
                            phi1, dphi, what)


def conv_cone(sh, s):
    return _cone_like(sh.GetRmin1() * s, sh.GetRmax1() * s,
                      sh.GetRmin2() * s, sh.GetRmax2() * s, sh.GetDz() * s,
                      0.0, 360.0, "TGeoCone")


def conv_coneseg(sh, s):
    phi1, dphi = _phi_span(sh.GetPhi1(), sh.GetPhi2())
    return _cone_like(sh.GetRmin1() * s, sh.GetRmax1() * s,
                      sh.GetRmin2() * s, sh.GetRmax2() * s, sh.GetDz() * s,
                      phi1, dphi, "TGeoConeSeg")


def conv_pcon(sh, s):
    nz = int(sh.GetNz())
    if nz < 2:
        raise ShapeDeclined("TGeoPcon: fewer than two z planes")
    z = [sh.GetZ(i) * s for i in range(nz)]
    rmin = [sh.GetRmin(i) * s for i in range(nz)]
    rmax = [sh.GetRmax(i) * s for i in range(nz)]
    phi1, dphi = float(sh.GetPhi1()), float(sh.GetDphi())
    outer = [(rmax[i], z[i]) for i in range(nz)]
    if all(r <= EPS for r in rmin):
        inner = [(0.0, z[nz - 1]), (0.0, z[0])]
    else:
        inner = [(rmin[i], z[i]) for i in range(nz - 1, -1, -1)]
    return _revolve_profile(outer + inner, phi1, dphi, "TGeoPcon")


def _pgon_ring(r_apothem, z, phi1_deg, dphi_deg, nedges, full):
    """The polygon at one z plane. TGeo's rmin/rmax are inscribed-circle radii."""
    dseg = math.radians(dphi_deg) / nedges
    R = r_apothem / math.cos(dseg / 2.0)
    n = nedges if full else nedges + 1
    out = []
    for k in range(n):
        a = math.radians(phi1_deg) + k * dseg
        out.append((R * math.cos(a), R * math.sin(a), z))
    return out


def conv_pgon(sh, s):
    nz = int(sh.GetNz())
    nedges = int(sh.GetNedges())
    if nz < 2 or nedges < 1:
        raise ShapeDeclined("TGeoPgon: fewer than two z planes or no edges")
    phi1, dphi = float(sh.GetPhi1()), float(sh.GetDphi())
    full = abs(dphi - 360.0) < 1e-9
    z = [sh.GetZ(i) * s for i in range(nz)]
    rmin = [sh.GetRmin(i) * s for i in range(nz)]
    rmax = [sh.GetRmax(i) * s for i in range(nz)]
    hollow = any(r > EPS for r in rmin)
    rings = []
    for i in range(nz):
        outer = _pgon_ring(rmax[i], z[i], phi1, dphi, nedges, full)
        if hollow:
            inner = _pgon_ring(max(rmin[i], EPS), z[i], phi1, dphi, nedges, full)
            ring = outer + list(reversed(inner))
        elif full:
            ring = outer
        else:
            ring = outer + [(0.0, 0.0, z[i])]
        rings.append(ring)
    if hollow and full:
        # A full hollow polyhedra has two disjoint rings per section: sew outer and inner stacks.
        outer_rings = [_pgon_ring(rmax[i], z[i], phi1, dphi, nedges, True) for i in range(nz)]
        inner_rings = [_pgon_ring(max(rmin[i], EPS), z[i], phi1, dphi, nedges, True) for i in range(nz)]
        return _prism_from_rings(outer_rings, inner_rings, what="TGeoPgon")
    return _prism_from_rings(rings, what="TGeoPgon")


def conv_sphere(sh, s):
    rmin, rmax = sh.GetRmin() * s, sh.GetRmax() * s
    th1, th2 = float(sh.GetTheta1()), float(sh.GetTheta2())
    phi1, dphi = _phi_span(sh.GetPhi1(), sh.GetPhi2())
    if rmax <= 0:
        raise ShapeDeclined("TGeoSphere: rmax is zero")
    if th2 <= th1:
        raise ShapeDeclined("TGeoSphere: theta2 <= theta1")

    def rz(r, theta_deg):
        a = math.radians(theta_deg)
        return (r * math.sin(a), r * math.cos(a))

    thm = 0.5 * (th1 + th2)
    p1, pm, p2 = rz(rmax, th1), rz(rmax, thm), rz(rmax, th2)
    elems = [("arc", p1, pm, p2)]
    if rmin > EPS:
        q1, qm, q2 = rz(rmin, th1), rz(rmin, thm), rz(rmin, th2)
        elems += [("line", p2, q2), ("arc", q2, qm, q1), ("line", q1, p1)]
    elif p1[0] < 1e-9 and p2[0] < 1e-9:
        elems += [("line", p2, p1)]                 # both poles: close on the axis
    else:
        elems += [("line", p2, (0.0, 0.0)), ("line", (0.0, 0.0), p1)]
    return _revolve_edges(elems, phi1, dphi, "TGeoSphere")


def conv_torus(sh, s):
    R = sh.GetR() * s
    rmin, rmax = sh.GetRmin() * s, sh.GetRmax() * s
    phi1, dphi = float(sh.GetPhi1()), float(sh.GetDphi())
    if rmax <= 0 or R <= 0:
        raise ShapeDeclined("TGeoTorus: R or Rmax is zero")

    def mk(r):
        m = BRepPrimAPI_MakeTorus(_ax2(0.0, phi1), R, r, math.radians(dphi))
        m.Build()
        if not m.IsDone():
            raise ShapeDeclined("TGeoTorus: BRepPrimAPI_MakeTorus failed")
        return _check(m.Shape(), "TGeoTorus")

    outer = mk(rmax)
    if rmin > EPS:
        return _boolean(BRepAlgoAPI_Cut, outer, mk(rmin), "TGeoTorus(hollow)")
    return outer


def conv_eltu(sh, s):
    a, b, dz = sh.GetA() * s, sh.GetB() * s, sh.GetDz() * s
    if a <= 0 or b <= 0 or dz <= 0:
        raise ShapeDeclined("TGeoEltu: a semi-axis or dz is zero")
    if a >= b:
        ax = gp_Ax2(gp_Pnt(0, 0, -dz), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
        maj, mnr = a, b
    else:
        ax = gp_Ax2(gp_Pnt(0, 0, -dz), gp_Dir(0, 0, 1), gp_Dir(0, 1, 0))
        maj, mnr = b, a
    edge = BRepBuilderAPI_MakeEdge(gp_Elips(ax, maj, mnr)).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    face = BRepBuilderAPI_MakeFace(wire).Face()
    pr = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, 2 * dz))
    pr.Build()
    if not pr.IsDone():
        raise ShapeDeclined("TGeoEltu: prism failed")
    return _check(pr.Shape(), "TGeoEltu")


def conv_trd1(sh, s):
    dx1, dx2 = sh.GetDx1() * s, sh.GetDx2() * s
    dy, dz = sh.GetDy() * s, sh.GetDz() * s
    return _prism_from_rings([
        [(-dx1, -dy, -dz), (dx1, -dy, -dz), (dx1, dy, -dz), (-dx1, dy, -dz)],
        [(-dx2, -dy, dz), (dx2, -dy, dz), (dx2, dy, dz), (-dx2, dy, dz)],
    ], what="TGeoTrd1")


def conv_trd2(sh, s):
    dx1, dx2 = sh.GetDx1() * s, sh.GetDx2() * s
    dy1, dy2 = sh.GetDy1() * s, sh.GetDy2() * s
    dz = sh.GetDz() * s
    return _prism_from_rings([
        [(-dx1, -dy1, -dz), (dx1, -dy1, -dz), (dx1, dy1, -dz), (-dx1, dy1, -dz)],
        [(-dx2, -dy2, dz), (dx2, -dy2, dz), (dx2, dy2, dz), (-dx2, dy2, dz)],
    ], what="TGeoTrd2")


def conv_arb8(sh, s):
    """TGeoArb8 and its subclasses (Trap): eight vertices, ruled lateral faces."""
    v = sh.GetVertices()
    dz = sh.GetDz() * s
    bot = [(v[2 * i] * s, v[2 * i + 1] * s, -dz) for i in range(4)]
    top = [(v[8 + 2 * i] * s, v[8 + 2 * i + 1] * s, dz) for i in range(4)]
    return _prism_from_rings([bot, top], what=sh.ClassName())


def conv_xtru(sh, s):
    nv, nz = int(sh.GetNvert()), int(sh.GetNz())
    if nv < 3 or nz < 2:
        raise ShapeDeclined("TGeoXtru: fewer than 3 vertices or 2 sections")
    x = [sh.GetX(i) for i in range(nv)]
    y = [sh.GetY(i) for i in range(nv)]
    rings = []
    for k in range(nz):
        z = sh.GetZ(k) * s
        x0, y0, sc = sh.GetXOffset(k) * s, sh.GetYOffset(k) * s, sh.GetScale(k)
        rings.append([(x0 + sc * x[i] * s, y0 + sc * y[i] * s, z) for i in range(nv)])
    return _prism_from_rings(rings, what="TGeoXtru")


def conv_ctub(sh, s):
    """A cut tube.

    TGeo's cut planes replace the +-dz end faces, so the tube is built long enough to reach
    past both planes before they cut it.
    """
    rmin, rmax = sh.GetRmin() * s, sh.GetRmax() * s
    dz = sh.GetDz() * s
    phi1, dphi = _phi_span(sh.GetPhi1(), sh.GetPhi2())
    planes = []
    ext = 0.0
    for nvec, z0 in ((sh.GetNlow(), -dz), (sh.GetNhigh(), dz)):
        n = (float(nvec[0]), float(nvec[1]), float(nvec[2]))
        norm = math.sqrt(sum(c * c for c in n))
        if norm < EPS:
            raise ShapeDeclined("TGeoCtub: a cut normal is null")
        n = tuple(c / norm for c in n)
        if abs(n[2]) < 1e-9:
            raise ShapeDeclined("TGeoCtub: a cut plane is parallel to the axis")
        planes.append((n, z0))
        ext = max(ext, rmax * math.hypot(n[0], n[1]) / abs(n[2]))
    ext = ext * 1.5 + 1e-3 * max(rmax, dz)
    base = _tube_like(rmin, rmax, dz + ext, phi1, dphi, "TGeoCtub(base tube)")
    big = 4.0 * max(rmax, dz + ext) + 10.0
    for (n, z0) in planes:
        pl = BRepBuilderAPI_MakeFace(
            gp_Pln(gp_Pnt(0.0, 0.0, z0), gp_Dir(*n)), -big, big, -big, big)
        if not pl.IsDone():
            raise ShapeDeclined("TGeoCtub: could not build a cut plane")
        ref = gp_Pnt(n[0] * big, n[1] * big, z0 + n[2] * big)   # outside the solid
        hs = BRepPrimAPI_MakeHalfSpace(pl.Face(), ref)
        hs.Build()
        if not hs.IsDone():
            raise ShapeDeclined("TGeoCtub: half-space construction failed")
        base = _boolean(BRepAlgoAPI_Cut, base, hs.Solid(), "TGeoCtub", lower=False)
    return base


_BOOL_OPS = {
    "TGeoUnion": (BRepAlgoAPI_Fuse, "union"),
    "TGeoSubtraction": (BRepAlgoAPI_Cut, "subtraction"),
    "TGeoIntersection": (BRepAlgoAPI_Common, "intersection"),
}

# A runaway guard on the boolean-tree walk, well above any real chain.
MAX_BOOLEAN_DEPTH = 512


def conv_composite(sh, s, depth=0):
    if depth > MAX_BOOLEAN_DEPTH:
        raise ShapeDeclined(f"TGeoCompositeShape: boolean tree deeper than {MAX_BOOLEAN_DEPTH}")
    bn = sh.GetBoolNode()
    if bn is None:
        raise ShapeDeclined("TGeoCompositeShape: no boolean node")
    op = _BOOL_OPS.get(bn.ClassName())
    if op is None:
        raise ShapeDeclined(f"TGeoCompositeShape: unknown boolean node {bn.ClassName()}")
    algo, opname = op
    left = shape_to_occ(bn.GetLeftShape(), s, depth + 1)
    right = shape_to_occ(bn.GetRightShape(), s, depth + 1)
    left = apply_tgeo_matrix(left, bn.GetLeftMatrix(), "composite left operand")
    right = apply_tgeo_matrix(right, bn.GetRightMatrix(), "composite right operand")
    return _boolean(algo, left, right, f"TGeoCompositeShape({opname})")


def conv_scaled(sh, s, depth=0):
    inner = shape_to_occ(sh.GetShape(), s, depth + 1)
    sc = sh.GetScale().GetScale()
    g = gp_GTrsf()
    g.SetVectorialPart(gp_Mat(sc[0], 0, 0, 0, sc[1], 0, 0, 0, sc[2]))
    algo = BRepBuilderAPI_GTransform(inner, g, True)
    if not algo.IsDone():
        raise ShapeDeclined("TGeoScaledShape: could not apply the scale")
    return _check(algo.Shape(), "TGeoScaledShape")


_DISPATCH = {
    "TGeoBBox": conv_box,
    "TGeoTube": conv_tube,
    "TGeoTubeSeg": conv_tubeseg,
    "TGeoCtub": conv_ctub,
    "TGeoCone": conv_cone,
    "TGeoConeSeg": conv_coneseg,
    "TGeoPcon": conv_pcon,
    "TGeoPgon": conv_pgon,
    "TGeoSphere": conv_sphere,
    "TGeoTorus": conv_torus,
    "TGeoEltu": conv_eltu,
    "TGeoTrd1": conv_trd1,
    "TGeoTrd2": conv_trd2,
    "TGeoArb8": conv_arb8,
    "TGeoTrap": conv_arb8,
    "TGeoXtru": conv_xtru,
}

# Shapes we know about and deliberately do not map, with the reason.
_KNOWN_DECLINES = {
    "TGeoHalfSpace": "unbounded solid: a half-space has no B-rep body of its own",
    "TGeoGtra": "twisted trapezoid: the lateral twist is not a ruled loft of the "
                "eight Arb8 vertices",
    "TGeoParaboloid": "quadric of revolution not mapped (no OCCT primitive; would "
                      "need a revolved parabola profile)",
    "TGeoHype": "hyperboloid of revolution not mapped",
    "TGeoPara": "parallelepiped not mapped",
    "TGeoTessellated": "already a mesh: STEP would carry facets, not a B-rep solid",
    "TGeoShapeAssembly": "assembly shape: emitted as a pure XCAF assembly, no solid",
}


def shape_to_occ(sh, s=SCALE_TO_MM, depth=0):
    """TGeoShape -> TopoDS_Shape in mm. Raises ShapeDeclined with a reason."""
    if sh is None:
        raise ShapeDeclined("volume has no shape")
    cls = sh.ClassName()
    if cls == "TGeoCompositeShape":
        return conv_composite(sh, s, depth)
    if cls == "TGeoScaledShape":
        return conv_scaled(sh, s, depth)
    fn = _DISPATCH.get(cls)
    if fn is None:
        raise ShapeDeclined(_KNOWN_DECLINES.get(cls, f"shape class {cls} is not mapped"))
    return fn(sh, s)


# --------------------------------------------------------------------------
# media and materials, dumped verbatim into a sidecar keyed by emitted STEP part name
#
# The eight medium parameters are Geant's, in TGeoMedium's own order:
#   0 isvol  1 ifield  2 fieldm  3 tmaxfd  4 stemax  5 deemax  6 epsil  7 stmin
# --------------------------------------------------------------------------

MEDIUM_PARAM_NAMES = ("isvol", "ifield", "fieldm", "tmaxfd",
                      "stemax", "deemax", "epsil", "stmin")


def material_record(mat):
    """Everything needed to rebuild one TGeoMaterial or TGeoMixture."""
    rec = {
        "name": str(mat.GetName()),
        "class": str(mat.ClassName()),
        "Z": float(mat.GetZ()),
        "A": float(mat.GetA()),
        "density": float(mat.GetDensity()),
        "radLen": float(mat.GetRadLen()),
        "intLen": float(mat.GetIntLen()),
        "isMixture": bool(mat.IsMixture()),
    }
    if mat.IsMixture():
        n = int(mat.GetNelements())
        zs, as_, ws = mat.GetZmixt(), mat.GetAmixt(), mat.GetWmixt()
        rec["nElements"] = n
        rec["elements"] = [{"Z": float(zs[i]), "A": float(as_[i]),
                            "W": float(ws[i])} for i in range(n)]
    return rec


def medium_record(med):
    """Everything needed to rebuild one TGeoMedium, its material included."""
    return {
        "name": str(med.GetName()),
        "id": int(med.GetId()),
        "params": {k: float(med.GetParam(i))
                   for i, k in enumerate(MEDIUM_PARAM_NAMES)},
        "material": material_record(med.GetMaterial()),
    }


class TGeoToStep:
    def __init__(self, opts):
        self.opts = opts
        self.doc = TDocStd_Document("O2_TGeoToCAD")
        self.shape_tool = XCAFDoc_DocumentTool.ShapeTool(self.doc.Main())
        self.definitions = {}      # definition id -> (label, occ solid or None)
        self.records = {}          # definition id -> report record
        self.media = {}            # medium name -> medium_record()
        # Volumes emitted as pure assemblies, with daughters but no body (the experiment hall).
        self.hollow = set(getattr(self.opts, 'hollow_volumes', None) or ())
        self.hollow_tag = getattr(self.opts, 'hollow_tag', None) or ''
        self._intern = {}          # definition key -> definition id
        self._byvol = {}           # (volume address, mirrored) -> definition id
        self._seen_vols = set()    # distinct TGeoVolume objects visited
        self._sigcache = {}        # TGeoShape address -> value signature
        self._name_slots = {}      # TGeo name -> {slot key: emitted STEP name}
        self._asm_names = {}       # volume address -> per-occurrence assembly name
        self.nvolumes = 0          # distinct TGeoVolume objects visited
        self.ncomponents = 0
        self.nbaked = 0
        self.nscaled = 0
        self.northo = 0            # placements snapped to the nearest rotation
        self.ortho_worst = (0.0, 0.0, None)   # (deviation, correction, where)
        self.ortho_records = []    # per-placement, capped
        self.scaled_records = []   # matrices refused as rigid, with the number
        self.placed_world = set()      # (definition id, world key), --dedup-world only
        self.ndropped = 0
        self.dropped_examples = []
        self.reflected_nodes = []   # TGeo placements whose matrix reflects
        self.nmirrored_components = 0   # components that place a mirrored prototype
        self.share_worst = (0.0, None)   # the shared-definition capacity self-check
        self.t0 = time.time()

    # ------------------------------------------------------------------

    def log(self, *a):
        if not self.opts.quiet:
            print(*a, file=sys.stderr, flush=True)


    def _record(self, did, vol, emitted, **kw):
        rec = self.records.setdefault(did, {
            "name": str(vol.GetName()),
            "emittedName": emitted,
            "shapeClass": vol.GetShape().ClassName() if vol.GetShape() else None,
            "ndaughters": int(vol.GetNdaughters()),
            "isAssembly": bool(vol.IsAssembly()),
            "converted": False,
            "reason": None,
            "capacity_cm3": None,
            "occVolume_cm3": None,
            "relDev": None,
            "mirrored": False,
            "sharedByVolumes": 1,
        })
        rec.update(kw)
        med = vol.GetMedium()
        if med is not None:
            name = str(med.GetName())
            rec["medium"] = name
            if name not in self.media:
                self.media[name] = medium_record(med)
        return rec

    # ------------------------------------------------------------------
    # definition keys, name disambiguation, and the sharing self-check
    # ------------------------------------------------------------------

    def _kid(self, key):
        """Intern a definition key as a small integer.

        Assembly keys quote their children, so without interning the top volume's
        key would be a nested copy of the whole tree.
        """
        i = self._intern.get(key)
        if i is None:
            i = len(self._intern) + 1
            self._intern[key] = i
        return i

    def _shape_sig(self, vol):
        sh = vol.GetShape()
        if sh is None:
            return ("noShape", obj_id(vol))
        a = obj_id(sh)
        s = self._sigcache.get(a)
        if s is None:
            s = shape_signature(sh)
            self._sigcache[a] = s
        return s

    def _emit_name(self, vol, slot):
        """The STEP name of a definition: `name`, then `name#2`, `name#3`, ...

        One TGeo name can cover several definitions, so the emitted names are
        disambiguated in the order the definitions are created, which the
        depth-first walk makes deterministic.  The mapping goes into the report.
        """
        base = str(vol.GetName())
        # Hall volumes are tagged per module so that two converted modules do not collide.
        if base in self.hollow and self.hollow_tag:
            base = f"{base}_{self.hollow_tag}"
        slots = self._name_slots.setdefault(base, {})
        nm = slots.get(slot)
        if nm is None:
            nm = base if not slots else f"{base}#{len(slots) + 1}"
            slots[slot] = nm
        return nm

    def _occ_asm_name(self, vol):
        """The STEP name of a per-occurrence assembly label (`--dedup-world`).

        One name per *volume*, not per occurrence, so a shared subtree keeps one
        name however often it is expanded.
        """
        a = obj_id(vol)
        nm = self._asm_names.get(a)
        if nm is None:
            nm = self._emit_name(vol, ("vol", a))
            self._asm_names[a] = nm
        return nm

    def _note_orthonormalised(self, where, dev, corr):
        """Record a placement matrix that had to be snapped to the nearest rotation.

        The correction is not absorbed silently: it is counted, the worst is in the
        report summary and the first 50 are listed with their numbers.
        """
        if corr <= _ORTHO_NOISE:
            return          # double-precision dust, not a correction
        self.northo += 1
        if len(self.ortho_records) < 50:
            self.ortho_records.append({"placement": where,
                                       "orthogonalityDeviation": dev,
                                       "rotationCorrection": corr})
        if dev > self.ortho_worst[0]:
            self.ortho_worst = (dev, corr, where)

    def _note_scaled(self, where, child, dev):
        """A placement matrix refused as a rigid one: say so, with the number."""
        self.nscaled += 1
        self.scaled_records.append({"placement": where,
                                    "volume": str(child.GetName()),
                                    "orthogonalityDeviation": dev})
        self.log(f"  [WARN] {where}: placement matrix is not an isometry "
                 f"(|M^T M - I| = {dev:.3e} > {_ORTHO_TOL:.0e}); baking it into a "
                 f"private copy of {child.GetName()}"
                 + (", whose daughters cannot follow" if child.GetNdaughters() else ""))

    def _note_shared(self, did, vol):
        """Price a shared definition against the sharing volume's own capacity.

        A `TGeoCompositeShape` is skipped: its `Capacity()` is a Monte Carlo estimate.
        """
        rec = self.records.get(did)
        if rec is None:
            return
        rec["sharedByVolumes"] = rec.get("sharedByVolumes", 1) + 1
        occv = rec.get("occVolume_cm3")
        sh = vol.GetShape()
        if occv is None or sh is None or sh.ClassName() == "TGeoCompositeShape":
            return
        try:
            cap = float(sh.Capacity())
        except Exception:
            return
        if not cap:
            return
        rel = abs(occv - cap) / abs(cap)
        if rel > rec.get("shareMaxRelDev", 0.0):
            rec["shareMaxRelDev"] = rel
        if rel > self.share_worst[0]:
            self.share_worst = (rel, str(vol.GetName()))

    # ------------------------------------------------------------------

    def _solid_for(self, vol):
        """The OCCT solid of a volume's own shape, or (None, reason)."""
        sh = vol.GetShape()
        if sh is None or vol.IsAssembly() or sh.ClassName() == "TGeoShapeAssembly":
            return None, "pure assembly: no solid of its own, by design"
        try:
            occ = shape_to_occ(sh, SCALE_TO_MM)
        except ShapeDeclined as e:
            return None, str(e)
        except Exception as e:                                   # OCCT can throw
            return None, f"{sh.ClassName()}: OCCT raised {type(e).__name__}: {e}"
        return occ, None

    def _verify(self, vol, occ, rec):
        sh = vol.GetShape()
        try:
            cap = float(sh.Capacity())
        except Exception:
            cap = None
        rec["capacity_cm3"] = cap
        if not self.opts.verify:
            return
        try:
            v_cm3 = solid_volume_mm3(occ) / 1000.0
        except Exception as e:
            rec["occVolume_cm3"] = None
            rec["verifyError"] = str(e)
            return
        rec["occVolume_cm3"] = v_cm3
        if cap and abs(cap) > 0:
            rec["relDev"] = abs(v_cm3 - cap) / abs(cap)

    # ------------------------------------------------------------------

    def build(self, vol, depth=0):
        """Return the XCAF label for `vol`, building it (once) if needed."""
        return self.definitions[self.build_def(vol, depth)][0]

    def build_def(self, vol, depth=0, mirrored=False):
        """The definition id of `vol`; `self.definitions[id]` is (label, solid).

        `mirrored` asks for the volume's Z-mirrored prototype instead of the volume
        itself; see `child_location`.
        """
        a = obj_id(vol)
        k = (a, mirrored)
        hit = self._byvol.get(k)
        if hit is not None:
            return hit
        if a not in self._seen_vols:
            self._seen_vols.add(a)
            self.nvolumes += 1
        did = self._build_def(vol, depth, mirrored)
        self._byvol[k] = did
        return did

    def _own_solid(self, vol, wanted, mirrored):
        """The volume's own OCCT solid, Z-mirrored if this is the prototype."""
        if not wanted:
            return None, "excluded by --include-name"
        # A hollowed volume contributes structure only, with or without daughters.
        if str(vol.GetName()) in self.hollow:
            return None, "hollow volume (--hollow-volume)"
        occ, reason = self._solid_for(vol)
        if occ is None or not mirrored:
            return occ, reason
        try:
            return mirror_solid_z(occ, f"{vol.GetName()}: mirrored prototype"), None
        except ShapeDeclined as e:
            return None, str(e)

    def _build_def(self, vol, depth, mirrored=False):
        name = str(vol.GetName())
        nd = int(vol.GetNdaughters())
        descend = nd > 0
        wanted = (self.opts.include_name is None
                  or fnmatch.fnmatch(name, self.opts.include_name))
        sig = self._shape_sig(vol)

        if not descend:
            did = self._kid(("leaf", name, sig, wanted, mirrored))
            if did in self.definitions:
                self._note_shared(did, vol)
                return did
            occ, reason = self._own_solid(vol, wanted, mirrored)
            emitted = self._emit_name(vol, ("shape", sig)) if occ is not None else name
            if occ is not None and mirrored:
                emitted += "__mirrored"
                self.nbaked += 1
            rec = self._record(did, vol, emitted, mirrored=mirrored,
                               converted=occ is not None, reason=reason)
            if occ is None:
                self.definitions[did] = (None, None)
                return did
            self._verify(vol, occ, rec)
            lab = self.shape_tool.AddShape(occ, False)
            TDataStd_Name.Set(lab, emitted)
            self.definitions[did] = (lab, occ)
            return did

        # A volume with daughters becomes an assembly; children first, so the key can quote them.
        occ, reason = self._own_solid(vol, wanted, mirrored)
        emit_body = (occ is not None and self.opts.mother_bodies
                     and not (depth == 0 and self.opts.skip_top_body))
        plan = []
        for i in range(nd):
            node = vol.GetNode(i)
            child = node.GetVolume()
            mat, tr = tgeo_matrix_components(node.GetMatrix())
            t, cmir, lmat, ltr, dev, corr = child_location(mirrored, mat, tr)
            if _det3(mat) < 0.0:
                self.reflected_nodes.append(f"{name}/{node.GetName()}")
            if cmir:
                self.nmirrored_components += 1
            self._note_orthonormalised(f"{name}/{node.GetName()}", dev, corr)
            if t is None:
                # Not an isometry -- a genuine non-uniform scale.  There is no
                # prototype for that, so it stays a baked private copy.
                self._bake_scaled(vol, node, child, depth, plan, dev)
                continue
            cdid = self.build_def(child, depth + 1, cmir)
            if self.definitions[cdid][0] is None:
                continue
            plan.append((cdid, self._world_key(lmat, ltr), str(node.GetName()), t))

        did = self._kid(("asm", name, sig, wanted, emit_body, mirrored,
                         bool(self.opts.carve_mothers),
                         tuple((p[0], p[1]) for p in plan)))
        if did in self.definitions:
            self._note_shared(did, vol)
            return did

        base = self._emit_name(vol, ("vol", obj_id(vol)))
        mir = "__mirrored" if mirrored else ""
        emitted = base + mir
        rec = self._record(did, vol, emitted, mirrored=mirrored,
                           converted=occ is not None, reason=reason)
        if occ is not None:
            self._verify(vol, occ, rec)

        asm = self.shape_tool.NewShape()
        TDataStd_Name.Set(asm, emitted)
        ncomp0 = self.ncomponents
        placed_children = []
        for (cdid, _mk, nodename, t) in plan:
            comp = self.shape_tool.AddComponent(asm, self.definitions[cdid][0],
                                                TopLoc_Location(t))
            placed_children.append((self.definitions[cdid][1], t))
            TDataStd_Name.Set(comp, nodename)
            self.ncomponents += 1

        if emit_body:
            body = occ
            if self.opts.carve_mothers:
                carved, complete = self._carve(occ, placed_children, name)
                body = carved or occ
                rec["carveComplete"] = bool(complete and carved is not None)
            blab = self.shape_tool.AddShape(body, False)
            TDataStd_Name.Set(blab, f"{base}__body{mir}")
            comp = self.shape_tool.AddComponent(asm, blab, TopLoc_Location(gp_Trsf()))
            TDataStd_Name.Set(comp, f"{base}__body{mir}")
            self.ncomponents += 1
            rec["bodyComponent"] = f"{base}__body{mir}"
        elif occ is not None:
            rec["bodyComponent"] = None
            rec["reason"] = "mother solid omitted (--no-mother-bodies/--skip-top-body)"
            rec["converted"] = False
        if self.ncomponents == ncomp0:
            # every child declined and there is no body: an empty XCAF label reads
            # back as a leaf holding an empty compound, so drop it instead.
            self.shape_tool.RemoveShape(asm)
            self.definitions[did] = (None, occ)
            return did
        self.definitions[did] = (asm, occ)
        return did

    def _bake_scaled(self, vol, node, child, depth, plan, dev=None):
        """A non-uniformly scaling placement: bake it, as there is no prototype."""
        self._note_scaled(f"{vol.GetName()}/{node.GetName()}", child, dev)
        cdid = self.build_def(child, depth + 1, False)
        csolid = self.definitions[cdid][1]
        cname = self.records.get(cdid, {}).get("emittedName", str(child.GetName()))
        if csolid is None:
            self._record(cdid, child, cname,
                         reason="scaling placement of a volume with daughters "
                                "cannot be baked")
            return
        try:
            baked = apply_tgeo_matrix(csolid, node.GetMatrix(), "scaling placement")
        except ShapeDeclined as e:
            self._record(cdid, child, cname, reason=str(e))
            return
        blab = self.shape_tool.AddShape(baked, False)
        TDataStd_Name.Set(blab, f"{cname}__scaled")
        sdid = self._kid(("scaled", obj_id(node)))
        self.definitions[sdid] = (blab, baked)
        plan.append((sdid, ("scaled", obj_id(node)), str(node.GetName()), gp_Trsf()))

    # ------------------------------------------------------------------

    @staticmethod
    def _world_key(mat, tr):
        return (tuple(round(mat[i][j], 9) for i in range(3) for j in range(3))
                + tuple(round(v, 6) for v in tr))

    @staticmethod
    def _compose(pmat, ptr, cmat, ctr):
        """Compose a parent world transform with a child's (matrix, translation), in mm."""
        mat = [[sum(pmat[i][k] * cmat[k][j] for k in range(3)) for j in range(3)]
               for i in range(3)]
        tr = [sum(pmat[i][k] * ctr[k] for k in range(3)) + ptr[i] for i in range(3)]
        return mat, tr

    def _shared_definition(self, vol, kind, mirrored=False):
        """The shared XCAF definition of a volume's own solid (built once).

        `kind` is "leaf" for a volume without daughters and "body" for the mother
        solid of one with daughters; both are keyed on the volume's *name and shape
        value*, never on the name alone.  `mirrored` asks for the Z-mirrored
        prototype, which every reflected use of the volume shares.
        """
        sig = self._shape_sig(vol)
        did = self._kid((kind, str(vol.GetName()), sig, True, mirrored))
        if did in self.definitions:
            self._note_shared(did, vol)
            return did
        occ, reason = self._own_solid(vol, True, mirrored)
        # A mother body shares the slot of its own assembly label, so the two carry
        # one disambiguated base name between them.
        slot = ("vol", obj_id(vol)) if kind == "body" else ("shape", sig)
        emitted = self._emit_name(vol, slot) if occ is not None \
            else str(vol.GetName())
        if kind == "body":
            emitted = emitted + "__body"
        if occ is not None and mirrored:
            emitted = emitted + "__mirrored"
            self.nbaked += 1
        # (body first, then the mirror suffix: `X__body__mirrored`)
        rec = self._record(did, vol, emitted, mirrored=mirrored,
                           converted=occ is not None, reason=reason)
        if occ is None:
            self.definitions[did] = (None, None)
            return did
        self._verify(vol, occ, rec)
        if kind == "body":
            rec["bodyComponent"] = emitted
        lab = self.shape_tool.AddShape(occ, False)
        TDataStd_Name.Set(lab, emitted)
        self.definitions[did] = (lab, occ)
        return did

    def build_world(self, vol, depth=0, wmat=None, wtr=None, path="",
                    mirrored=False):
        """Per-occurrence walk that drops coincident (definition, world transform) pairs.

        Assembly labels are one per occurrence and leaf solids one per (name, shape value,
        mirrored).  `wmat`/`wtr` are the volume's TGeo world transform (the coincidence key);
        `mirrored` says whether the label is the Z-mirrored prototype.
        """
        if wmat is None:
            wmat = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            wtr = [0., 0., 0.]
        name = str(vol.GetName())
        a = obj_id(vol)
        if a not in self._seen_vols:
            self._seen_vols.add(a)
            self.nvolumes += 1
        nd = int(vol.GetNdaughters())
        descend = nd > 0
        if not (self.opts.include_name is None
                or fnmatch.fnmatch(name, self.opts.include_name)):
            return None

        if not descend:
            did = self._shared_definition(vol, "leaf", mirrored)
            lab = self.definitions[did][0]
            if lab is None:
                return None
            key = (did, self._world_key(wmat, wtr))
            if key in self.placed_world:
                self.ndropped += 1
                if len(self.dropped_examples) < 50:
                    self.dropped_examples.append(path)
                return None
            self.placed_world.add(key)
            return lab

        # Surviving children first: an occurrence with none must not leave an empty assembly label.
        comps = []
        for i in range(nd):
            node = vol.GetNode(i)
            child = node.GetVolume()
            mat, tr = tgeo_matrix_components(node.GetMatrix())
            cmat, ctr = self._compose(wmat, wtr, mat, tr)
            t, cmir, _lmat, _ltr, dev, corr = child_location(mirrored, mat, tr)
            if _det3(mat) < 0.0:
                self.reflected_nodes.append(f"{name}/{node.GetName()}")
            if cmir:
                self.nmirrored_components += 1
            self._note_orthonormalised(f"{name}/{node.GetName()}", dev, corr)
            if t is None:
                # Not an isometry -- a genuine non-uniform scale.  There is no
                # prototype for that, so it stays a baked private copy.
                self._bake_scaled_world(child, node, cmat, ctr, comps, dev)
                continue
            clab = self.build_world(child, depth + 1, cmat, ctr,
                                    f"{path}/{node.GetName()}", cmir)
            if clab is None:
                continue
            comps.append((clab, t, str(node.GetName())))

        if (self.opts.mother_bodies
                and not (depth == 0 and self.opts.skip_top_body)):
            bdid = self._shared_definition(vol, "body", mirrored)
            blab = self.definitions[bdid][0]
            if blab is not None:
                key = (bdid, self._world_key(wmat, wtr))
                if key in self.placed_world:
                    self.ndropped += 1
                else:
                    self.placed_world.add(key)
                    comps.append((blab, gp_Trsf(),
                                  self.records[bdid]["emittedName"]))

        if not comps:
            return None
        asm = self.shape_tool.NewShape()
        TDataStd_Name.Set(asm, self._occ_asm_name(vol)
                          + ("__mirrored" if mirrored else ""))
        for (clab, t, cname) in comps:
            comp = self.shape_tool.AddComponent(asm, clab, TopLoc_Location(t))
            TDataStd_Name.Set(comp, cname)
            self.ncomponents += 1
        return asm

    def _bake_scaled_world(self, child, node, cmat, ctr, comps, dev=None):
        """A non-uniformly scaling placement in the per-occurrence walk."""
        self._note_scaled(str(node.GetName()), child, dev)
        cdid = self._shared_definition(child, "leaf", False)
        csolid = self.definitions[cdid][1]
        cname = self.records.get(cdid, {}).get("emittedName", str(child.GetName()))
        if csolid is None:
            self._record(cdid, child, cname,
                         reason="scaling placement of a volume with daughters "
                                "cannot be baked")
            return
        key = (("scaled", cdid), self._world_key(cmat, ctr))
        if key in self.placed_world:
            self.ndropped += 1
            return
        try:
            baked = apply_tgeo_matrix(csolid, node.GetMatrix(), "scaling placement")
        except ShapeDeclined as e:
            self._record(cdid, child, cname, reason=str(e))
            return
        self.placed_world.add(key)
        blab = self.shape_tool.AddShape(baked, False)
        TDataStd_Name.Set(blab, f"{cname}__scaled")
        comps.append((blab, gp_Trsf(), str(node.GetName())))

    # ------------------------------------------------------------------

    def _carve(self, mother, placed, name):
        """Subtract the placed daughters from the mother; return (body, every daughter subtracted).

        An assembly daughter (`None` in `placed`) has no solid to subtract; the reverse
        converter nests exactly the mothers whose carve is incomplete.
        """
        missing = sum(1 for (sh, _t) in placed if sh is None)
        cutters = [_moved(sh, t) for (sh, t) in placed if sh is not None]
        if not cutters:
            if missing:
                self.log(f"  [WARN] {name}: {missing} daughter(s) are assemblies and have "
                         f"no solid to subtract; the mother stays uncarved")
            return mother, not missing
        try:
            tool = cutters[0]
            for c in cutters[1:]:
                tool = _boolean(BRepAlgoAPI_Fuse, tool, c, "carve fuse")
            carved = _boolean(BRepAlgoAPI_Cut, mother, tool, "carve cut")
        except ShapeDeclined as e:
            self.log(f"  [WARN] {name}: carving failed ({e}); keeping the uncarved mother")
            return mother, False
        if missing:
            # Carve every daughter or none: a nested partial carve leaves daughters outside the mother.
            self.log(f"  [WARN] {name}: {missing} of {len(placed)} daughter(s) are "
                     f"assemblies and have no solid to subtract; discarding the partial "
                     f"carve and keeping the mother whole, to be nested instead")
            return mother, False
        return carved, True

    # ------------------------------------------------------------------

    def write(self, path):
        self.shape_tool.UpdateAssemblies()
        Interface_Static.SetCVal("write.step.schema", "AP214IS")
        Interface_Static.SetCVal("write.step.unit", "MM")
        Interface_Static.SetCVal("write.step.product.name", "O2_TGeoToCAD")
        w = STEPCAFControl_Writer()
        w.SetNameMode(True)
        w.SetColorMode(False)
        w.SetLayerMode(False)
        if not w.Transfer(self.doc):
            raise RuntimeError("STEPCAFControl_Writer.Transfer failed")
        if w.Write(path) != IFSelect_RetDone:
            raise RuntimeError(f"STEP write failed for {path}")

    def media_sidecar(self, source):
        """The media sidecar: every medium used, and which emitted STEP part wears which."""
        parts = {}
        for r in self.records.values():
            if not r.get("medium"):
                continue
            name = r.get("emittedName")
            # Only a part emitted as a solid wears a medium; a mother's lives in its `__body` leaf.
            is_body = bool(name) and (name.endswith("__body")
                                      or name.endswith("__body__mirrored"))
            if name and r.get("converted") and (r.get("ndaughters", 0) == 0 or is_body):
                parts[name] = r["medium"]
            # The mother's `__body` leaf carries its material.
            if r.get("bodyComponent"):
                parts[r["bodyComponent"]] = r["medium"]
        # Which emitted part is the body of which assembly, so the reverse converter can nest.
        bodies = {}
        carved_complete = {}
        for r in self.records.values():
            if r.get("bodyComponent") and r.get("emittedName"):
                bodies[r["bodyComponent"]] = r["emittedName"]
                if "carveComplete" in r:
                    carved_complete[r["emittedName"]] = bool(r["carveComplete"])

        return {
            "generator": "O2_TGeoToCAD.py",
            "source": os.path.abspath(source),
            "mediumParamOrder": list(MEDIUM_PARAM_NAMES),
            "bodyOfAssembly": bodies,
            # --carve-mothers only: True means every daughter was subtracted, so do not nest.
            "carvedComplete": carved_complete,
            "nBodies": len(bodies),
            "nMedia": len(self.media),
            "nParts": len(parts),
            "media": self.media,
            "parts": parts,
        }

    def report(self, source, out_step):
        by_class = {}
        npure = 0
        for r in self.records.values():
            pure = r["isAssembly"] or r["shapeClass"] == "TGeoShapeAssembly"
            c = by_class.setdefault(r["shapeClass"], {"converted": 0, "declined": 0,
                                                      "pureAssembly": 0, "reasons": {}})
            if r["converted"]:
                c["converted"] += 1
            elif pure:
                c["pureAssembly"] += 1
                npure += 1
            else:
                c["declined"] += 1
                key = (r["reason"] or "unknown").split(":")[0]
                c["reasons"][key] = c["reasons"].get(key, 0) + 1
        recs = sorted(self.records.values(),
                      key=lambda r: (r["name"], r.get("emittedName") or ""))
        devs = [r["relDev"] for r in recs if r.get("relDev") is not None]
        disambiguated = {}
        for base, slots in self._name_slots.items():
            if len(slots) > 1:
                disambiguated[base] = sorted(set(slots.values()))
        return {
            "source": os.path.abspath(source),
            "output": os.path.abspath(out_step),
            "scaleToMm": SCALE_TO_MM,
            "generator": "O2_TGeoToCAD.py",
            "wallSeconds": round(time.time() - self.t0, 2),
            "volumesVisited": self.nvolumes,
            "definitions": sum(1 for r in recs if r["converted"]),
            "pureAssemblies": npure,
            "declined": sum(1 for r in recs if not r["converted"]
                            and not (r["isAssembly"] or r["shapeClass"] == "TGeoShapeAssembly")),
            "assemblies": sum(1 for r in recs if r["ndaughters"] > 0),
            "components": self.ncomponents,
            "mirroredPrototypes": self.nbaked,
            "mirroredComponents": self.nmirrored_components,
            "scaledPlacementsBaked": self.nscaled,
            "scaledPlacements": self.scaled_records[:50],
            "orthonormalisedPlacements": self.northo,
            "maxOrthogonalityDeviation": self.ortho_worst[0],
            "maxRotationCorrection": self.ortho_worst[1],
            "worstOrthogonalityPlacement": self.ortho_worst[2],
            "orthonormalisations": self.ortho_records[:50],
            "coincidentPlacementsDropped": self.ndropped,
            "coincidentPlacementExamples": self.dropped_examples,
            "reflectedPlacements": self.reflected_nodes[:50],
            "nReflectedPlacements": len(self.reflected_nodes),
            "maxRelDev": max(devs) if devs else None,
            "medianRelDev": sorted(devs)[len(devs) // 2] if devs else None,
            "hollowVolumes": sorted(self.hollow),
            "hollowTag": self.hollow_tag or None,
            "nameDisambiguation": disambiguated,
            "nDisambiguatedNames": len(disambiguated),
            "sharedDefinitionMaxRelDev": self.share_worst[0],
            "sharedDefinitionWorstVolume": self.share_worst[1],
            "byShapeClass": by_class,
            "volumes": recs,
        }


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _cap_check(label, tgeo_shape, band, results, expect_fail=False, occ=None):
    """One capacity-parity check: BRepGProp on our solid vs TGeoShape::Capacity()."""
    try:
        if occ is None:
            occ = shape_to_occ(tgeo_shape, SCALE_TO_MM)
        v = solid_volume_mm3(occ) / 1000.0
        cap = float(tgeo_shape.Capacity())
        rel = abs(v - cap) / abs(cap) if cap else float("inf")
        ok = rel <= band
    except Exception as e:
        v, cap, rel, ok = None, None, None, False
        label = f"{label} [{type(e).__name__}: {e}]"
    passed = (ok != expect_fail)
    results.append((label, passed, cap, v, rel))
    return passed


def _print_suite(title, results):
    fails = [r for r in results if not r[1]]
    print(f"\n--- {title}: {len(results)} checks, {len(fails)} failures")
    for (label, ok, cap, v, rel) in results:
        mark = "ok  " if ok else "FAIL"
        if rel is None:
            print(f"  [{mark}] {label}")
        else:
            print(f"  [{mark}] {label:44s} TGeo {cap:14.6f}  OCC {v:14.6f}  rel {rel:.3e}")
    return len(fails)


def self_test():
    import ROOT
    ROOT.gROOT.SetBatch(True)
    import array

    total = 0
    failures = 0

    import random
    rngc = random.Random(4242)

    def mc_volume(shape, n=200000):
        dx, dy, dz = shape.GetDX(), shape.GetDY(), shape.GetDZ()
        o = shape.GetOrigin()
        ox, oy, oz = o[0], o[1], o[2]
        vbox = 8.0 * dx * dy * dz
        pt = array.array("d", [0.0, 0.0, 0.0])
        hits = 0
        for _ in range(n):
            pt[0] = ox + rngc.uniform(-dx, dx)
            pt[1] = oy + rngc.uniform(-dy, dy)
            pt[2] = oz + rngc.uniform(-dz, dz)
            if shape.Contains(pt):
                hits += 1
        pf = hits / float(n)
        return pf * vbox, vbox * math.sqrt(max(pf * (1.0 - pf), 1e-15) / n)


    # ---- suite 1: primitives, analytic Capacity() ----
    band = 1e-9
    r1 = []
    _cap_check("TGeoBBox(1,2,3)", ROOT.TGeoBBox("b", 1, 2, 3), band, r1)
    _cap_check("TGeoTube(0,2,5)", ROOT.TGeoTube("t0", 0, 2, 5), band, r1)
    _cap_check("TGeoTube(1,2,5) rmin>0", ROOT.TGeoTube("t1", 1, 2, 5), band, r1)
    _cap_check("TGeoTubeSeg(1,2,5,30,150)", ROOT.TGeoTubeSeg("ts", 1, 2, 5, 30, 150), band, r1)
    _cap_check("TGeoTubeSeg(0,2,5,200,340)", ROOT.TGeoTubeSeg("ts2", 0, 2, 5, 200, 340), band, r1)
    _cap_check("TGeoCone(3,0,2,0,4)", ROOT.TGeoCone("c0", 3, 0, 2, 0, 4), band, r1)
    _cap_check("TGeoCone(3,1,2,0.5,4) rmin>0", ROOT.TGeoCone("c1", 3, 1, 2, 0.5, 4), band, r1)
    _cap_check("TGeoConeSeg(2,.5,1,.7,1.5,30,150)",
               ROOT.TGeoConeSeg("cs", 2, .5, 1, .7, 1.5, 30, 150), band, r1)
    _cap_check("TGeoEltu(2,3,4)", ROOT.TGeoEltu("e", 2, 3, 4), band, r1)
    _cap_check("TGeoTorus(10,0,2)", ROOT.TGeoTorus("to0", 10, 0, 2, 0, 360), band, r1)
    _cap_check("TGeoTorus(10,1,2) hollow", ROOT.TGeoTorus("to1", 10, 1, 2, 0, 360), band, r1)
    _cap_check("TGeoTorus(10,1,2,45,120) wedge",
               ROOT.TGeoTorus("to2", 10, 1, 2, 45, 120), band, r1)
    _cap_check("TGeoTrd1(1,2,3,4)", ROOT.TGeoTrd1("d1", 1, 2, 3, 4), band, r1)
    _cap_check("TGeoTrd2(1,2,3,4,5)", ROOT.TGeoTrd2("d2", 1, 2, 3, 4, 5), band, r1)
    _cap_check("TGeoSphere(0,2) full", ROOT.TGeoSphere("s0", 0, 2, 0, 180, 0, 360), band, r1)
    _cap_check("TGeoSphere(1,2) shell", ROOT.TGeoSphere("s1", 1, 2, 0, 180, 0, 360), band, r1)
    _cap_check("TGeoSphere(0,2,30,120) theta",
               ROOT.TGeoSphere("s2", 0, 2, 30, 120, 0, 360), band, r1)
    _cap_check("TGeoSphere(1,2,30,120,20,200)",
               ROOT.TGeoSphere("s3", 1, 2, 30, 120, 20, 200), band, r1)
    _cap_check("TGeoCtub(0,1,1) straight",
               ROOT.TGeoCtub("ct", 0, 1, 1, 0, 360, 0, 0, -1, 0, 0, 1), band, r1)
    arb8v = array.array("d", [-1, -1, -1, 1, 1, 1, 1, -1, -2, -2, -2, 2, 2, 2, 2, -2])
    _cap_check("TGeoArb8 (pyramid frustum)", ROOT.TGeoArb8("a8", 1.0, arb8v), band, r1)
    _cap_check("TGeoTrap(2,0,0,1,1,1,0,1,1,1,0)",
               ROOT.TGeoTrap("tp", 2, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0), band, r1)
    x = ROOT.TGeoXtru(2)
    x.DefinePolygon(4, array.array("d", [0, 0, 2, 2]), array.array("d", [0, 1, 1, 0]))
    x.DefineSection(0, -1, 0, 0, 1)
    x.DefineSection(1, 1, 0.5, 0, 2)
    _cap_check("TGeoXtru (4-gon, scaled+offset)", x, band, r1)
    pg = ROOT.TGeoPgon("pg", 0, 360, 6, 2)
    pg.DefineSection(0, -1, 0, 1)
    pg.DefineSection(1, 1, 0, 1)
    _cap_check("TGeoPgon(0,360,6) solid", pg, band, r1)
    pg2 = ROOT.TGeoPgon("pg2", 10, 90, 3, 2)
    pg2.DefineSection(0, -1, 0.5, 1)
    pg2.DefineSection(1, 1, 0.5, 1)
    _cap_check("TGeoPgon(10,90,3) hollow wedge", pg2, band, r1)
    pg3 = ROOT.TGeoPgon("pg3", 0, 360, 8, 3)
    pg3.DefineSection(0, -2, 0.5, 1)
    pg3.DefineSection(1, 0, 0.5, 2)
    pg3.DefineSection(2, 2, 0.8, 2)
    _cap_check("TGeoPgon(0,360,8) hollow stack", pg3, band, r1)
    pc = ROOT.TGeoPcon("pc", 0, 360, 3)
    pc.DefineSection(0, -1, 0, 1)
    pc.DefineSection(1, 0, 0, 2)
    pc.DefineSection(2, 1, 0, 2)
    _cap_check("TGeoPcon(0,360) rmin=0", pc, band, r1)
    pc2 = ROOT.TGeoPcon("pc2", 0, 360, 3)
    pc2.DefineSection(0, -1, 0.5, 1)
    pc2.DefineSection(1, 0, 0.5, 2)
    pc2.DefineSection(2, 1, 0.8, 2)
    _cap_check("TGeoPcon(0,360) rmin>0", pc2, band, r1)
    pc3 = ROOT.TGeoPcon("pc3", 20, 150, 4)
    pc3.DefineSection(0, -3, 0.5, 1)
    pc3.DefineSection(1, -1, 0.5, 2)
    pc3.DefineSection(2, -1, 1.2, 2)      # a zero-thickness radius jump
    pc3.DefineSection(3, 2, 1.2, 1.8)
    _cap_check("TGeoPcon(20,150) wedge + z-jump", pc3, band, r1)
    total += len(r1)
    failures += _print_suite("primitives vs TGeoShape::Capacity(), band 1e-9", r1)

    # ---- suite 2: composites, against an independent Monte-Carlo of TGeo itself ----
    # Composite Capacity() is itself an MC estimate: require the OCCT volume within 4 sigma of our own MC.
    r2 = []
    _keep = [ROOT.TGeoBBox("ca", 2, 2, 2), ROOT.TGeoTube("cb", 0, 1, 3)]
    tr = ROOT.TGeoTranslation("shift", 3, 0, 0)
    tr.RegisterYourself()
    rot = ROOT.TGeoRotation("rot90", 0, 90, 0)
    rot.RegisterYourself()
    composites = [
        # TGeoCtub's z extent follows its cut planes, so it is scored by MC too.
        ("cut tube, slanted (TGeoCtub)",
         ROOT.TGeoCtub("ct2", 0, 1, 1, 0, 360, 0, -0.6, -0.8, 0, 0.6, 0.8)),
        ("box - tube (subtraction)", ROOT.TGeoCompositeShape("sub", "ca - cb")),
        ("box * tube (intersection)", ROOT.TGeoCompositeShape("inter", "ca * cb")),
        ("box + shifted tube (union)", ROOT.TGeoCompositeShape("uni", "ca + cb:shift")),
        ("(box - tube) + shifted tube (nested)",
         ROOT.TGeoCompositeShape("nest", "(ca - cb) + cb:shift")),
        ("box - rotated tube (rotated operand)",
         ROOT.TGeoCompositeShape("rotsub", "ca - cb:rot90")),
    ]
    for (label, cs) in composites:
        try:
            occ = shape_to_occ(cs, SCALE_TO_MM)
            v = solid_volume_mm3(occ) / 1000.0
            vmc, sig = mc_volume(cs)
            ok = abs(v - vmc) <= 4.0 * sig
            print(f"    {label:40s} OCC {v:10.6f}  MC {vmc:10.6f} +- {sig:.4f}"
                  f"  ({abs(v - vmc) / sig:.2f} sigma)")
        except Exception as e:
            ok = False
            label = f"{label} [{type(e).__name__}: {e}]"
        r2.append((label, ok, None, None, None))
    # the control on the control: a 1% wrong volume must be outside 4 sigma
    _, cs0 = composites[0]
    vmc0, sig0 = mc_volume(cs0)
    r2.append((f"a +2% wrong volume would be rejected ({0.02 * vmc0 / sig0:.1f} sigma)",
               abs(1.02 * vmc0 - vmc0) > 4.0 * sig0, None, None, None))
    # A depth-40 union chain of 41 disjoint boxes must convert, with a closed-form volume.
    chain = ROOT.TGeoBBox("chain0", 1, 1, 1)
    ROOT.SetOwnership(chain, False)
    for i in range(1, 41):
        box = ROOT.TGeoBBox(f"chain{i}", 1, 1, 1)
        shift = ROOT.TGeoTranslation(f"chainT{i}", 2.5 * i, 0, 0)
        node = ROOT.TGeoUnion(chain, box, ROOT.nullptr, shift)
        for obj in (box, shift, node):
            ROOT.SetOwnership(obj, False)
        chain = ROOT.TGeoCompositeShape(f"chainC{i}", node)
        ROOT.SetOwnership(chain, False)
    try:
        v_chain = solid_volume_mm3(shape_to_occ(chain, SCALE_TO_MM)) / 1000.0
        ok_chain = abs(v_chain - 41 * 8.0) <= 1.0e-9 * 41 * 8.0
        chain_detail = f"OCC {v_chain:.9f} vs closed form {41 * 8.0}"
    except Exception as e:
        ok_chain, chain_detail = False, f"{type(e).__name__}: {e}"
    r2.append((f"a depth-40 union chain converts exactly ({chain_detail})", ok_chain,
               None, None, None))
    # ... and the guard still refuses loudly past the real bound.
    try:
        shape_to_occ(chain, SCALE_TO_MM, MAX_BOOLEAN_DEPTH)
        guarded, guard_msg = False, "no exception"
    except ShapeDeclined as e:
        guarded, guard_msg = str(MAX_BOOLEAN_DEPTH) in str(e), str(e)
    r2.append((f"the depth guard still refuses past {MAX_BOOLEAN_DEPTH}", guarded,
               None, None, None))
    total += len(r2)
    failures += _print_suite("composites vs an independent MC of TGeo (N=200k, 4 sigma)", r2)

    # ---- suite 2b: point-by-point Contains agreement, TGeo vs OCCT ----
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_IN
    r2b = []
    for (label, cs) in composites[:2] + [("pcon rmin>0", None)]:
        if cs is None:
            cs = ROOT.TGeoPcon("pcx", 0, 360, 3)
            cs.DefineSection(0, -1, 0.5, 1)
            cs.DefineSection(1, 0, 0.5, 2)
            cs.DefineSection(2, 1, 0.8, 2)
        occ = shape_to_occ(cs, SCALE_TO_MM)
        clf = BRepClass3d_SolidClassifier(occ)
        dx, dy, dz = cs.GetDX(), cs.GetDY(), cs.GetDZ()
        o = cs.GetOrigin()
        pt = array.array("d", [0.0, 0.0, 0.0])
        bad = skipped = 0
        ntot = 3000
        for _ in range(ntot):
            pt[0] = o[0] + rngc.uniform(-dx, dx)
            pt[1] = o[1] + rngc.uniform(-dy, dy)
            pt[2] = o[2] + rngc.uniform(-dz, dz)
            tin = bool(cs.Contains(pt))
            if cs.Safety(pt, tin) < 1e-6:      # on the surface: not a fair question
                skipped += 1
                continue
            clf.Perform(gp_Pnt(pt[0] * SCALE_TO_MM, pt[1] * SCALE_TO_MM,
                               pt[2] * SCALE_TO_MM), 1e-7)
            if (clf.State() == TopAbs_IN) != tin:
                bad += 1
        print(f"    {label:40s} {ntot - skipped} scored, {bad} disagreements"
              f" ({skipped} within 1e-6 cm of a surface)")
        r2b.append((f"Contains agrees, TGeo vs OCCT: {label}", bad == 0, None, None, None))
    total += len(r2b)
    failures += _print_suite("Contains agreement, TGeo vs OCCT classifier (3000 pts each)", r2b)

    # ---- suite 3: placement transforms ----
    r3 = []
    rng = random.Random(20260822)
    for k, m in enumerate([
        ROOT.TGeoTranslation("t", 1.5, -2.5, 3.5),
        ROOT.TGeoRotation("r", 30, 40, 50),
        ROOT.TGeoCombiTrans("ct", 1, 2, 3, ROOT.TGeoRotation("r2", 11, 22, 33)),
    ]):
        t = _isometry_trsf(*tgeo_matrix_components(m), proper_only=True)[0]
        worst = 0.0
        for _ in range(200):
            loc = [rng.uniform(-5, 5) for _ in range(3)]
            mas = array.array("d", [0, 0, 0])
            m.LocalToMaster(array.array("d", loc), mas)
            p = gp_Pnt(loc[0] * SCALE_TO_MM, loc[1] * SCALE_TO_MM, loc[2] * SCALE_TO_MM)
            p.Transform(t)
            worst = max(worst,
                        abs(p.X() - mas[0] * SCALE_TO_MM),
                        abs(p.Y() - mas[1] * SCALE_TO_MM),
                        abs(p.Z() - mas[2] * SCALE_TO_MM))
        ok = worst < 1e-9
        r3.append((f"{m.ClassName()} LocalToMaster vs gp_Trsf (200 pts, mm)", ok,
                   None, None, None))
        print(f"    worst |delta| = {worst:.3e} mm")
    # a reflection must be refused as a rigid placement and offered as a GTrsf
    refl = ROOT.TGeoRotation("refl")
    refl.ReflectZ(True)
    r3.append(("reflecting TGeoRotation refused as a gp_Trsf",
               _isometry_trsf(*tgeo_matrix_components(refl), proper_only=True)[0] is None,
               None, None, None))
    box = shape_to_occ(ROOT.TGeoBBox("rb", 1, 2, 3), SCALE_TO_MM)
    mirrored = apply_tgeo_matrix(box, refl, "reflection test")
    r3.append(("reflected box keeps its volume (baked as an exact isometry)",
               abs(solid_volume_mm3(mirrored) - solid_volume_mm3(box)) < 1e-6,
               None, None, None))
    total += len(r3)
    failures += _print_suite("placement transforms", r3)

    # ---- suite 4: negative controls ----
    r4 = []
    # each of these compares a deliberately WRONG TGeo shape against our solid for
    # the RIGHT one; the band must reject it, or the band proves nothing.
    good_tube = shape_to_occ(ROOT.TGeoTube("ngt", 1, 2, 5), SCALE_TO_MM)
    _cap_check("wrong rmin: Capacity(0,2,5) vs solid(1,2,5) must FAIL",
               ROOT.TGeoTube("ngt2", 0, 2, 5), 1e-9, r4, expect_fail=True, occ=good_tube)
    good_pcon = shape_to_occ(pc2, SCALE_TO_MM)
    pc2b = ROOT.TGeoPcon("pc2b", 0, 360, 3)
    pc2b.DefineSection(0, -1, 0.5, 1)
    pc2b.DefineSection(1, 0, 0.5, 2)
    pc2b.DefineSection(2, 1, 0.9, 2)          # rmin 0.8 -> 0.9
    _cap_check("wrong pcon rmin (0.8 -> 0.9) must FAIL",
               pc2b, 1e-9, r4, expect_fail=True, occ=good_pcon)
    good_pgon = shape_to_occ(pg, SCALE_TO_MM)
    pgb = ROOT.TGeoPgon("pgb", 0, 360, 7, 2)   # 6 -> 7 edges
    pgb.DefineSection(0, -1, 0, 1)
    pgb.DefineSection(1, 1, 0, 1)
    _cap_check("wrong pgon nedges (6 -> 7) must FAIL",
               pgb, 1e-9, r4, expect_fail=True, occ=good_pgon)
    # and a control on the control: the band accepts the right answer
    _cap_check("same pgon accepted (control on the control)", pg, 1e-9, r4, occ=good_pgon)
    total += len(r4)
    failures += _print_suite("negative controls (a wrong parameter must be rejected)", r4)

    # ---- suite 5: analytic surface types ----
    # The carriers must be the analytic surfaces TGeo meant, not B-splines.
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
        GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
        GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface, GeomAbs_OtherSurface)
    _SNAME = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
              GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus",
              GeomAbs_BezierSurface: "bezier", GeomAbs_BSplineSurface: "bspline",
              GeomAbs_SurfaceOfRevolution: "revolution",
              GeomAbs_SurfaceOfExtrusion: "extrusion",
              GeomAbs_OffsetSurface: "offset", GeomAbs_OtherSurface: "other"}

    def face_types(shape):
        import collections as _c
        c = _c.Counter()
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            c[_SNAME.get(BRepAdaptor_Surface(topods.Face(ex.Current())).GetType(), "?")] += 1
            ex.Next()
        return dict(c)

    pgf = ROOT.TGeoPgon("pgf", 0, 360, 6, 2)
    pgf.DefineSection(0, -1, 0.5, 1)
    pgf.DefineSection(1, 1, 0.5, 1)
    vtw = array.array("d", [-1, -1, -1, 1, 1, 1, 1, -1,
                            -1.5, -0.5, -0.5, 1.5, 1.5, 0.5, 0.5, -1.5])
    r6 = []
    for nm, tsh, want in [
        ("TGeoBBox", ROOT.TGeoBBox("fb", 1, 2, 3), {"plane": 6}),
        ("TGeoTube", ROOT.TGeoTube("ft", 1, 2, 5), {"plane": 2, "cylinder": 2}),
        ("TGeoTubeSeg", ROOT.TGeoTubeSeg("fts", 1, 2, 5, 30, 150),
         {"plane": 4, "cylinder": 2}),
        ("TGeoCone", ROOT.TGeoCone("fc", 3, 1, 2, 0.5, 4), {"plane": 2, "cone": 2}),
        ("TGeoPcon", pc2, {"plane": 2, "cylinder": 2, "cone": 2}),
        ("TGeoSphere", ROOT.TGeoSphere("fs", 1, 2, 30, 120, 20, 200),
         {"sphere": 2, "cone": 2, "plane": 2}),
        ("TGeoTorus", ROOT.TGeoTorus("fto", 10, 1, 2, 45, 120), {"torus": 2, "plane": 2}),
        ("TGeoEltu", ROOT.TGeoEltu("fe", 2, 3, 4), {"extrusion": 1, "plane": 2}),
        ("TGeoTrd1", ROOT.TGeoTrd1("fd1", 1, 2, 3, 4), {"plane": 6}),
        ("TGeoTrd2", ROOT.TGeoTrd2("fd2", 1, 2, 3, 4, 5), {"plane": 6}),
        ("TGeoXtru", x, {"plane": 6}),
        ("TGeoPgon hollow", pgf, {"plane": 14}),
        ("TGeoTrap", ROOT.TGeoTrap("ftp", 2, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0), {"plane": 6}),
        ("TGeoArb8 planar", ROOT.TGeoArb8("fa8", 1.0, arb8v), {"plane": 6}),
        ("TGeoArb8 twisted", ROOT.TGeoArb8("fa8t", 1.0, vtw), {"plane": 2, "bspline": 4}),
        ("TGeoCtub", ROOT.TGeoCtub("fct", 0, 1, 1, 0, 360, 0, -0.6, -0.8, 0, 0.6, 0.8),
         {"cylinder": 1, "plane": 2}),
    ]:
        try:
            got = face_types(shape_to_occ(tsh, SCALE_TO_MM))
            ok = got == want
            if not ok:
                nm = f"{nm} (got {got}, want {want})"
        except Exception as e:
            ok = False
            nm = f"{nm} [{type(e).__name__}: {e}]"
        r6.append((f"{nm}", ok, None, None, None))
    total += len(r6)
    failures += _print_suite("analytic surface types of every face", r6)

    # ---- suite 6: the XCAF document, written and read back ----
    r5 = []
    import tempfile
    mgr = ROOT.TGeoManager("stmgr", "self-test")
    vac = mgr.MakeBox("world", ROOT.nullptr, 50, 50, 50)
    mgr.SetTopVolume(vac)
    inner = mgr.MakeTube("innertube", ROOT.nullptr, 1, 2, 5)
    vac.AddNode(inner, 1, ROOT.TGeoTranslation(3, 0, 0))
    vac.AddNode(inner, 2, ROOT.TGeoTranslation(-3, 0, 0))
    grp = mgr.MakeVolumeAssembly("grp")
    leaf = mgr.MakeBox("leafbox", ROOT.nullptr, 1, 1, 1)
    grp.AddNode(leaf, 1, ROOT.TGeoTranslation(0, 4, 0))
    vac.AddNode(grp, 1, ROOT.TGeoTranslation(0, 0, 7))
    mgr.CloseGeometry()

    class _O:
        pass
    o = _O()
    o.quiet = True
    o.verify = True
    o.mother_bodies = True
    o.skip_top_body = False
    o.carve_mothers = False
    o.include_name = None
    o.dedup_world = False
    conv = TGeoToStep(o)
    conv.build(mgr.GetTopVolume())
    tmp = os.path.join(tempfile.mkdtemp(), "selftest.step")
    conv.write(tmp)
    rep = conv.report("in-memory", tmp)
    r5.append(("STEP file written", os.path.getsize(tmp) > 0, None, None, None))
    r5.append(("one definition per logical volume (3 shaped, 2 assemblies)",
               rep["definitions"] == 3 and rep["assemblies"] == 2, None, None, None))
    r5.append(("shared tube emitted once, placed twice",
               conv.ncomponents == 5, None, None, None))

    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    d2 = TDocStd_Document("rb")
    rd = STEPCAFControl_Reader()
    rd.SetNameMode(True)
    r5.append(("STEP reads back", rd.ReadFile(tmp) == IFSelect_RetDone, None, None, None))
    rd.Transfer(d2)
    st2 = XCAFDoc_DocumentTool.ShapeTool(d2.Main())
    roots = TDF_LabelSequence()
    st2.GetFreeShapes(roots)
    r5.append(("exactly one free shape (the top assembly)",
               roots.Length() == 1, None, None, None))
    names = []
    leaves = []

    def walk(lb):
        ch = TDF_LabelSequence()
        st2.GetComponents(lb, ch)
        if ch.Length() == 0:
            leaves.append(lb)
            names.append(lb.GetLabelName())
            return
        for i in range(ch.Length()):
            c = ch.Value(i + 1)
            if st2.IsReference(c):
                ref = TDF_Label()
                st2.GetReferredShape(c, ref)
                walk(ref)
            else:
                walk(c)

    walk(roots.Value(1))
    r5.append(("names survive the write/read (innertube present)",
               "innertube" in names, None, None, None))
    r5.append(("the mother body is a named leaf (world__body)",
               "world__body" in names, None, None, None))
    r5.append((f"leaf occurrences == placements ({len(leaves)} == 4)",
               len(leaves) == 4, None, None, None))
    total += len(r5)
    failures += _print_suite("XCAF assembly document, written and read back", r5)

    # ---- suite 7: the definition cache is keyed on identity, not on the name ----
    # Volume names need not be unique: check the keying and the value signature sharing rests on.
    r7 = []
    sig_a = shape_signature(ROOT.TGeoTube("sgA", 1, 2, 5))
    sig_b = shape_signature(ROOT.TGeoTube("sgB", 1, 2, 5))
    sig_c = shape_signature(ROOT.TGeoTube("sgC", 0, 2, 5))
    r7.append(("two equal tubes have equal signatures", sig_a == sig_b,
               None, None, None))
    r7.append(("a wrong rmin changes the signature (negative control)",
               sig_a != sig_c, None, None, None))
    sgp1 = ROOT.TGeoPcon("sgp1", 0, 360, 3)
    sgp2 = ROOT.TGeoPcon("sgp2", 0, 360, 3)
    for p, rin in ((sgp1, 0.5), (sgp2, 0.9)):
        p.DefineSection(0, -1, 0.5, 1)
        p.DefineSection(1, 0, 0.5, 2)
        p.DefineSection(2, 1, rin, 2)
    r7.append(("a wrong pcon inner radius changes the signature (negative control)",
               shape_signature(sgp1) != shape_signature(sgp2), None, None, None))
    _kc = [ROOT.TGeoBBox("kca", 2, 2, 2), ROOT.TGeoTube("kcb", 0, 1, 3)]
    cs1 = ROOT.TGeoCompositeShape("kc1", "kca - kcb")
    cs2 = ROOT.TGeoCompositeShape("kc2", "kca - kcb")
    r7.append(("a composite is keyed on its address, never shared by value",
               shape_signature(cs1) != shape_signature(cs2)
               and shape_signature(cs1) == shape_signature(cs1), None, None, None))

    mgr2 = ROOT.TGeoManager("stmgr2", "name-collision self-test")
    w2 = mgr2.MakeBox("nworld", ROOT.nullptr, 50, 50, 50)
    mgr2.SetTopVolume(w2)
    dupA = mgr2.MakeTube("dup", ROOT.nullptr, 0, 2, 5)     # two volumes, one name,
    dupB = mgr2.MakeTube("dup", ROOT.nullptr, 0, 1, 5)     # four times the volume
    same1 = mgr2.MakeBox("same", ROOT.nullptr, 1, 1, 1)    # two volumes, one name,
    same2 = mgr2.MakeBox("same", ROOT.nullptr, 1, 1, 1)    # one shape
    w2.AddNode(dupA, 1, ROOT.TGeoTranslation(-10, 0, 0))
    w2.AddNode(dupB, 1, ROOT.TGeoTranslation(10, 0, 0))
    w2.AddNode(same1, 1, ROOT.TGeoTranslation(0, -10, 0))
    w2.AddNode(same2, 1, ROOT.TGeoTranslation(0, 10, 0))
    mgr2.CloseGeometry()
    conv2 = TGeoToStep(o)
    conv2.build(mgr2.GetTopVolume())
    rep2 = conv2.report("in-memory", "none")
    emitted2 = sorted(r["emittedName"] for r in rep2["volumes"] if r["converted"])
    r7.append((f"one name, two shapes -> two definitions {emitted2}",
               emitted2 == ["dup", "dup#2", "nworld", "same"], None, None, None))
    r7.append(("the disambiguation is recorded in the report",
               rep2["nameDisambiguation"] == {"dup": ["dup", "dup#2"]},
               None, None, None))
    r7.append(("one name, one shape -> still one definition, placed twice",
               rep2["definitions"] == 4 and conv2.ncomponents == 5,
               None, None, None))
    devs2 = [r["relDev"] for r in rep2["volumes"] if r.get("relDev") is not None]
    shared2 = rep2["sharedDefinitionMaxRelDev"]
    print(f"    every definition vs its own volume's Capacity(): worst "
          f"{max(devs2):.3e}, worst over a *shared* definition {shared2:.3e}")
    r7.append(("every volume gets a solid that is its own shape",
               max(devs2) <= 1e-9 and shared2 <= 1e-9, None, None, None))
    capA = float(dupA.GetShape().Capacity())
    capB = float(dupB.GetShape().Capacity())
    ratio = abs(capA - capB) / capB
    print(f"    a name-keyed cache would have given one of them the other's solid:"
          f" {capA:.6f} vs {capB:.6f} cm3, {ratio:.2f} relative")
    r7.append((f"the test could have failed: the two shapes differ by {ratio:.2f}",
               ratio > 1e-2, None, None, None))
    total += len(r7)
    failures += _print_suite("definition cache keyed on volume identity", r7)

    # ---- suite 8: baking a reflection is an isometry, and keeps the carriers ----
    # Volume and carriers are asserted; the gp_GTrsf route is the negative control.
    r8 = []
    mtube = ROOT.TGeoTube("mt", 4, 5, 10)
    occ_t = shape_to_occ(mtube, SCALE_TO_MM)
    v_t = solid_volume_mm3(occ_t)
    f_t = face_types(occ_t)
    mir_t = mirror_solid_z(occ_t, "self-test tube")
    v_m = solid_volume_mm3(mir_t)
    f_m = face_types(mir_t)
    rel_t = abs(v_m - v_t) / v_t
    g = gp_GTrsf()
    g.SetVectorialPart(gp_Mat(1, 0, 0, 0, 1, 0, 0, 0, -1))
    old = BRepBuilderAPI_GTransform(occ_t, g, True).Shape()
    v_o = solid_volume_mm3(old)
    f_o = face_types(old)
    rel_o = abs(v_o - v_t) / v_t
    print(f"    tube {f_t} -> gp_Trsf mirror {f_m}, rel {rel_t:.3e}")
    print(f"    the retired gp_GTrsf route: {f_o}, rel {rel_o:.3e}")
    r8.append((f"a mirrored tube keeps its volume (rel {rel_t:.3e})",
               rel_t <= 1e-12, None, None, None))
    r8.append((f"a mirrored tube keeps its analytic faces {f_m}",
               f_m == f_t and sum(f_m.get(k, 0) for k in
                                  ("bspline", "bezier", "revolution")) == 0,
               None, None, None))
    r8.append((f"the retired gp_GTrsf route is wrong by {rel_o:.3e} and all "
               f"B-spline (negative control)",
               rel_o > 1e-3 and f_o.get("bspline", 0) == 4, None, None, None))
    mpc = ROOT.TGeoPcon("mpc", 0, 360, 3)
    mpc.DefineSection(0, -1, 0.5, 1)
    mpc.DefineSection(1, 0, 0.5, 2)
    mpc.DefineSection(2, 1, 0.8, 2)
    occ_p = shape_to_occ(mpc, SCALE_TO_MM)
    mir_p = mirror_solid_z(occ_p, "self-test pcon")
    cap_p = float(mpc.Capacity())
    rel_p = abs(solid_volume_mm3(mir_p) / 1000.0 - cap_p) / cap_p
    r8.append((f"a mirrored Pcon matches its analytic Capacity() ({rel_p:.3e})",
               rel_p <= 1e-9, None, None, None))
    r8.append(("a mirrored Pcon keeps its analytic faces",
               face_types(mir_p) == face_types(occ_p), None, None, None))
    rotxz = ROOT.TGeoRotation("rotxz_st", 90., 0., 90., 90., 180., 0.)
    baked = apply_tgeo_matrix(occ_t, rotxz, "self-test rotxz")
    r8.append(("apply_tgeo_matrix takes a real reflecting TGeoRotation exactly",
               abs(solid_volume_mm3(baked) - v_t) <= 1e-12 * v_t
               and face_types(baked) == f_t, None, None, None))
    r8.append(("the mirrored solid is not inside out",
               _signed_volume(mir_t) > 0, None, None, None))
    bad = gp_Trsf()
    bad.SetScale(gp_Pnt(0, 0, 0), 1.01)
    try:
        apply_isometry(occ_t, bad, "not an isometry")
        caught = False
    except ShapeDeclined:
        caught = True
    r8.append(("the volume invariant rejects a transform that is not an isometry "
               "(negative control)", caught, None, None, None))
    # A real hand-written rotation must be snapped, not refused.
    sloppy = [[+0.681268213, 0.0, +0.732033940],
              [0.0, 1.0, 0.0],
              [-0.732033894, 0.0, +0.681268164]]     # TRD BM49/B051_1, verbatim
    dev0 = orthogonality_deviation(sloppy)
    fixed, dev1, corr = orthonormalise(sloppy)
    print(f"    TRD BM49/B051_1: |M^T M - I| {dev0:.3e} -> "
          f"{orthogonality_deviation(fixed):.3e}, rotation moved by {corr:.3e}")
    r8.append((f"a hand-written rotation is snapped to an exact one "
               f"({dev0:.2e} -> {orthogonality_deviation(fixed):.2e})",
               dev0 > 1e-9 and orthogonality_deviation(fixed) < 1e-14
               and 0.0 < corr < 1e-6, None, None, None))
    exact_rot = tgeo_matrix_components(ROOT.TGeoRotation("orr", 30, 40, 50))[0]
    r8.append((f"an exact rotation is left alone to the double-precision floor "
               f"({orthonormalise(exact_rot)[2]:.1e})",
               orthonormalise(exact_rot)[2] <= 1e-15, None, None, None))
    sloppy_refl = [[r[0], r[1], -r[2]] for r in sloppy]
    r8.append(("the snap keeps a reflection a reflection",
               _det3(orthonormalise(sloppy_refl)[0]) < 0, None, None, None))
    r8.append(("a genuine non-uniform scale is still refused as a placement "
               "(negative control)",
               _isometry_trsf([[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]],
                              [0., 0., 0.], True)[0] is None, None, None, None))
    total += len(r8)
    failures += _print_suite("mirror baking: exact isometry, analytic carriers", r8)

    # ---- suite 9: a reflected subtree is emitted, and lands where TGeo puts it --
    # TGeoManager's world matrix is the oracle for where the mirrored leaves land.
    r9 = []
    mgr3 = ROOT.TGeoManager("stmgr3", "reflected-subtree self-test")
    w3 = mgr3.MakeBox("rworld", ROOT.nullptr, 100, 100, 100)
    mgr3.SetTopVolume(w3)
    grp3 = mgr3.MakeVolumeAssembly("rgrp")        # an assembly: no solid to bake
    rtube = mgr3.MakeTube("rtube", ROOT.nullptr, 1, 2, 5)
    rbox = mgr3.MakeBox("rbox", ROOT.nullptr, 1, 2, 3)
    rflip = mgr3.MakeBox("rflip", ROOT.nullptr, 1, 1, 4)
    refl3 = ROOT.TGeoRotation("reflz3")
    refl3.ReflectZ(True)
    grp3.AddNode(rtube, 1, ROOT.TGeoTranslation(0, 0, 7))
    grp3.AddNode(rbox, 1, ROOT.TGeoCombiTrans(3, 0, 2,
                                              ROOT.TGeoRotation("rr3", 20, 30, 40)))
    grp3.AddNode(rflip, 1, ROOT.TGeoCombiTrans(0, 4, 1, refl3))   # already mirrored
    w3.AddNode(grp3, 1, ROOT.TGeoTranslation(0, 0, 20))
    w3.AddNode(grp3, 2, ROOT.TGeoCombiTrans(0, 0, -20, refl3))
    mgr3.CloseGeometry()
    conv3 = TGeoToStep(o)
    conv3.build(mgr3.GetTopVolume())
    tmp3 = os.path.join(tempfile.mkdtemp(), "reflected.step")
    conv3.write(tmp3)

    d3 = TDocStd_Document("rb3")
    rd3 = STEPCAFControl_Reader()
    rd3.SetNameMode(True)
    rd3.ReadFile(tmp3)
    rd3.Transfer(d3)
    st3 = XCAFDoc_DocumentTool.ShapeTool(d3.Main())
    roots3 = TDF_LabelSequence()
    st3.GetFreeShapes(roots3)
    from OCC.Core.TopLoc import TopLoc_Location as _TL
    found3 = {}

    def _walk3(lab, loc):
        ch = TDF_LabelSequence()
        st3.GetComponents(lab, ch)
        if ch.Length() == 0:
            t = loc.Transformation()
            mat = [[t.Value(i + 1, j + 1) for j in range(3)] for i in range(3)]
            tr = [t.Value(i + 1, 4) for i in range(3)]
            nm = str(lab.GetLabelName())
            if nm.endswith("__mirrored"):
                mat = [[mat[i][0], mat[i][1], -mat[i][2]] for i in range(3)]
            found3.setdefault(nm, []).append((mat, tr))
            return
        for i in range(ch.Length()):
            c = ch.Value(i + 1)
            cloc = loc.Multiplied(st3.GetLocation(c))
            if st3.IsReference(c):
                ref = TDF_Label()
                st3.GetReferredShape(c, ref)
                _walk3(ref, cloc)
            else:
                _walk3(c, cloc)

    for i in range(roots3.Length()):
        _walk3(roots3.Value(i + 1), _TL())

    def _tgeo_world(path):
        if not mgr3.cd(path):
            return None
        gm = mgr3.GetCurrentMatrix()
        rr = gm.GetRotationMatrix()
        tt = gm.GetTranslation()
        return ([[float(rr[3 * i + j]) for j in range(3)] for i in range(3)],
                [float(tt[i]) * SCALE_TO_MM for i in range(3)])

    def _worst(step_entries, want):
        best = None
        for (mat, tr) in step_entries:
            d = max(max(abs(mat[i][j] - want[0][i][j]) for j in range(3))
                    for i in range(3))
            d = max(d, max(abs(tr[i] - want[1][i]) for i in range(3)))
            if best is None or d < best:
                best = d
        return best if best is not None else float("inf")

    nleaf3 = sum(len(v) for v in found3.values())
    r9.append((f"exactly one free shape, no orphaned subtree "
               f"({roots3.Length()} root(s))", roots3.Length() == 1,
               None, None, None))
    r9.append((f"every leaf occurrence is emitted ({nleaf3} == 7)",
               nleaf3 == 7, None, None, None))
    for (nm, path, mirror_expected) in (
            ("rtube", "/rworld_1/rgrp_2/rtube_1", True),
            ("rbox", "/rworld_1/rgrp_2/rbox_1", True),
            ("rflip", "/rworld_1/rgrp_2/rflip_1", False),
            ("rtube", "/rworld_1/rgrp_1/rtube_1", False)):
        want = _tgeo_world(path)
        key3 = nm + ("__mirrored" if mirror_expected else "")
        got = found3.get(key3, [])
        d = _worst(got, want) if want else float("inf")
        r9.append((f"{path} lands where TGeo puts it, as `{key3}` (worst "
                   f"|delta| {d:.2e} mm)", bool(got) and d < 1e-9,
                   None, None, None))
    # Negative control: the prototype placed at M instead of M*Z must land elsewhere.
    want = _tgeo_world("/rworld_1/rgrp_2/rbox_1")
    wrong = [([[m[i][0], m[i][1], -m[i][2]] for i in range(3)], t)
             for (m, t) in found3.get("rbox__mirrored", [])]
    dwrong = _worst(wrong, want)
    r9.append((f"the un-conjugated convention would be wrong by {dwrong:.2e} mm "
               f"(negative control)", dwrong > 1e-6, None, None, None))
    # rflip is reflected inside rgrp, so the parities multiply: each prototype shows up once.
    r9.append(("a reflection under a reflection is the plain volume again",
               len(found3.get("rflip", [])) == 1
               and len(found3.get("rflip__mirrored", [])) == 1,
               None, None, None))
    r9.append((f"3 mirrored solid definitions carry 4 mirrored components, rather "
               f"than one bake per placement ({conv3.nbaked}, "
               f"{conv3.nmirrored_components})",
               conv3.nbaked == 3 and conv3.nmirrored_components == 4,
               None, None, None))
    total += len(r9)
    failures += _print_suite("reflected subtrees: mirrored prototypes, placed", r9)

    # ---- suite 10: the TGeoPgon z-step, and the collinear-face guard ----------
    # Two sections at one z give collinear closure points; the Newell-area guard keeps the shell valid.
    r10 = []
    shift10 = 1.5 / math.sin(math.radians(10.0))
    pg10 = ROOT.TGeoPgon("st_zstep", 0.0, 20.0, 1, 4)
    pg10.DefineSection(0, -3.5, 86.3 - shift10, 240.4 - shift10)
    pg10.DefineSection(1, -1.5, 86.3 - shift10, 240.4 - shift10)
    pg10.DefineSection(2, -1.5, 86.3 - shift10, 243.4 - shift10)
    pg10.DefineSection(3, 3.5, 86.3 - shift10, 243.4 - shift10)
    occ10 = shape_to_occ(pg10)
    from OCC.Core.BRepCheck import BRepCheck_Analyzer as _BCA10
    r10.append(("a z-step TGeoPgon builds a VALID shell (TPC_WSEG's tpc_hole)",
                _BCA10(occ10).IsValid(), None, None, None))
    v10 = solid_volume_mm3(occ10) / 1000.0
    d10 = abs(v10 - pg10.Capacity()) / pg10.Capacity()
    r10.append((f"its volume matches Capacity() ({d10:.2e})", d10 < 1e-9, None, None, None))
    r10.append(("three collinear points yield no face (the guard, negative control)",
                _quad_face((0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 0, 0), "st") is None,
                None, None, None))
    r10.append(("a genuine triangle still yields a face",
                _quad_face((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), "st") is not None,
                None, None, None))
    total += len(r10)
    failures += _print_suite("TGeoPgon z-step and the collinear-face guard", r10)

    # ---- the media sidecar -------------------------------------------------
    # No medium may lose a parameter and no mixture an element; Air_F/Air_NF differ only in ifield.
    r11 = []
    mgr11 = ROOT.TGeoManager("mediatest", "media sidecar")
    mix11 = ROOT.TGeoMixture("Air", 4, 0.00120479)
    mix11.AddElement(12.0107, 6.0, 0.000124)
    mix11.AddElement(14.0067, 7.0, 0.755267)
    mix11.AddElement(15.9994, 8.0, 0.231781)
    mix11.AddElement(39.9480, 18.0, 0.012827)
    fe11 = ROOT.TGeoMaterial("Fe", 55.85, 26.0, 7.87)
    med_f = ROOT.TGeoMedium("Air_F", 1, mix11, ROOT.nullptr)
    med_f.SetParam(1, 1.0)      # ifield: in field
    med_f.SetParam(4, 0.75)     # stemax
    med_n = ROOT.TGeoMedium("Air_NF", 2, mix11, ROOT.nullptr)
    med_n.SetParam(1, 0.0)      # ifield: field free
    med_n.SetParam(4, 0.75)
    med_fe = ROOT.TGeoMedium("Fe", 3, fe11, ROOT.nullptr)

    top11 = mgr11.MakeBox("top", med_f, 50, 50, 50)
    mgr11.SetTopVolume(top11)
    a11 = mgr11.MakeBox("a", med_n, 5, 5, 5)
    b11 = mgr11.MakeBox("b", med_fe, 5, 5, 5)
    top11.AddNode(a11, 1, ROOT.TGeoTranslation(10, 0, 0))
    top11.AddNode(b11, 1, ROOT.TGeoTranslation(-10, 0, 0))
    mgr11.CloseGeometry()

    class _O11:
        pass
    o11 = _O11()
    o11.quiet = True
    o11.verify = False
    o11.mother_bodies = True
    o11.skip_top_body = False
    o11.carve_mothers = False
    o11.include_name = None
    o11.dedup_world = False
    o11.hollow_volumes = []
    o11.hollow_tag = None
    conv11 = TGeoToStep(o11)
    conv11.build(top11)
    side11 = conv11.media_sidecar("in-memory")

    r11.append((f"every emitted part carries a medium ({side11['nParts']} parts)",
                side11["nParts"] >= 3, None, None, None))
    r11.append(("a mother's own body leaf is in the table, not just its assembly "
                "label (else every mother comes back transparent)",
                side11["parts"].get("top__body") == "Air_F", None, None, None))
    r11.append((f"the three media are collected once each ({side11['nMedia']})",
                side11["nMedia"] == 3, None, None, None))
    r11.append(("each part names its own medium",
                side11["parts"].get("top__body") == "Air_F"
                and side11["parts"].get("a") == "Air_NF"
                and side11["parts"].get("b") == "Fe", None, None, None))
    mf, mn = side11["media"]["Air_F"], side11["media"]["Air_NF"]
    r11.append(("the in-field and field-free twins differ ONLY in ifield",
                mf["params"]["ifield"] == 1.0 and mn["params"]["ifield"] == 0.0
                and all(mf["params"][k] == mn["params"][k]
                        for k in MEDIUM_PARAM_NAMES if k != "ifield"),
                None, None, None))
    r11.append(("all eight medium parameters are recorded, in Geant's order",
                list(mf["params"]) == list(MEDIUM_PARAM_NAMES), None, None, None))
    r11.append(("stemax survives (the parameter a medium loses most quietly)",
                abs(mf["params"]["stemax"] - 0.75) < 1e-12, None, None, None))
    r11.append(("a mixture keeps all four elements with their weights",
                mf["material"]["isMixture"] and mf["material"]["nElements"] == 4
                and abs(sum(e["W"] for e in mf["material"]["elements"]) - 1.0) < 1e-6,
                None, None, None))
    r11.append(("a plain material is not reported as a mixture, and keeps Z/A/rho",
                not side11["media"]["Fe"]["material"]["isMixture"]
                and side11["media"]["Fe"]["material"]["Z"] == 26.0
                and abs(side11["media"]["Fe"]["material"]["density"] - 7.87) < 1e-9,
                None, None, None))
    r11.append(("radiation and interaction length are carried, not recomputed",
                mf["material"]["radLen"] > 0.0 and mf["material"]["intLen"] > 0.0,
                None, None, None))
    r11.append(("the sidecar says which part is the body of which assembly, so "
                "the converter can put the mother/daughter nesting back",
                side11["bodyOfAssembly"].get("top__body") == "top",
                None, None, None))
    # --hollow-volume: the hall is structure the CAD run already has.
    o11.hollow_volumes = ["top"]
    conv11h = TGeoToStep(o11)
    conv11h.build(top11)
    side11h = conv11h.media_sidecar("in-memory")
    r11.append(("a hollow volume emits no body of its own",
                "top__body" not in side11h["parts"], None, None, None))
    r11.append(("its daughters are still emitted, with their media",
                side11h["parts"].get("a") == "Air_NF"
                and side11h["parts"].get("b") == "Fe", None, None, None))
    r11.append(("hollowing is opt-in: the same build without it keeps the body "
                "(negative control)",
                side11["parts"].get("top__body") == "Air_F", None, None, None))
    o11.hollow_tag = "MOD"
    conv11t = TGeoToStep(o11)
    conv11t.build(top11)
    side11t = conv11t.media_sidecar("in-memory")
    r11.append(("--hollow-tag renames only the hollowed volume, so two modules "
                "converted from one world do not collide",
                conv11t.records[[d for d in conv11t.records
                                 if conv11t.records[d]["name"] == "top"][0]]
                ["emittedName"].startswith("top_MOD")
                and side11t["parts"].get("a") == "Air_NF", None, None, None))
    # A hollowed volume with NO daughters takes the leaf path, not the assembly one.
    mgr11b = ROOT.TGeoManager("mediatest2", "hollow leaf")
    fe11b = ROOT.TGeoMaterial("Fe2", 55.85, 26.0, 7.87)
    med11b = ROOT.TGeoMedium("Fe2", 1, fe11b, ROOT.nullptr)
    top11b = mgr11b.MakeBox("w", med11b, 50, 50, 50)
    mgr11b.SetTopVolume(top11b)
    leaf11b = mgr11b.MakeBox("hall", med11b, 5, 5, 5)
    keep11b = mgr11b.MakeBox("keepme", med11b, 5, 5, 5)
    top11b.AddNode(leaf11b, 1, ROOT.TGeoTranslation(10, 0, 0))
    top11b.AddNode(keep11b, 1, ROOT.TGeoTranslation(-10, 0, 0))
    mgr11b.CloseGeometry()
    o11.hollow_volumes = ["hall"]
    o11.hollow_tag = None
    conv11b = TGeoToStep(o11)
    conv11b.build(top11b)
    side11b = conv11b.media_sidecar("in-memory")
    r11.append(("a hollowed volume with no daughters is not emitted either",
                "hall" not in side11b["parts"], None, None, None))
    r11.append(("its daughterless sibling still is (negative control)",
                side11b["parts"].get("keepme") == "Fe2", None, None, None))

    # --carve-mothers. These build their own managers, so they come last: gGeoManager follows
    # the most recently created one.
    mgr11c = ROOT.TGeoManager("carvetest", "all-solid daughters")
    fe11c = ROOT.TGeoMaterial("Fe3", 55.85, 26.0, 7.87)
    med11c = ROOT.TGeoMedium("Fe3", 1, fe11c, ROOT.nullptr)
    top11c = mgr11c.MakeBox("cw", med11c, 50, 50, 50)
    mgr11c.SetTopVolume(top11c)
    solid11c = mgr11c.MakeBox("csolid", med11c, 5, 5, 5)
    top11c.AddNode(solid11c, 1)
    mgr11c.CloseGeometry()
    o11.hollow_volumes = []
    o11.hollow_tag = None
    o11.carve_mothers = True
    conv11c = TGeoToStep(o11)
    conv11c.build(top11c)
    side11c = conv11c.media_sidecar("in-memory")
    r11.append(("--carve-mothers reports a mother whose daughters are all solids as "
                "completely carved, so the converter leaves it flat",
                side11c["carvedComplete"].get("cw") is True, None, None, None))
    r11.append(("carving is opt-in: without it the sidecar makes no claim "
                "(negative control)",
                side11["carvedComplete"] == {}, None, None, None))

    # An assembly daughter has no solid to subtract, so the carve is incomplete.
    mgr11d = ROOT.TGeoManager("carvetest2", "assembly daughter")
    fe11d = ROOT.TGeoMaterial("Fe4", 55.85, 26.0, 7.87)
    med11d = ROOT.TGeoMedium("Fe4", 1, fe11d, ROOT.nullptr)
    top11d = mgr11d.MakeBox("dw", med11d, 50, 50, 50)
    mgr11d.SetTopVolume(top11d)
    mother11d = mgr11d.MakeBox("dmother", med11d, 20, 20, 20)
    asm11d = ROOT.TGeoVolumeAssembly("dasm")
    inner11d = mgr11d.MakeBox("dinner", med11d, 2, 2, 2)
    asm11d.AddNode(inner11d, 1)
    mother11d.AddNode(asm11d, 1)
    top11d.AddNode(mother11d, 1)
    mgr11d.CloseGeometry()
    conv11d = TGeoToStep(o11)
    conv11d.build(top11d)
    side11d = conv11d.media_sidecar("in-memory")
    r11.append(("a mother whose daughter is an assembly reports an INCOMPLETE carve, "
                "so the converter keeps nesting it",
                side11d["carvedComplete"].get("dmother") is False, None, None, None))

    # An incomplete carve must return the mother whole; asked of _carve directly.
    _cbox = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
    _ccut = _moved(BRepPrimAPI_MakeBox(2.0, 2.0, 2.0).Shape(), gp_Trsf())
    _full = conv11d._carve(_cbox, [(_ccut, gp_Trsf())], "all-solid")
    _part = conv11d._carve(_cbox, [(_ccut, gp_Trsf()), (None, gp_Trsf())], "mixed")
    r11.append(("a carve with every daughter subtracted returns a new, smaller body",
                _full[1] is True and _full[0] is not _cbox, None, None, None))
    r11.append(("a carve that cannot subtract every daughter returns the mother "
                "WHOLE, so nesting stays correct",
                _part[1] is False and _part[0] is _cbox, None, None, None))
    o11.carve_mothers = False
    total += len(r11)
    failures += _print_suite("the media/material sidecar", r11)

    print(f"\n{total} checks, {failures} failures")
    sys.stdout.flush()
    sys.stderr.flush()
    # PyROOT double-frees loose TGeoShapes at teardown; leave first so the exit status is the verdict.
    os._exit(1 if failures else 0)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def load_manager(path):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    geo = ROOT.TGeoManager.Import(path)
    if geo is None:
        raise RuntimeError(f"could not import a TGeoManager from {path}")
    return geo


def main(argv=None):
    ap = argparse.ArgumentParser(description="TGeo -> STEP (AP214) converter")
    ap.add_argument("input", nargs="?", help="ROOT geometry file")
    ap.add_argument("output", nargs="?", help="output .step file")
    ap.add_argument("--report", default=None)
    ap.add_argument("--hollow-volume", dest="hollow_volumes", action="append",
                    default=[], metavar="NAME",
                    help="emit this volume as a pure assembly: its daughters at their "
                         "own transforms, but no body of its own. Repeatable. Meant for "
                         "the experiment hall (cave, barrel, caveRB24), which o2-sim "
                         "builds natively whatever module list is asked for.")
    ap.add_argument("--hollow-tag", default=None, metavar="TAG",
                    help="suffix the name of every --hollow-volume with _TAG. Two "
                         "modules converted from the same world would otherwise emit "
                         "the same hall volume names and collide when placed together.")
    ap.add_argument("--media-json", default=None,
                    help="write the media/material sidecar here (default: "
                         "<output>_media.json). The reverse converter reads it "
                         "with its own --media-json and rebuilds the media "
                         "verbatim instead of using a placeholder.")
    ap.add_argument("--top", default=None)
    ap.add_argument("--include-name", default=None)
    ap.add_argument("--no-mother-bodies", dest="mother_bodies", action="store_false")
    ap.add_argument("--skip-top-body", action="store_true")
    ap.add_argument("--carve-mothers", action="store_true")
    ap.add_argument("--dedup-world", action="store_true")
    ap.add_argument("--no-step", dest="write_step", action="store_false",
                    help="build every solid and write the report, but skip the STEP "
                         "write (which is where OCCT gives out on very large models)")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    opts = ap.parse_args(argv)

    if opts.self_test:
        return self_test()
    if not opts.input or not opts.output:
        ap.error("input and output are required (or use --self-test)")

    geo = load_manager(opts.input)
    if opts.top:
        vol = geo.GetVolume(opts.top)
        if vol is None:
            raise SystemExit(f"no volume named {opts.top}")
    else:
        vol = geo.GetTopVolume()

    conv = TGeoToStep(opts)
    conv.log(f"walking {vol.GetName()} ...")
    if opts.dedup_world:
        conv.build_world(vol)
    else:
        conv.build(vol)
    conv.log(f"  {conv.nvolumes} logical volumes (by identity), "
             f"{len(conv.definitions)} definitions, {conv.ncomponents} components")
    if opts.write_step:
        conv.log(f"  writing {opts.output}")
        conv.write(opts.output)

    rep = conv.report(opts.input, opts.output)
    rpath = opts.report or (os.path.splitext(opts.output)[0] + "_report.json")
    with open(rpath, "w") as f:
        json.dump(rep, f, indent=1)

    media = conv.media_sidecar(opts.input)
    mpath = opts.media_json or (os.path.splitext(opts.output)[0] + "_media.json")
    with open(mpath, "w") as f:
        json.dump(media, f, indent=1)

    print(f"{rep['definitions']} solids, {rep['assemblies']} volumes with daughters, "
          f"{rep['pureAssemblies']} pure assemblies, {rep['components']} components, "
          f"{rep['declined']} volumes declined")
    if rep["maxRelDev"] is not None:
        print(f"capacity check: max relative deviation {rep['maxRelDev']:.3e}, "
              f"median {rep['medianRelDev']:.3e}")
    if rep["coincidentPlacementsDropped"]:
        print(f"{rep['coincidentPlacementsDropped']} coincident placement(s) dropped "
              f"(--dedup-world); e.g. {rep['coincidentPlacementExamples'][:2]}")
    if rep["nReflectedPlacements"]:
        print(f"{rep['nReflectedPlacements']} reflecting placement(s); "
              f"{rep['mirroredComponents']} component(s) place a mirrored prototype, "
              f"drawn from {rep['mirroredPrototypes']} mirrored definition(s)")
    if rep["scaledPlacementsBaked"]:
        print(f"[WARN] {rep['scaledPlacementsBaked']} placement matrix/matrices are not "
              f"isometries and were baked, not placed: "
              f"{[r['placement'] for r in rep['scaledPlacements'][:3]]}")
    if rep["orthonormalisedPlacements"]:
        print(f"{rep['orthonormalisedPlacements']} placement matrix/matrices snapped to "
              f"the nearest rotation; worst |M^T M - I| "
              f"{rep['maxOrthogonalityDeviation']:.3e}, correction "
              f"{rep['maxRotationCorrection']:.3e} ({rep['worstOrthogonalityPlacement']})")
    if rep["nDisambiguatedNames"]:
        ex = sorted(rep["nameDisambiguation"].items())[:3]
        print(f"{rep['nDisambiguatedNames']} TGeo name(s) cover more than one definition "
              f"and were disambiguated; e.g. {ex}")
    if rep["sharedDefinitionMaxRelDev"] > 1e-6:
        print(f"  [WARN] a shared definition disagrees with a sharing volume's own "
              f"capacity by {rep['sharedDefinitionMaxRelDev']:.3e} "
              f"({rep['sharedDefinitionWorstVolume']})")
    for cls, c in sorted(rep["byShapeClass"].items(), key=lambda kv: -(kv[1]["declined"])):
        if c["declined"]:
            print(f"  declined {cls}: {c['declined']} ({c['reasons']})")
    size = (f"{os.path.getsize(opts.output) / 1e6:.2f} MB"
            if opts.write_step else "no STEP written (--no-step)")
    print(f"report: {rpath}   ({rep['wallSeconds']} s, {size})")
    print(f"media:  {mpath}   ({media['nMedia']} media over {media['nParts']} parts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
