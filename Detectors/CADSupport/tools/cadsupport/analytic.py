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

"""Analytic-surface helpers shared by `O2_CADtoTGeo.py` and the `cadsupport` package."""

import math
from typing import List

import numpy as np
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
    GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface, GeomAbs_OtherSurface,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola,
    GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OffsetCurve, GeomAbs_OtherCurve,
)
from OCC.Core.gp import gp_Pnt, gp_Vec

# Names of the OCCT surface and curve types, shared by the converter and the package.
SURFACE_TYPE_NAME = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_OffsetSurface: "offset",
    GeomAbs_OtherSurface: "other",
}

CURVE_TYPE_NAME = {
    GeomAbs_Line: "line",
    GeomAbs_Circle: "circle",
    GeomAbs_Ellipse: "ellipse",
    GeomAbs_Hyperbola: "hyperbola",
    GeomAbs_Parabola: "parabola",
    GeomAbs_BezierCurve: "bezier",
    GeomAbs_BSplineCurve: "bspline",
    GeomAbs_OffsetCurve: "offset",
    GeomAbs_OtherCurve: "other",
}


def _v_dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_cross(a: List[float], b: List[float]) -> List[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _analytic_surface_gap(kind: str, model: dict, P) -> float:
    """The largest distance, in native CAD units, from any sampled point to the candidate surface.

    One quantity for every kind, so an ill-conditioned proposal cannot pass on its own residual.
    """
    if kind == "plane":
        normal = np.asarray(model["normal"], dtype=float)
        normal = normal / np.linalg.norm(normal)
        return float(np.abs((P - np.asarray(model["point"], dtype=float)) @ normal).max())
    if kind == "sphere":
        return float(np.abs(np.linalg.norm(P - model["centre"], axis=1) - model["radius"]).max())
    if kind == "cylinder":
        axis = np.asarray(model["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        radial = P - model["origin"]
        radial = radial - np.outer(radial @ axis, axis)
        return float(np.abs(np.linalg.norm(radial, axis=1) - model["radius"]).max())
    if kind == "cone":
        axis = np.asarray(model["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        rel = P - model["apex"]
        h = rel @ axis
        r = np.linalg.norm(rel - np.outer(h, axis), axis=1)
        half = model["half_angle"]
        return float(np.abs(r * math.cos(half) - h * math.sin(half)).max())
    return float("inf")


def _sample_surface_for_recognition(adaptor, umin: float, umax: float, vmin: float, vmax: float, n: int = 9):
    """Sample an (n x n) grid over the face's actual trimmed (u, v) box (from `breptools.UVBounds`,
    not the underlying surface's full natural domain). Returns (points, unit normals) in *native*
    (unscaled) CAD length units, or (None, None) if unsampleable."""
    if not all(math.isfinite(x) for x in (umin, umax, vmin, vmax)):
        return None, None
    points, normals = [], []
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    for i in range(n):
        u = umin + (umax - umin) * i / (n - 1.0)
        for j in range(n):
            v = vmin + (vmax - vmin) * j / (n - 1.0)
            try:
                adaptor.D1(u, v, p, du, dv)
            except Exception:
                return None, None
            nrm = _v_cross([du.X(), du.Y(), du.Z()], [dv.X(), dv.Y(), dv.Z()])
            length = math.sqrt(_v_dot(nrm, nrm))
            if length < 1e-14:  # parametric degeneracy (pole/seam): skip this sample
                continue
            points.append([p.X(), p.Y(), p.Z()])
            normals.append([c / length for c in nrm])
    if len(points) < 3 * n:
        return None, None
    return np.array(points), np.array(normals)


def _analytic_surface_proposals(P, N):
    """Yield `(kind, model)` for every candidate surface these samples propose, in order of
    parsimony: plane (3 parameters), sphere (4), cylinder (5), cone (6).

    Nothing here decides: degenerate proposals are left in and the gap judges them.
    """
    # --- plane (3): the samples lie in one plane; the frame is the sampled normal itself
    yield "plane", {"normal": N[0] / np.linalg.norm(N[0]), "point": P[0]}

    # --- sphere (4): normal lines concurrent, P_i = C + r*N_i
    A = np.zeros((3 * len(P), 4))
    b = np.zeros(3 * len(P))
    for i in range(len(P)):
        A[3 * i:3 * i + 3, 0:3] = np.eye(3)
        A[3 * i:3 * i + 3, 3] = N[i]
        b[3 * i:3 * i + 3] = P[i]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    yield "sphere", {"centre": sol[:3], "radius": abs(sol[3])}

    # --- cylinder (5): normals coplanar; axis = smallest right singular vector of the normal field
    _, _, Vt = np.linalg.svd(N, full_matrices=False)
    axis = Vt[-1]
    if np.abs(N @ axis).max() < 1e-9:
        e1 = Vt[0]
        e2 = np.cross(axis, e1)
        x, y = P @ e1, P @ e2
        M = np.column_stack([x, y, np.ones_like(x)])
        D, E, F = np.linalg.lstsq(M, -(x ** 2 + y ** 2), rcond=None)[0]
        cx, cy = -D / 2, -E / 2
        r2 = cx * cx + cy * cy - F
        if r2 > 0:
            origin = cx * e1 + cy * e2  # a point on the axis (axial component is free)
            yield "cylinder", {"axis": axis, "refu": e1, "origin": origin,
                               "radius": math.sqrt(r2)}

    # --- cone (6): N_i . (P_i - A) = 0 is linear in the apex A
    apex, *_ = np.linalg.lstsq(N, np.einsum('ij,ij->i', N, P), rcond=None)
    d = P - apex
    dn = np.linalg.norm(d, axis=1)
    ok = dn > 1e-12
    if ok.sum() > 10:
        u = d[ok] / dn[ok, None]
        mean_dir = u.mean(axis=0)
        _, _, Vt2 = np.linalg.svd(u - mean_dir, full_matrices=False)
        ax2 = np.cross(Vt2[0], Vt2[1])
        n2 = np.linalg.norm(ax2)
        if n2 > 1e-12:  # a ruling axis exists
            ax2 = ax2 / n2
            if np.dot(mean_dir, ax2) < 0.0:
                ax2 = -ax2
            ref = u[0] - np.dot(u[0], ax2) * ax2
            refn = np.linalg.norm(ref)
            if refn > 1e-9:
                half_angle = float(np.arccos(np.clip(np.abs(u @ ax2), -1.0, 1.0)).mean())
                yield "cone", {"axis": ax2, "apex": apex, "refu": ref / refn,
                               "half_angle": half_angle}


def _self_test_bezier_patch(fn, nu: int, nv: int):
    """A non-rational Bezier patch whose control net is `fn(s, t)` on a (nu x nv) grid."""
    from OCC.Core.Geom import Geom_BSplineSurface
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    poles = TColgp_Array2OfPnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            x, y, z = fn(i / (nu - 1.0), j / (nv - 1.0))
            poles.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
    uk = TColStd_Array1OfReal(1, 2)
    uk.SetValue(1, 0.0)
    uk.SetValue(2, 1.0)
    vk = TColStd_Array1OfReal(1, 2)
    vk.SetValue(1, 0.0)
    vk.SetValue(2, 1.0)
    um = TColStd_Array1OfInteger(1, 2)
    um.SetValue(1, nu)
    um.SetValue(2, nu)
    vm = TColStd_Array1OfInteger(1, 2)
    vm.SetValue(1, nv)
    vm.SetValue(2, nv)
    surface = Geom_BSplineSurface(poles, uk, vk, um, vm, nu - 1, nv - 1)
    return BRepBuilderAPI_MakeFace(surface, 1e-6).Face()


def _self_test_tapered_near_circle(bulge: float, taper: float):
    """Negative control: a tapered non-circular profile that must never be recognised as a cone."""
    def fn(s, t):
        a = 1.2 * s - 0.6
        r = 9.8 * (1.0 + bulge * math.cos(3.0 * a)) * (1.0 + taper * t)
        return (r * math.cos(a), r * math.sin(a), 10.0 * t)
    return _self_test_bezier_patch(fn, nu=6, nv=3)
