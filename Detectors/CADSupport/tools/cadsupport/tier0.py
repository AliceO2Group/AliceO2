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

"""Tier 0: the plane, cylinder, cone, sphere or torus a stored B-spline face already is.

Proposals come from `analytic._analytic_surface_proposals` plus a torus solve here. A proposal is
admissible when its gap <= REL_TOL * max(diag, 1 cm), measured on samples independent of the ones
it was fitted to, and the fewest-parameter admissible proposal wins.
"""

import math

# The same band as `recognise.REL_TOL`; the emitter self-test asserts the two agree.
REL_TOL = 1.0e-6

# The proposal grid (the converter's own) and the independent, denser acceptance grid.
_PROPOSE_N = 9
_ACCEPT_N = 17


class _Unavailable(Exception):
    """The converter module could not be imported, so nothing here can run."""


_CONVERTER = None


def _converter():
    """`cadsupport.analytic`, imported lazily and kept."""
    global _CONVERTER
    if _CONVERTER is None:
        try:
            from cadsupport import analytic
        except Exception as exc:                                 # noqa: BLE001
            raise _Unavailable(str(exc)) from None
        _CONVERTER = analytic
    return _CONVERTER


# ------------------------------------------------------------------------------------------
# the instrument
# ------------------------------------------------------------------------------------------

def surface_gap(kind, model, points):
    """The largest distance, in cm, from any of `points` to the candidate surface.

    For plane / sphere / cylinder / cone it is `analytic._analytic_surface_gap`.
    """
    if kind == "torus":
        return _torus_gap(points, model)
    return _converter()._analytic_surface_gap(kind, model, points)


def _torus_residual(points, centre, axis, major, minor):
    import numpy as np
    h = (points - centre) @ axis
    rho = np.linalg.norm(points - centre - np.outer(h, axis), axis=1)
    return np.sqrt((rho - major) ** 2 + h ** 2) - minor


def _torus_gap(points, model):
    import numpy as np
    return float(np.abs(_torus_residual(points, model["centre"], model["axis"],
                                        model["major"], model["minor"])).max())


# ------------------------------------------------------------------------------------------
# the torus proposal (the one model the converter's recogniser does not carry)
# ------------------------------------------------------------------------------------------

def _torus_radii(points, centre, axis):
    """`(R, r)` by least squares once the axis is fixed, or None if the solve is not a torus.

    `rho^2 + h^2 = 2R rho + (r^2 - R^2)` is linear in `2R` and `r^2 - R^2`.
    """
    import numpy as np
    h = (points - centre) @ axis
    rho = np.linalg.norm(points - centre - np.outer(h, axis), axis=1)
    design = np.column_stack([rho, np.ones_like(rho)])
    sol, *_ = np.linalg.lstsq(design, rho ** 2 + h ** 2, rcond=None)
    major = 0.5 * float(sol[0])
    minor_sq = float(sol[1]) + major * major
    if not (major > 0.0 and minor_sq > 0.0):
        return None
    return major, math.sqrt(minor_sq)


def _propose_torus(points, normals, refinements=25):
    """`{axis, centre, major, minor}` for the torus these samples propose, or None.

    `(N_i x P_i) . d + N_i . g = 0` with `g = c x d` is linear in `(d, g)`: one SVD gives the axis,
    then Gauss-Newton polishes the gap. A cylinder's degenerate `d = 0` answer is declined.
    """
    import numpy as np

    design = np.column_stack([np.cross(normals, points), normals])
    _, _singular, right = np.linalg.svd(design, full_matrices=False)
    solution = right[-1]
    axis, moment = solution[:3], solution[3:]
    length = float(np.linalg.norm(axis))
    if length < 1.0e-6:
        return None                     # d = 0: coplanar normals, i.e. a cylinder, not a torus
    axis = axis / length
    centre = np.cross(axis, moment / length)

    radii = _torus_radii(points, centre, axis)
    if radii is None:
        return None
    major, minor = radii
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) or 1.0
    step = 1.0e-7 * span
    for _ in range(refinements):
        tangent_a = np.cross(axis, [1.0, 0.0, 0.0])
        if np.linalg.norm(tangent_a) < 1e-6:
            tangent_a = np.cross(axis, [0.0, 1.0, 0.0])
        tangent_a = tangent_a / np.linalg.norm(tangent_a)
        tangent_b = np.cross(axis, tangent_a)
        base = _torus_residual(points, centre, axis, major, minor)

        def at(delta):
            tilted = axis + delta[3] * tangent_a + delta[4] * tangent_b
            tilted = tilted / np.linalg.norm(tilted)
            return _torus_residual(points, centre + delta[:3], tilted,
                                   major + delta[5], minor + delta[6])

        jacobian = np.zeros((len(points), 7))
        for column in range(7):
            probe = np.zeros(7)
            probe[column] = step
            jacobian[:, column] = (at(probe) - base) / step
        try:
            delta, *_ = np.linalg.lstsq(jacobian, -base, rcond=None)
        except np.linalg.LinAlgError:
            break
        if np.abs(at(delta)).max() >= np.abs(base).max():
            break                                    # no longer improving: keep what converged
        centre = centre + delta[:3]
        axis = axis + delta[3] * tangent_a + delta[4] * tangent_b
        axis = axis / np.linalg.norm(axis)
        major += float(delta[5])
        minor += float(delta[6])
    if not (major > 0.0 and minor > 0.0):
        return None
    return {"axis": axis, "centre": centre, "major": float(major), "minor": float(minor)}


# ------------------------------------------------------------------------------------------
# the service
# ------------------------------------------------------------------------------------------

def canonicalise(face, adaptor, scale):
    """`(carrier, gap)`: the canonical carrier this face IS, and the gap that decided.

    `carrier` is None when the face is not canonical, and `gap` is then the best proposal's gap;
    both are None where the face cannot be sampled. `scale` is `max(part diagonal, 1 cm)`. The
    record speaks `_face_records`' vocabulary plus `canonicalised`, `tier0GapCm` and
    `tier0GapRelative`; `uv` is the trim box in the canonical chart, None for a plane or a sphere.
    """
    try:
        conv = _converter()
    except _Unavailable:
        return None, None
    from OCC.Core.BRepTools import breptools

    try:
        uv_bounds = breptools.UVBounds(face)
    except Exception:                                            # noqa: BLE001
        return None, None
    propose_points, propose_normals = conv._sample_surface_for_recognition(
        adaptor, *uv_bounds, n=_PROPOSE_N)
    if propose_points is None:
        return None, None
    accept_points, _accept_normals = conv._sample_surface_for_recognition(
        adaptor, *uv_bounds, n=_ACCEPT_N)
    if accept_points is None:
        return None, None

    # In order of parsimony: plane (3 parameters) < sphere (4) < cylinder (5) < cone (6) < torus (7).
    proposals = list(conv._analytic_surface_proposals(propose_points, propose_normals))
    torus = _propose_torus(propose_points, propose_normals)
    if torus is not None:
        proposals.append(("torus", torus))

    # The gap decides admissibility and the fewest-parameter admissible proposal wins, so a sphere
    # is never taken for a zero-major-radius torus.
    kind, model, gap, best_gap = None, None, None, float("inf")
    for candidate_kind, candidate in proposals:
        try:
            candidate_gap = surface_gap(candidate_kind, candidate, accept_points)
        except Exception:                                        # noqa: BLE001
            continue
        if not math.isfinite(candidate_gap):
            continue
        best_gap = min(best_gap, candidate_gap)
        if kind is None and candidate_gap <= REL_TOL * scale:
            kind, model, gap = candidate_kind, candidate, candidate_gap

    if kind is None:
        return None, (None if not math.isfinite(best_gap) else best_gap)
    record = _carrier_record(kind, model, adaptor, uv_bounds)
    if record is None:
        return None, gap
    record["canonicalised"] = True
    record["tier0GapCm"] = gap
    record["tier0GapRelative"] = gap / scale
    return record, gap


def carrier_side(face, adaptor, carrier):
    """`interior` / `exterior` for a canonicalised face, by `census`'s one rule."""
    from cadsupport import census
    return census.halfspace_side_of(face, adaptor, carrier)


def _carrier_record(kind, model, adaptor, uv_bounds):
    """The canonical carrier as `recognise._face_records` states one."""
    import numpy as np
    if kind == "plane":
        normal = np.asarray(model["normal"], dtype=float)
        normal = normal / np.linalg.norm(normal)
        # Unflipped, i.e. the underlying surface's own normal: both callers apply the face's
        # REVERSED flag themselves, exactly as they do for a native plane.
        return {"kind": "plane", "n": tuple(float(c) for c in normal),
                "p": tuple(float(c) for c in model["point"]), "uv": None}
    if kind == "sphere":
        return {"kind": "sphere", "p": tuple(float(c) for c in model["centre"]),
                "r": float(model["radius"]), "uv": None}
    if kind == "torus":
        axis = _unit_array(model["axis"])
        # A fitted torus brings no reference direction of its own, so one is chosen here and the
        # chart below is measured against that same one -- the two cannot disagree.
        ref = np.asarray(_perpendicular_to(axis), dtype=float)
        chart = _canonical_chart(adaptor, uv_bounds, np.asarray(model["centre"], dtype=float),
                                 axis, ref, semi_angle=None, major=float(model["major"]))
        if chart is None:
            return None
        return {"kind": "torus", "d": tuple(float(c) for c in axis),
                "p": tuple(float(c) for c in model["centre"]),
                "x": tuple(float(c) for c in ref), "r": float(model["major"]),
                "rt": float(model["minor"]), "uv": chart}
    if kind == "cylinder":
        axis = _unit_array(model["axis"])
        origin = np.asarray(model["origin"], dtype=float)
        ref = _orthonormalise(np.asarray(model["refu"], dtype=float), axis)
        if ref is None:
            return None
        chart = _canonical_chart(adaptor, uv_bounds, origin, axis, ref, semi_angle=None)
        if chart is None:
            return None
        return {"kind": "cylinder", "d": tuple(float(c) for c in axis),
                "p": tuple(float(c) for c in origin), "x": tuple(float(c) for c in ref),
                "r": float(model["radius"]), "uv": chart}
    if kind == "cone":
        axis = _unit_array(model["axis"])
        apex = np.asarray(model["apex"], dtype=float)
        ref = _orthonormalise(np.asarray(model["refu"], dtype=float), axis)
        if ref is None:
            return None
        half = float(model["half_angle"])
        if not (1.0e-9 < half < 0.5 * math.pi - 1.0e-9):
            return None
        chart = _canonical_chart(adaptor, uv_bounds, apex, axis, ref, semi_angle=half)
        if chart is None:
            return None
        # Stated at the apex, in OCC's gp_Cone chart: r = RefRadius + v sin(a), t = v cos(a).
        return {"kind": "cone", "d": tuple(float(c) for c in axis),
                "p": tuple(float(c) for c in apex), "x": tuple(float(c) for c in ref),
                "r": 0.0, "a": half, "uv": chart}
    return None


def _unit_array(vec):
    import numpy as np
    v = np.asarray(vec, dtype=float)
    return v / np.linalg.norm(v)


def _orthonormalise(vec, axis):
    import numpy as np
    ref = np.asarray(vec, dtype=float)
    ref = ref - float(ref @ axis) * axis
    length = float(np.linalg.norm(ref))
    if length < 1.0e-9:
        return None
    return ref / length


def _perpendicular_to(axis):
    import numpy as np
    seed = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    ref = _orthonormalise(seed, axis)
    return tuple(float(c) for c in ref)


_CHART_N = 33


def _canonical_chart(adaptor, uv_bounds, origin, axis, ref, semi_angle, major=None):
    """`(umin, umax, vmin, vmax)`: the trim's bounding box in the carrier's OWN chart.

    Measured along the patch's two midlines, where the azimuth is monotone and can be unwrapped.
    """
    import numpy as np
    umin, umax, vmin, vmax = uv_bounds
    umid, vmid = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
    binormal = np.cross(axis, ref)

    def chart_of(u, v):
        try:
            point = adaptor.Value(u, v)
        except Exception:                                        # noqa: BLE001
            return None
        rel = np.array([point.X(), point.Y(), point.Z()]) - origin
        axial = float(rel @ axis)
        perp = rel - axial * axis
        if float(np.linalg.norm(perp)) < 1.0e-30:
            return None
        azimuth = math.atan2(float(perp @ binormal), float(perp @ ref))
        if major is not None:                                    # a torus: the meridian angle
            return azimuth, math.atan2(axial, float(np.linalg.norm(perp)) - major)
        return azimuth, axial if semi_angle is None else axial / math.cos(semi_angle)

    anchor = chart_of(umid, vmid)
    if anchor is None:
        return None
    phis, axials = [anchor[0]], [anchor[1]]
    for fixed, lo, hi, along_u in ((vmid, umin, umax, True), (umid, vmin, vmax, False)):
        samples, centre_index = [], None
        for k in range(_CHART_N):
            t = lo + (hi - lo) * k / (_CHART_N - 1.0)
            got = chart_of(t, fixed) if along_u else chart_of(fixed, t)
            if got is None:
                continue
            if centre_index is None and t >= 0.5 * (lo + hi):
                centre_index = len(samples)
            samples.append(got)
        if len(samples) < 2 or centre_index is None:
            continue
        unwrapped = np.unwrap(np.array([s[0] for s in samples]))
        unwrapped += 2.0 * math.pi * round((anchor[0] - unwrapped[centre_index])
                                           / (2.0 * math.pi))
        phis.extend(float(p) for p in unwrapped)
        axials.extend(s[1] for s in samples)
    return (min(phis), max(phis), min(axials), max(axials))
