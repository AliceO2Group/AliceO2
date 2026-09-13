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

"""The intermediate CSG description, and the two builders that realise it.

A recognised part is a JSON-serialisable tree of placed primitives, realised twice from the same
description: `build_occ()` gives the OCCT solid the symmetric difference measures, `build_root()`
the `TGeoShape` the oracle gate scores. Each leaf carries a frame `(origin, x, y, z)` in the part
frame, in cm, with `z` as the primitive's axis.

`build_root()` returns `(shape, placement)`: the shape in its own canonical frame and a 3x4
row-major `[R | t]` with `part = R * canonical + t`, or None for the identity. `build_occ()`
builds the solid in the part frame.
"""

import math

# Frames closer than this are the same; the identity fast path needs an exact rotation.
_IDENTITY_EPS = 1.0e-12

# Below this relative difference a cone's two radii are the same radius, and OCCT wants a
# cylinder rather than a cone. See `_occ_frustum`.
_CONE_DEGENERATE_EPS = 1.0e-12


def identity_frame(origin=(0.0, 0.0, 0.0)):
    return {"origin": [float(c) for c in origin],
            "x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}


def frame_from_axis(origin, axis_z, ref_x=None):
    """An orthonormal right-handed frame with `z` along `axis_z`, `x` along `ref_x` if given."""
    z = _unit(axis_z)
    if ref_x is not None:
        x = _sub(ref_x, _scale(z, _dot(ref_x, z)))
        if _norm(x) < 1.0e-9:
            x = None
        else:
            x = _unit(x)
    else:
        x = None
    if x is None:
        # any vector not parallel to z
        seed = (1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else (0.0, 1.0, 0.0)
        x = _unit(_sub(seed, _scale(z, _dot(seed, z))))
    y = _cross(z, x)
    return {"origin": [float(c) for c in origin], "x": list(x), "y": list(y), "z": list(z)}


def frame_is_identity_rotation(frame):
    return (abs(frame["x"][0] - 1.0) < _IDENTITY_EPS and abs(frame["x"][1]) < _IDENTITY_EPS
            and abs(frame["x"][2]) < _IDENTITY_EPS and abs(frame["y"][1] - 1.0) < _IDENTITY_EPS
            and abs(frame["y"][0]) < _IDENTITY_EPS and abs(frame["y"][2]) < _IDENTITY_EPS
            and abs(frame["z"][2] - 1.0) < _IDENTITY_EPS and abs(frame["z"][0]) < _IDENTITY_EPS
            and abs(frame["z"][1]) < _IDENTITY_EPS)


def frame_is_identity(frame):
    return frame_is_identity_rotation(frame) and all(abs(c) < _IDENTITY_EPS
                                                     for c in frame["origin"])


# ------------------------------------------------------------------------------------------
# tiny vector helpers
# ------------------------------------------------------------------------------------------

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    if n == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return (a[0] / n, a[1] / n, a[2] / n)


# ------------------------------------------------------------------------------------------
# the description
# ------------------------------------------------------------------------------------------

LEAF_TYPES = ("TGeoBBox", "TGeoTube", "TGeoTubeSeg", "TGeoCone", "TGeoSphere", "TGeoPcon",
              "TGeoTrd1", "TGeoTrd2", "TGeoArb8", "TGeoXtru", "TGeoPgon", "TGeoTorus", "TGeoEltu")

_REQUIRED_PARAMS = {
    "TGeoBBox": ("dx", "dy", "dz"),
    "TGeoTube": ("rmin", "rmax", "dz"),
    "TGeoTubeSeg": ("rmin", "rmax", "dz", "phi1", "phi2"),
    "TGeoCone": ("dz", "rmin1", "rmax1", "rmin2", "rmax2"),
    "TGeoSphere": ("rmin", "rmax"),
    "TGeoPcon": ("phi1", "dphi"),
    "TGeoTorus": ("r", "rmin", "rmax", "phi1", "dphi"),
    "TGeoEltu": ("a", "b", "dz"),
    "TGeoTrd1": ("dx1", "dx2", "dy", "dz"),
    "TGeoTrd2": ("dx1", "dx2", "dy1", "dy2", "dz"),
    "TGeoArb8": ("dz",),
    "TGeoXtru": (),
    "TGeoPgon": ("phi1", "dphi", "nedges"),
}

# Array parameters per leaf type, as lists; by default all of one leaf's arrays share a length.
_REQUIRED_ARRAY_PARAMS = {
    "TGeoPcon": ("z", "rmin", "rmax"),
    "TGeoPgon": ("z", "rmin", "rmax"),
    "TGeoArb8": ("vertices",),
    "TGeoXtru": ("x", "y", "z", "xoff", "yoff", "scale"),
}

_MIN_ARRAY_LENGTH = {
    "TGeoPcon": 2,
    "TGeoPgon": 2,
    "TGeoArb8": 16,
}

# A TGeoXtru's polygon and section counts are independent: groups of arrays, each with a minimum.
_ARRAY_LENGTH_GROUPS = {
    "TGeoXtru": ((("x", "y"), 3), (("z", "xoff", "yoff", "scale"), 2)),
}


class InvalidDescription(ValueError):
    """The numbers do not describe a legal solid of that class, so a recogniser declines.

    A missing parameter or an unknown leaf type is a caller bug and stays a plain `ValueError`.
    """


def _validate_eltu(p):
    for key in ("a", "b", "dz"):
        if p[key] <= 0.0:
            raise InvalidDescription(f"TGeoEltu: {key} = {p[key]} is not positive")


def _validate_torus(p):
    if p["r"] <= 0.0:
        raise InvalidDescription(f"TGeoTorus: the major radius {p['r']} is not positive")
    if p["rmax"] <= 0.0:
        raise InvalidDescription(f"TGeoTorus: rmax {p['rmax']} is not positive")
    if p["rmin"] < 0.0:
        raise InvalidDescription(f"TGeoTorus: rmin {p['rmin']} is negative")
    if p["rmin"] >= p["rmax"]:
        raise InvalidDescription(f"TGeoTorus: rmin {p['rmin']} is not below rmax {p['rmax']}")
    if p["rmax"] > p["r"]:
        # A tube radius above the major radius is a self-intersecting torus: refused.
        raise InvalidDescription(
            f"TGeoTorus: rmax {p['rmax']} exceeds the major radius {p['r']}, so this is a "
            "self-intersecting torus (a fillet blend) that TGeoTorus cannot state")
    if not 0.0 < p["dphi"] <= 360.0 + 1.0e-9:
        raise InvalidDescription(f"TGeoTorus: dphi {p['dphi']} is not in (0, 360]")


def _validate_pcon(p):
    if not 0.0 < p["dphi"] <= 360.0 + 1.0e-9:
        raise InvalidDescription(f"TGeoPcon: dphi {p['dphi']} is not in (0, 360]")
    z, rmin, rmax = p["z"], p["rmin"], p["rmax"]
    for i in range(len(z)):
        if rmin[i] < 0.0:
            raise InvalidDescription(f"TGeoPcon: rmin[{i}] = {rmin[i]} is negative")
        if rmin[i] > rmax[i]:
            raise InvalidDescription(
                f"TGeoPcon: rmin[{i}] = {rmin[i]} exceeds rmax[{i}] = {rmax[i]}")
    for i in range(1, len(z)):
        if z[i] < z[i - 1]:
            raise InvalidDescription(f"TGeoPcon: z is not non-decreasing at section {i} "
                             f"({z[i]} < {z[i - 1]})")
    if z[-1] <= z[0]:
        raise InvalidDescription("TGeoPcon: the profile has no axial extent")
    for i in range(2, len(z)):
        if z[i] == z[i - 1] == z[i - 2]:
            raise InvalidDescription(f"TGeoPcon: three sections share z = {z[i]}")


def _validate_pgon(p):
    _validate_pcon(p)
    if p["nedges"] < 1 or abs(p["nedges"] - round(p["nedges"])) > 1.0e-9:
        raise InvalidDescription(f"TGeoPgon: nedges {p['nedges']} is not a positive whole number")


def _validate_trd1(p):
    if p["dy"] <= 0.0 or p["dz"] <= 0.0:
        raise InvalidDescription(f"TGeoTrd1: dy {p['dy']} and dz {p['dz']} must both be positive")
    if min(p["dx1"], p["dx2"]) < 0.0 or max(p["dx1"], p["dx2"]) <= 0.0:
        raise InvalidDescription(f"TGeoTrd1: dx1 {p['dx1']}, dx2 {p['dx2']} do not bound a solid")


def _validate_trd2(p):
    if p["dz"] <= 0.0:
        raise InvalidDescription(f"TGeoTrd2: dz {p['dz']} must be positive")
    for a, b in (("dx1", "dx2"), ("dy1", "dy2")):
        if min(p[a], p[b]) < 0.0 or max(p[a], p[b]) <= 0.0:
            raise InvalidDescription(f"TGeoTrd2: {a} {p[a]}, {b} {p[b]} do not bound a solid")


def _validate_arb8(p):
    if p["dz"] <= 0.0:
        raise InvalidDescription(f"TGeoArb8: dz {p['dz']} must be positive")
    if len(p["vertices"]) != 16:
        raise InvalidDescription(f"TGeoArb8: needs 16 vertex coordinates, got {len(p['vertices'])}")
    for half, name in ((p["vertices"][:8], "-dz"), (p["vertices"][8:], "+dz")):
        corners = [(half[2 * i], half[2 * i + 1]) for i in range(4)]
        if len({(round(c[0], 12), round(c[1], 12)) for c in corners}) < 3:
            raise InvalidDescription(
                f"TGeoArb8: the {name} face has fewer than three distinct corners")


def _validate_xtru(p):
    z, scale = p["z"], p["scale"]
    for i in range(1, len(z)):
        if z[i] <= z[i - 1]:
            raise InvalidDescription(f"TGeoXtru: z is not strictly increasing at section {i} "
                             f"({z[i]} <= {z[i - 1]})")
    for i, s in enumerate(scale):
        if s <= 0.0:
            raise InvalidDescription(f"TGeoXtru: scale[{i}] = {s} is not positive")
    corners = {(round(a, 12), round(b, 12)) for a, b in zip(p["x"], p["y"])}
    if len(corners) != len(p["x"]):
        raise InvalidDescription("TGeoXtru: the polygon repeats a corner")


_LEAF_VALIDATORS = {
    "TGeoEltu": _validate_eltu,
    "TGeoTorus": _validate_torus,
    "TGeoPcon": _validate_pcon,
    "TGeoPgon": _validate_pgon,
    "TGeoTrd1": _validate_trd1,
    "TGeoTrd2": _validate_trd2,
    "TGeoArb8": _validate_arb8,
    "TGeoXtru": _validate_xtru,
}


def leaf(kind, params, frame, outside=False):
    """One placed primitive. `outside` marks a halfspace whose material is *outside* it.

    ROOT writes such a leaf as a `TGeoSubtraction` and OCCT as a `BRepAlgoAPI_Cut`.
    """
    if kind not in LEAF_TYPES:
        raise ValueError(f"unknown leaf type {kind!r}")
    arrays = _REQUIRED_ARRAY_PARAMS.get(kind, ())
    missing = [k for k in _REQUIRED_PARAMS[kind] + arrays if k not in params]
    if missing:
        raise ValueError(f"{kind}: missing parameter(s) {missing}")
    out = {k: float(params[k]) for k in _REQUIRED_PARAMS[kind]}
    for k in arrays:
        out[k] = [float(v) for v in params[k]]
    if arrays:
        groups = _ARRAY_LENGTH_GROUPS.get(kind,
                                          ((arrays, _MIN_ARRAY_LENGTH.get(kind, 1)),))
        for names, want in groups:
            lengths = {len(out[k]) for k in names}
            if len(lengths) != 1:
                raise InvalidDescription(
                    f"{kind}: array parameters {list(names)} have unequal lengths "
                    + ", ".join(f"{k}={len(out[k])}" for k in names))
            n = lengths.pop()
            if n < want:
                raise InvalidDescription(
                    f"{kind}: needs at least {want} of {list(names)}, got {n}")
    validator = _LEAF_VALIDATORS.get(kind)
    if validator is not None:
        validator(out)
    described = {"type": kind, "params": out, "frame": frame}
    if outside:
        # Only written when true, so every leaf recorded before halfspaces existed keeps its
        # bytes and the frozen digests of the self-test stay meaningful.
        described["outside"] = True
    return described


def placement_from_frame(frame):
    """The frame as a 3x4 row-major `[R | t]`, with `part = R * canonical + t`.

    `R`'s columns are the frame's basis vectors, as in `TGeoRotation::SetMatrix`.
    """
    x, y, z, o = frame["x"], frame["y"], frame["z"], frame["origin"]
    return [[x[0], y[0], z[0], o[0]],
            [x[1], y[1], z[1], o[1]],
            [x[2], y[2], z[2], o[2]]]


def placement_to_local(placement, point):
    """`R^T (p - t)`: a point in the part frame expressed in the shape's own frame."""
    if placement is None:
        return tuple(float(c) for c in point)
    d = (point[0] - placement[0][3], point[1] - placement[1][3], point[2] - placement[2][3])
    return tuple(sum(placement[r][c] * d[r] for r in range(3)) for c in range(3))


def placement_for_candidate(cand):
    """The rigid transform `build_root()` hands back beside the shape, or None for identity.

    It needs no ROOT, so `csg_<part>.json` and `--from-json` agree on the placement.
    """
    if cand["op"] != "primitive":
        # A genuine multi-leaf boolean is still a TGeoCompositeShape, whose TGeoBoolNode carries
        # the leaves' matrices itself; the composite is already in the part frame.
        return None
    lf = cand["leaves"][0]
    frame = lf["frame"]
    if frame_is_identity(frame):
        return None
    if lf_is_box(lf) and frame_is_identity_rotation(frame):
        # TGeoBBox carries a pure translation itself, through fOrigin. Leaving it there keeps
        # every artefact written for an axis-aligned box byte-identical to before this change.
        return None
    return placement_from_frame(frame)


def candidate(op, leaves, recogniser, notes=None):
    """A described solid: `primitive`, `union`, or `intersection` (of halfspaces).

    An intersection folds its leaves left to right and an `outside` leaf subtracts; the first leaf
    cannot be one, since an intersection of complements is unbounded.
    """
    if op not in ("primitive", "union", "intersection"):
        raise ValueError(f"unknown op {op!r}")
    if op == "primitive" and len(leaves) != 1:
        raise ValueError("op 'primitive' takes exactly one leaf")
    if op == "union" and len(leaves) < 2:
        raise ValueError("op 'union' takes at least two leaves")
    if op == "intersection":
        if len(leaves) < 2:
            raise ValueError("op 'intersection' takes at least two leaves")
        if leaves[0].get("outside"):
            raise ValueError("op 'intersection': the first leaf cannot be a complement")
    if op != "intersection" and any(lf.get("outside") for lf in leaves):
        raise ValueError(f"op {op!r} has no meaning for a complemented leaf")
    return {"op": op, "leaves": leaves, "recogniser": recogniser, "notes": notes or {}}


CELL_OPS = ("primitive", "intersection")


def cell(op, leaves):
    """One cell of a two-level DNF: a bare placed primitive, or an intersection of halfspaces.

    Validated as a candidate, then stripped to `{op, leaves}`.
    """
    if op not in CELL_OPS:
        raise ValueError(f"a cell is {' or '.join(CELL_OPS)}, not {op!r}")
    described = candidate(op, leaves, "cell")
    return {"op": described["op"], "leaves": described["leaves"]}


def union_of_cells(cells, recogniser, notes=None):
    """A union of intersection-cells: `{op: "unionOfCells", cells, recogniser, notes}`.

    It has no `leaves` key, so a one-level reader fails loudly; a cell may not itself be a union.
    """
    if len(cells) < 2:
        raise ValueError("op 'unionOfCells' takes at least two cells; one cell is that cell")
    for i, c in enumerate(cells):
        if not isinstance(c, dict) or set(c) != {"op", "leaves"}:
            raise ValueError(f"cell {i} is not a bare {{op, leaves}} description: "
                             f"{sorted(c) if isinstance(c, dict) else type(c).__name__}")
        if c["op"] not in CELL_OPS:
            raise ValueError(f"cell {i} has op {c['op']!r}: a DNF is two levels deep, so a cell "
                             f"is {' or '.join(CELL_OPS)} and never a union")
        cell(c["op"], c["leaves"])
    return {"op": "unionOfCells", "cells": cells, "recogniser": recogniser, "notes": notes or {}}


FLAT_CELL_KEYS = ("blocks", "volume", "lo", "hi")


def flat_cells(cells, recogniser, notes=None):
    """A union of halfspace cells for `O2FlatCSG`: `{op: "flatCells", cells, recogniser, notes}`.

    A cell is `{blocks, volume, lo, hi}`; `lo`/`hi` must be an outer bound of the cell, since
    `O2FlatCSG` builds its sub-cell boxes inside it. One cell is legal. `build_occ` folds the
    padded cells in `notes["occCells"]`.
    """
    if not cells:
        raise ValueError("op 'flatCells' takes at least one cell")
    for i, c in enumerate(cells):
        if not isinstance(c, dict) or set(c) != set(FLAT_CELL_KEYS):
            raise ValueError(f"cell {i} is not a bare {{{', '.join(FLAT_CELL_KEYS)}}} "
                             f"description: "
                             f"{sorted(c) if isinstance(c, dict) else type(c).__name__}")
        if not c["blocks"]:
            raise ValueError(f"cell {i} has no halfspace block: an empty intersection is "
                             "everything, not a cell")
        for key in ("lo", "hi"):
            if len(c[key]) != 3 or not all(math.isfinite(float(v)) for v in c[key]):
                raise ValueError(f"cell {i}'s {key} is not three finite numbers: {c[key]!r}")
        for axis in range(3):
            if float(c["lo"][axis]) > float(c["hi"][axis]):
                raise ValueError(f"cell {i}'s bounding box is inverted on axis {axis}: "
                                 f"{c['lo'][axis]} > {c['hi'][axis]}")
        if not (float(c["volume"]) > 0.0):
            raise ValueError(f"cell {i} has non-positive volume {c['volume']!r}")
    return {"op": "flatCells", "cells": cells, "recogniser": recogniser, "notes": notes or {}}


def flat_occ_cells(cand):
    """The padded cells `build_occ` folds for a `flatCells` description, kept under `notes`."""
    occ = (cand.get("notes") or {}).get("occCells")
    if not occ:
        raise ValueError("a flatCells description carries no notes['occCells']: there is nothing "
                         "to realise it with in OCCT")
    return occ


def describe(cand):
    """One line, for reports."""
    if cand["op"] == "flatCells":
        blocks = sum(len(c["blocks"]) for c in cand["cells"])
        return (f"O2FlatCSG({len(cand['cells'])} cell(s), {blocks} halfspace(s))")
    if cand["op"] == "unionOfCells":
        return " u ".join(f"({describe(c)})" if len(c["leaves"]) > 1 else describe(c)
                          for c in cand["cells"])
    parts = []
    for lf in cand["leaves"]:
        p = lf["params"]
        if lf["type"] in ("TGeoTube", "TGeoTubeSeg"):
            parts.append(f"{lf['type']}(rmin={p['rmin']:.4g}, rmax={p['rmax']:.4g}, "
                         f"dz={p['dz']:.4g})")
        elif lf["type"] == "TGeoBBox":
            parts.append(f"TGeoBBox({p['dx']:.4g}, {p['dy']:.4g}, {p['dz']:.4g})")
        elif lf["type"] == "TGeoCone":
            parts.append(f"TGeoCone(dz={p['dz']:.4g}, {p['rmin1']:.4g}/{p['rmax1']:.4g} -> "
                         f"{p['rmin2']:.4g}/{p['rmax2']:.4g})")
        elif lf["type"] == "TGeoTrd1":
            parts.append(f"TGeoTrd1(dx {p['dx1']:.4g} -> {p['dx2']:.4g}, dy={p['dy']:.4g}, "
                         f"dz={p['dz']:.4g})")
        elif lf["type"] == "TGeoTrd2":
            parts.append(f"TGeoTrd2(dx {p['dx1']:.4g} -> {p['dx2']:.4g}, "
                         f"dy {p['dy1']:.4g} -> {p['dy2']:.4g}, dz={p['dz']:.4g})")
        elif lf["type"] == "TGeoArb8":
            v = p["vertices"]
            parts.append(f"TGeoArb8(dz={p['dz']:.4g}, x {min(v[0::2]):.4g}..{max(v[0::2]):.4g}, "
                         f"y {min(v[1::2]):.4g}..{max(v[1::2]):.4g})")
        elif lf["type"] == "TGeoXtru":
            parts.append(f"TGeoXtru(nvert={len(p['x'])}, nz={len(p['z'])}, "
                         f"z {p['z'][0]:.4g}..{p['z'][-1]:.4g}, "
                         f"scale {min(p['scale']):.4g}..{max(p['scale']):.4g})")
        elif lf["type"] == "TGeoPgon":
            parts.append(f"TGeoPgon(nedges={int(round(p['nedges']))}, nz={len(p['z'])}, "
                         f"phi1={p['phi1']:.4g}, dphi={p['dphi']:.4g}, "
                         f"z {p['z'][0]:.4g}..{p['z'][-1]:.4g}, "
                         f"rmin {min(p['rmin']):.4g}..{max(p['rmin']):.4g}, "
                         f"rmax {min(p['rmax']):.4g}..{max(p['rmax']):.4g})")
        elif lf["type"] == "TGeoEltu":
            parts.append(f"TGeoEltu(a={p['a']:.4g}, b={p['b']:.4g}, dz={p['dz']:.4g})")
        elif lf["type"] == "TGeoTorus":
            parts.append(f"TGeoTorus(r={p['r']:.4g}, rmin={p['rmin']:.4g}, "
                         f"rmax={p['rmax']:.4g}, phi1={p['phi1']:.4g}, dphi={p['dphi']:.4g})")
        elif lf["type"] == "TGeoPcon":
            parts.append(f"TGeoPcon(nz={len(p['z'])}, phi1={p['phi1']:.4g}, "
                         f"dphi={p['dphi']:.4g}, z {p['z'][0]:.4g}..{p['z'][-1]:.4g}, "
                         f"rmin {min(p['rmin']):.4g}..{max(p['rmin']):.4g}, "
                         f"rmax {min(p['rmax']):.4g}..{max(p['rmax']):.4g})")
        else:
            parts.append(f"TGeoSphere(rmin={p['rmin']:.4g}, rmax={p['rmax']:.4g})")
    if cand["op"] == "union":
        return " u ".join(parts)
    if cand["op"] == "intersection":
        out = [parts[0]]
        for lf, text in zip(cand["leaves"][1:], parts[1:]):
            out.append((" - " if lf.get("outside") else " ^ ") + text)
        return "".join(out)
    return parts[0]


# ------------------------------------------------------------------------------------------
# builder 1: OCCT (the acceptance test's candidate side)
# ------------------------------------------------------------------------------------------

def build_occ(cand):
    """Realise the description as a `TopoDS_Shape` in OCCT. Requires pythonOCC."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    if cand["op"] == "flatCells":
        # The padded realisation, per `flat_cells`'s docstring: OCCT has no unbounded halfspace
        # either, so the acceptance test measures `_cell_leaf`'s bounded forms of the same cells.
        return _occ_balanced_union([build_occ(c) for c in flat_occ_cells(cand)])
    if cand["op"] == "unionOfCells":
        return _occ_balanced_union([build_occ(c) for c in cand["cells"]])
    leaves = cand["leaves"]
    out = _occ_leaf(leaves[0])
    for lf in leaves[1:]:
        nxt = _occ_leaf(lf)
        if cand["op"] == "union":
            maker, what = BRepAlgoAPI_Fuse, "BRepAlgoAPI_Fuse"
        elif lf.get("outside"):
            maker, what = BRepAlgoAPI_Cut, "BRepAlgoAPI_Cut"
        else:
            maker, what = BRepAlgoAPI_Common, "BRepAlgoAPI_Common"
        op = maker(out, nxt)
        op.Build()
        if not op.IsDone():
            raise RuntimeError(f"{what} failed while building the candidate")
        out = op.Shape()
    return out


def _occ_balanced_union(shapes):
    """Fuse the cells pairwise, level by level, so the OCCT tree has the ROOT tree's shape."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    level = list(shapes)
    while len(level) > 1:
        higher = []
        for i in range(0, len(level) - 1, 2):
            op = BRepAlgoAPI_Fuse(level[i], level[i + 1])
            op.Build()
            if not op.IsDone():
                raise RuntimeError("BRepAlgoAPI_Fuse failed while building the candidate")
            higher.append(op.Shape())
        if len(level) % 2:
            higher.append(level[-1])
        level = higher
    return level[0]


def _occ_ax2(frame, along_z=0.0):
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
    o = _add(tuple(frame["origin"]), _scale(tuple(frame["z"]), along_z))
    return gp_Ax2(gp_Pnt(*o), gp_Dir(*frame["z"]), gp_Dir(*frame["x"]))


def _occ_cut(outer, inner):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    op = BRepAlgoAPI_Cut(outer, inner)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("BRepAlgoAPI_Cut failed while building the candidate")
    return op.Shape()


def _dedupe_ring(pts, tol=1.0e-12):
    """Drop consecutive duplicates in a closed (r, z) ring, the wrap included.

    The rule of `O2_TGeoToCAD._dedupe_ring`; a duplicated corner would be a zero-length edge.
    """
    out = []
    for pt in pts:
        if out and abs(pt[0] - out[-1][0]) < tol and abs(pt[1] - out[-1][1]) < tol:
            continue
        out.append(pt)
    while len(out) > 1 and abs(out[0][0] - out[-1][0]) < tol and abs(out[0][1] - out[-1][1]) < tol:
        out.pop()
    return out


def pcon_profile_rz(params, tol=1.0e-12):
    """The closed (r, z) profile of a `TGeoPcon`, outer chain then inner chain reversed.

    Exactly the ring `O2_TGeoToCAD.conv_pcon` revolves.
    """
    z, rmin, rmax = params["z"], params["rmin"], params["rmax"]
    nz = len(z)
    outer = [(rmax[i], z[i]) for i in range(nz)]
    if all(r <= tol for r in rmin):
        inner = [(0.0, z[nz - 1]), (0.0, z[0])]
    else:
        inner = [(rmin[i], z[i]) for i in range(nz - 1, -1, -1)]
    return _dedupe_ring(outer + inner, tol)


def _occ_pcon(lf):
    """Revolve the (r, z) profile face: true cone/cylinder/plane faces, nothing tessellated."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
    from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt
    p, frame = lf["params"], lf["frame"]
    pts = pcon_profile_rz(p)
    if len(pts) < 3:
        raise ValueError("TGeoPcon: degenerate (r, z) profile "
                         f"({len(pts)} distinct corner(s))")
    # OCCT sweeps from the profile's own half-plane, so the profile is laid out at phi1 and the
    # revolution covers dphi -- the same convention `_occ_leaf` uses for a TGeoTubeSeg.
    phi1 = math.radians(p["phi1"])
    xr = _add(_scale(tuple(frame["x"]), math.cos(phi1)),
              _scale(tuple(frame["y"]), math.sin(phi1)))
    origin, zax = tuple(frame["origin"]), tuple(frame["z"])
    poly = BRepBuilderAPI_MakePolygon()
    for (r, zz) in pts:
        poly.Add(gp_Pnt(*_add(origin, _add(_scale(xr, r), _scale(zax, zz)))))
    poly.Close()
    if not poly.IsDone():
        raise RuntimeError("TGeoPcon: could not build the (r, z) profile wire")
    face = BRepBuilderAPI_MakeFace(poly.Wire())
    if not face.IsDone():
        raise RuntimeError("TGeoPcon: the (r, z) profile is not a valid planar face")
    rev = BRepPrimAPI_MakeRevol(face.Face(), gp_Ax1(gp_Pnt(*origin), gp_Dir(*zax)),
                                math.radians(p["dphi"]))
    rev.Build()
    if not rev.IsDone():
        raise RuntimeError("TGeoPcon: revolution of the (r, z) profile failed")
    return rev.Shape()


# ------------------------------------------------------------------------------------------
# the prism family: Trd1 / Trd2 / Arb8 / Xtru / Pgon
# ------------------------------------------------------------------------------------------
#
# One construction, a stack of corresponding closed sections; `prism_rings` states it once.

_PRISM_TYPES = ("TGeoTrd1", "TGeoTrd2", "TGeoArb8", "TGeoXtru", "TGeoPgon")


def _dedupe_ring3(pts, tol=1.0e-9):
    """Drop consecutive duplicate corners of a closed 3-D ring, wrap included, as the writer does."""
    out = []
    for q in pts:
        if out and max(abs(q[i] - out[-1][i]) for i in range(3)) < tol:
            continue
        out.append(tuple(float(c) for c in q))
    while len(out) > 1 and max(abs(out[0][i] - out[-1][i]) for i in range(3)) < tol:
        out.pop()
    return out


def _pgon_section_ring(r_apothem, z, phi1_deg, dphi_deg, nedges, full):
    """One `TGeoPgon` section polygon, as `O2_TGeoToCAD._pgon_ring` builds it.

    ROOT's rmin/rmax are apothem radii, so the corners sit at `r / cos(dseg / 2)`.
    """
    dseg = math.radians(dphi_deg) / nedges
    radius = r_apothem / math.cos(dseg / 2.0)
    n = nedges if full else nedges + 1
    return [(radius * math.cos(math.radians(phi1_deg) + k * dseg),
             radius * math.sin(math.radians(phi1_deg) + k * dseg), z) for k in range(n)]


def pgon_rings(params):
    """`(outer_stack, inner_stack|None)` for a `TGeoPgon`, as `conv_pgon` builds them."""
    z, rmin, rmax = params["z"], params["rmin"], params["rmax"]
    phi1, dphi, nedges = params["phi1"], params["dphi"], int(round(params["nedges"]))
    full = abs(dphi - 360.0) < 1.0e-9
    hollow = any(r > 0.0 for r in rmin)
    if hollow and full:
        # An annular section is two disjoint rings, which no single wire can express: the outer
        # and the inner prism are separate stacks and the caps are annular.
        return ([_pgon_section_ring(rmax[i], z[i], phi1, dphi, nedges, True)
                 for i in range(len(z))],
                [_pgon_section_ring(max(rmin[i], 0.0), z[i], phi1, dphi, nedges, True)
                 for i in range(len(z))])
    rings = []
    for i in range(len(z)):
        outer = _pgon_section_ring(rmax[i], z[i], phi1, dphi, nedges, full)
        if hollow:
            inner = _pgon_section_ring(max(rmin[i], 0.0), z[i], phi1, dphi, nedges, full)
            rings.append(outer + list(reversed(inner)))
        elif full:
            rings.append(outer)
        else:
            rings.append(outer + [(0.0, 0.0, z[i])])
    return rings, None


def prism_rings(lf):
    """`(outer_stack, inner_stack|None)`: the leaf's sections, in the leaf's own frame.

    Every ring is a closed polygon in corner order, and corner `i` of section `k` is joined to
    corner `i` of section `k + 1`. Ring lengths agree across the stack by construction.
    """
    kind, p = lf["type"], lf["params"]
    if kind == "TGeoTrd1":
        dx1, dx2, dy, dz = p["dx1"], p["dx2"], p["dy"], p["dz"]
        return ([[(-dx1, -dy, -dz), (dx1, -dy, -dz), (dx1, dy, -dz), (-dx1, dy, -dz)],
                 [(-dx2, -dy, dz), (dx2, -dy, dz), (dx2, dy, dz), (-dx2, dy, dz)]], None)
    if kind == "TGeoTrd2":
        dx1, dx2, dy1, dy2, dz = p["dx1"], p["dx2"], p["dy1"], p["dy2"], p["dz"]
        return ([[(-dx1, -dy1, -dz), (dx1, -dy1, -dz), (dx1, dy1, -dz), (-dx1, dy1, -dz)],
                 [(-dx2, -dy2, dz), (dx2, -dy2, dz), (dx2, dy2, dz), (-dx2, dy2, dz)]], None)
    if kind == "TGeoArb8":
        v, dz = p["vertices"], p["dz"]
        return ([[(v[2 * i], v[2 * i + 1], -dz) for i in range(4)],
                 [(v[8 + 2 * i], v[8 + 2 * i + 1], dz) for i in range(4)]], None)
    if kind == "TGeoXtru":
        x, y, z = p["x"], p["y"], p["z"]
        xoff, yoff, sc = p["xoff"], p["yoff"], p["scale"]
        return ([[(xoff[k] + sc[k] * x[i], yoff[k] + sc[k] * y[i], z[k])
                  for i in range(len(x))] for k in range(len(z))], None)
    if kind == "TGeoPgon":
        return pgon_rings(p)
    raise ValueError(f"{kind} is not a prism-family leaf")


def _to_part(frame, q):
    return _add(tuple(frame["origin"]),
                _add(_scale(tuple(frame["x"]), q[0]),
                     _add(_scale(tuple(frame["y"]), q[1]), _scale(tuple(frame["z"]), q[2]))))


def prism_samples(lf):
    """Every corner and every edge midpoint of a prism-family leaf, in the part frame.

    Edge midpoints are included because corners alone miss a wrong corner order.
    """
    outer, inner = prism_rings(lf)
    frame = lf["frame"]
    out = []
    for stack in (outer, inner):
        if stack is None:
            continue
        rings = [_dedupe_ring3(r) for r in stack]
        for k, ring in enumerate(rings):
            n = len(ring)
            for i, q in enumerate(ring):
                out.append(_to_part(frame, q))
                nxt = ring[(i + 1) % n]
                out.append(_to_part(frame, _scale(_add(q, nxt), 0.5)))
                if k + 1 < len(rings) and len(rings[k + 1]) == n:
                    up = rings[k + 1][i]
                    out.append(_to_part(frame, _scale(_add(q, up), 0.5)))
    return out


def _occ_quad_face(b0, b1, t1, t0, tol=1.0e-7):
    """One lateral patch: planar when its corners are coplanar, ruled when they are not.

    The rule of `O2_TGeoToCAD._quad_face`, including the Newell area test for a degenerate patch.
    """
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
    from OCC.Core.BRepFill import brepfill
    from OCC.Core.gp import gp_Pnt
    pts = _dedupe_ring3([b0, b1, t1, t0])
    if len(pts) < 3:
        return None
    nrm = [0.0, 0.0, 0.0]
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2])
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0])
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1])
    span = max(_norm(_sub(q, pts[0])) for q in pts[1:])
    if _norm(nrm) <= tol * span * span:
        return None
    if len(pts) == 3:
        return BRepBuilderAPI_MakeFace(_occ_polygon_wire(pts)).Face()
    n = _cross(_sub(b1, b0), _sub(t0, b0))
    nn = _norm(n)
    scale = max(_norm(_sub(b1, b0)), _norm(_sub(t0, b0)), 1.0e-30)
    off = abs(_dot(n, _sub(t1, b0))) / nn if nn > 0.0 else 0.0
    if nn > 1.0e-24 and off <= tol * scale:
        mf = BRepBuilderAPI_MakeFace(_occ_polygon_wire(pts))
        if mf.IsDone():
            return mf.Face()
    e1 = BRepBuilderAPI_MakeEdge(gp_Pnt(*b0), gp_Pnt(*b1)).Edge()
    e2 = BRepBuilderAPI_MakeEdge(gp_Pnt(*t0), gp_Pnt(*t1)).Edge()
    return brepfill.Face(e1, e2)


def _occ_polygon_wire(pts):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCC.Core.gp import gp_Pnt
    poly = BRepBuilderAPI_MakePolygon()
    for q in pts:
        poly.Add(gp_Pnt(float(q[0]), float(q[1]), float(q[2])))
    poly.Close()
    if not poly.IsDone():
        raise RuntimeError("prism: could not build a section wire")
    return poly.Wire()


def _occ_prism(lf):
    """Sew a prism-family leaf out of explicit faces -- no tessellation, no approximation."""
    from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid,
                                         BRepBuilderAPI_Sewing)
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopoDS import topods
    kind = lf["type"]
    outer, inner = prism_rings(lf)
    frame = lf["frame"]
    stacks = []
    for stack in (outer, inner):
        if stack is None:
            continue
        rings = [_dedupe_ring3([_to_part(frame, q) for q in ring]) for ring in stack]
        nv = len(rings[0])
        if nv < 3 or any(len(r) != nv for r in rings):
            raise ValueError(f"{kind}: sections carry "
                             f"{sorted({len(r) for r in rings})} distinct corner counts")
        stacks.append(rings)
    faces = []
    for rings in stacks:
        nv = len(rings[0])
        for k in range(len(rings) - 1):
            lo, hi = rings[k], rings[k + 1]
            for i in range(nv):
                j = (i + 1) % nv
                face = _occ_quad_face(lo[i], lo[j], hi[j], hi[i])
                if face is not None:
                    faces.append(face)
    for idx in (0, -1):
        mf = BRepBuilderAPI_MakeFace(_occ_polygon_wire(stacks[0][idx]))
        if len(stacks) == 2:
            mf.Add(topods.Wire(_occ_polygon_wire(stacks[1][idx]).Reversed()))
        if not mf.IsDone():
            raise ValueError(f"{kind}: could not build a cap face")
        faces.append(mf.Face())
    extent = max(abs(c) for rings in stacks for r in rings for q in r for c in q) or 1.0
    sew = BRepBuilderAPI_Sewing(1.0e-7 * extent)
    for face in faces:
        sew.Add(face)
    sew.Perform()
    shell = sew.SewedShape()
    if shell is None or shell.IsNull():
        raise ValueError(f"{kind}: sewing the sections produced nothing")
    ms = BRepBuilderAPI_MakeSolid(topods.Shell(shell))
    ms.Build()
    solid = ms.Solid()
    props = GProp_GProps()
    brepgprop.VolumeProperties(solid, props)
    if props.Mass() < 0.0:
        solid = topods.Solid(solid.Reversed())
    return solid


def _occ_eltu(lf):
    """An elliptic cylinder, built exactly as `O2_TGeoToCAD.conv_eltu` builds it.

    `gp_Elips` wants its major radius first, so the frame's x is not assumed to be the major axis.
    """
    from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                         BRepBuilderAPI_MakeWire)
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Elips, gp_Pnt, gp_Vec
    p, frame = lf["params"], lf["frame"]
    base = _sub(tuple(frame["origin"]), _scale(tuple(frame["z"]), p["dz"]))
    if p["a"] >= p["b"]:
        major_dir, major, minor = frame["x"], p["a"], p["b"]
    else:
        major_dir, major, minor = frame["y"], p["b"], p["a"]
    axis = gp_Ax2(gp_Pnt(*base), gp_Dir(*frame["z"]), gp_Dir(*major_dir))
    edge = BRepBuilderAPI_MakeEdge(gp_Elips(axis, major, minor)).Edge()
    face = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(edge).Wire())
    if not face.IsDone():
        raise RuntimeError("TGeoEltu: the ellipse wire is not a valid planar face")
    prism = BRepPrimAPI_MakePrism(face.Face(),
                                  gp_Vec(*_scale(tuple(frame["z"]), 2.0 * p["dz"])))
    prism.Build()
    if not prism.IsDone():
        raise RuntimeError("TGeoEltu: the prism failed")
    return prism.Shape()


def _occ_torus(lf):
    """The torus, built exactly as `O2_TGeoToCAD.conv_torus` builds it.

    A hollow torus's inner cut is swept a hair further in phi, so no wedge face is coincident.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeTorus
    p, frame = lf["params"], lf["frame"]
    phi1, dphi = math.radians(p["phi1"]), math.radians(p["dphi"])
    xr = _add(_scale(tuple(frame["x"]), math.cos(phi1)),
              _scale(tuple(frame["y"]), math.sin(phi1)))
    rotated = {"origin": frame["origin"], "x": list(xr), "y": frame["y"], "z": frame["z"]}

    def make(minor, sweep):
        maker = BRepPrimAPI_MakeTorus(_occ_ax2(rotated), p["r"], minor, sweep)
        maker.Build()
        if not maker.IsDone():
            raise RuntimeError("BRepPrimAPI_MakeTorus failed while building the candidate")
        return maker.Shape()

    outer = make(p["rmax"], dphi)
    if p["rmin"] > 0.0:
        full = dphi >= 2.0 * math.pi - 1.0e-12
        inner = make(p["rmin"], dphi if full else min(dphi + 1.0e-4, 2.0 * math.pi))
        outer = _occ_cut(outer, inner)
    return outer


def _occ_leaf(lf):
    from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder,
                                      BRepPrimAPI_MakeSphere)
    from OCC.Core.gp import gp_Pnt
    kind, p, frame = lf["type"], lf["params"], lf["frame"]
    if kind == "TGeoTorus":
        return _occ_torus(lf)
    if kind == "TGeoEltu":
        return _occ_eltu(lf)
    if kind == "TGeoPcon":
        return _occ_pcon(lf)
    if kind in _PRISM_TYPES:
        return _occ_prism(lf)
    if kind == "TGeoBBox":
        corner = tuple(frame["origin"])
        for axis, half in (("x", p["dx"]), ("y", p["dy"]), ("z", p["dz"])):
            corner = _sub(corner, _scale(tuple(frame[axis]), half))
        ax2 = _occ_ax2({"origin": list(corner), "x": frame["x"], "y": frame["y"],
                        "z": frame["z"]})
        return BRepPrimAPI_MakeBox(ax2, 2 * p["dx"], 2 * p["dy"], 2 * p["dz"]).Shape()
    if kind in ("TGeoTube", "TGeoTubeSeg"):
        ax2 = _occ_ax2(frame, -p["dz"])
        if kind == "TGeoTubeSeg":
            # OCCT sweeps from the frame's own x direction, so rotate the reference direction to
            # phi1 and sweep by (phi2 - phi1); ROOT states the same wedge as two absolute angles.
            phi1 = math.radians(p["phi1"])
            xr = _add(_scale(tuple(frame["x"]), math.cos(phi1)),
                      _scale(tuple(frame["y"]), math.sin(phi1)))
            rotated = {"origin": frame["origin"], "x": list(xr), "y": frame["y"],
                       "z": frame["z"]}
            ax2 = _occ_ax2(rotated, -p["dz"])
            sweep = math.radians(p["phi2"] - p["phi1"])
            outer = BRepPrimAPI_MakeCylinder(ax2, p["rmax"], 2 * p["dz"], sweep).Shape()
            if p["rmin"] > 0.0:
                inner = BRepPrimAPI_MakeCylinder(_occ_ax2(rotated, -p["dz"] - _pad(p["dz"])),
                                                 p["rmin"], 2 * p["dz"] + 4 * _pad(p["dz"]),
                                                 sweep).Shape()
                outer = _occ_cut(outer, inner)
            return outer
        outer = BRepPrimAPI_MakeCylinder(ax2, p["rmax"], 2 * p["dz"]).Shape()
        if p["rmin"] > 0.0:
            # The inner cylinder is longer than the outer, so the cut has no coincident caps.
            pad = _pad(p["dz"])
            inner = BRepPrimAPI_MakeCylinder(_occ_ax2(frame, -p["dz"] - pad), p["rmin"],
                                             2 * p["dz"] + 2 * pad).Shape()
            outer = _occ_cut(outer, inner)
        return outer
    if kind == "TGeoCone":
        outer = _occ_frustum(_occ_ax2(frame, -p["dz"]), p["rmax1"], p["rmax2"], 2 * p["dz"])
        if p["rmin1"] > 0.0 or p["rmin2"] > 0.0:
            pad = _pad(p["dz"])
            slope = (p["rmin2"] - p["rmin1"]) / (2 * p["dz"])
            inner = _occ_frustum(_occ_ax2(frame, -p["dz"] - pad),
                                 max(p["rmin1"] - slope * pad, 0.0),
                                 max(p["rmin2"] + slope * pad, 0.0),
                                 2 * p["dz"] + 2 * pad)
            outer = _occ_cut(outer, inner)
        return outer
    if kind == "TGeoSphere":
        o = tuple(frame["origin"])
        outer = BRepPrimAPI_MakeSphere(gp_Pnt(*o), p["rmax"]).Shape()
        if p["rmin"] > 0.0:
            inner = BRepPrimAPI_MakeSphere(gp_Pnt(*o), p["rmin"]).Shape()
            outer = _occ_cut(outer, inner)
        return outer
    raise ValueError(f"unhandled leaf type {kind!r}")


def _occ_frustum(ax2, r1, r2, height):
    """A cone frustum, or a cylinder when its two radii are the same.

    `BRepPrimAPI_MakeCone` raises on two identical radii, which a `TGeoCone` barrel or bore can have.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
    if abs(r1 - r2) <= _CONE_DEGENERATE_EPS * max(abs(r1), abs(r2), 1.0):
        return BRepPrimAPI_MakeCylinder(ax2, 0.5 * (r1 + r2), height).Shape()
    return BRepPrimAPI_MakeCone(ax2, r1, r2, height).Shape()


def _pad(dz):
    return max(1.0e-3 * dz, 1.0e-6)


# ------------------------------------------------------------------------------------------
# builder 2: ROOT (what shape_<part>.root carries)
# ------------------------------------------------------------------------------------------

def build_root(cand, name="shape"):
    """Realise the description as `(TGeoShape, placement)`. Requires PyROOT.

    A single primitive is the bare ROOT class in its own canonical frame and `placement` places it;
    a multi-leaf union is a `TGeoCompositeShape` in the part frame with placement None.
    """
    import ROOT
    placement = placement_for_candidate(cand)
    if cand["op"] == "flatCells":
        return _root_flat_csg(cand, name), None
    if cand["op"] == "unionOfCells":
        return _root_balanced_union(name, cand["cells"]), None
    if cand["op"] == "primitive":
        lf = cand["leaves"][0]
        frame = lf["frame"]
        if placement is None and lf_is_box(lf) and not frame_is_identity(frame):
            # Axis-aligned box: TGeoBBox's own fOrigin is the placement.
            from array import array
            p = lf["params"]
            return ROOT.TGeoBBox(name, p["dx"], p["dy"], p["dz"],
                                 array("d", [float(c) for c in frame["origin"]])), None
        shape = _root_leaf(lf, name)
        return shape, placement
    shapes = [(_root_leaf(lf, f"{name}_l{i}"), lf["frame"])
              for i, lf in enumerate(cand["leaves"])]
    outside = [bool(lf.get("outside")) for lf in cand["leaves"]]
    return _root_composite(name, shapes, cand["op"], outside), placement


def _root_cell(c, name):
    """`(shape, frame)` for one cell of a DNF.

    An intersection cell is in the part frame with an identity frame; a primitive cell is the bare
    shape with the frame that places it in the union node.
    """
    if c["op"] == "primitive":
        lf = c["leaves"][0]
        return _root_leaf(lf, name), lf["frame"]
    shapes = [(_root_leaf(lf, f"{name}_l{i}"), lf["frame"]) for i, lf in enumerate(c["leaves"])]
    outside = [bool(lf.get("outside")) for lf in c["leaves"]]
    return _root_composite(name, shapes, "intersection", outside), identity_frame()


_FLAT_CSG_DECLARED = []


def _declare_flat_csg():
    """Make `o2::cad::O2FlatCSG` and `LoadFlatCSG` visible to Cling. Once per interpreter."""
    import ROOT
    if _FLAT_CSG_DECLARED:
        return
    ROOT.gInterpreter.AddIncludePath(f"{ROOT.gSystem.Getenv('O2_ROOT')}/include")
    ROOT.gSystem.Load("libO2CADSupport")
    ROOT.gInterpreter.Declare(
        '#include "CADSupport/O2FlatCSG.h"\n'
        'namespace o2 { namespace cad {\n'
        'bool LoadFlatCSG(const std::string& file, O2FlatCSG& solid);\n'
        '} }')
    _FLAT_CSG_DECLARED.append(True)


def _root_flat_csg(cand, name):
    """The `O2FlatCSG` a `flatCells` description describes, built through its sidecar.

    Writing and loading `flatcsg_*.bin` assembles the shape with the code `geom.C` runs.
    """
    import tempfile
    from pathlib import Path
    import ROOT
    from cadsupport import flat
    _declare_flat_csg()
    blocks, cells = flat_sidecar_records(cand)
    with tempfile.TemporaryDirectory() as folder:
        sidecar = Path(folder) / "flatcsg.bin"
        flat.write_sidecar(sidecar, blocks, cells)
        shape = ROOT.o2.cad.O2FlatCSG(name)
        ROOT.SetOwnership(shape, False)
        if not ROOT.o2.cad.LoadFlatCSG(str(sidecar), shape):
            raise ValueError(f"LoadFlatCSG refused the sidecar written for {name!r}")
    shape.CloseShape()
    if not shape.IsClosed():
        raise ValueError(f"O2FlatCSG::CloseShape refused the cells of {name!r}: see its Error "
                         "message above (a missing, inverted or non-finite cell bounding box)")
    return shape


def flat_sidecar_records(cand):
    """`(blocks, cells)` in the layout `cadsupport.flat.write_sidecar` takes.

    The blocks of every cell, concatenated, and the `(first, count, volume, lo, hi)` cell table.
    """
    blocks, cells = [], []
    for c in cand["cells"]:
        cells.append({"first": len(blocks), "count": len(c["blocks"]),
                      "volume": float(c["volume"]),
                      "lo": [float(v) for v in c["lo"]], "hi": [float(v) for v in c["hi"]]})
        blocks.extend(c["blocks"])
    return blocks, cells


def _root_balanced_union(name, cells):
    """The cells as a balanced binary tree of `TGeoUnion` nodes, so queries scale with log2 N."""
    import ROOT
    level = [_root_cell(c, f"{name}_c{i}") for i, c in enumerate(cells)]
    step = 0
    while len(level) > 1:
        higher = []
        for i in range(0, len(level) - 1, 2):
            (left, left_frame), (right, right_frame) = level[i], level[i + 1]
            ROOT.SetOwnership(left, False)
            ROOT.SetOwnership(right, False)
            node = ROOT.TGeoUnion(left, right, _root_matrix(left_frame, f"{name}_u{step}a"),
                                  _root_matrix(right_frame, f"{name}_u{step}b"))
            ROOT.SetOwnership(node, False)
            comp = ROOT.TGeoCompositeShape(f"{name}_u{step}", node)
            ROOT.SetOwnership(comp, False)
            higher.append((comp, identity_frame()))
            step += 1
        if len(level) % 2:
            higher.append(level[-1])
        level = higher
    shape = level[0][0]
    shape.SetName(name)
    return shape


def root_placement_matrix(placement, name="placement"):
    """The placement as the `TGeoHMatrix` stored under `placement`, or None for the identity."""
    if placement is None:
        return None
    import ROOT
    # Through TGeoRotation/TGeoCombiTrans, which set the kGeoRotation/kGeoTranslation bits.
    combi = _root_matrix({"x": [placement[0][0], placement[1][0], placement[2][0]],
                          "y": [placement[0][1], placement[1][1], placement[2][1]],
                          "z": [placement[0][2], placement[1][2], placement[2][2]],
                          "origin": [placement[0][3], placement[1][3], placement[2][3]]}, name)
    matrix = ROOT.TGeoHMatrix(combi)
    matrix.SetName(name)
    ROOT.SetOwnership(matrix, False)
    return matrix


def placement_from_root_matrix(matrix):
    """The inverse of `root_placement_matrix()`, for reading an artefact back."""
    if matrix is None:
        return None
    rot = matrix.GetRotationMatrix()
    tr = matrix.GetTranslation()
    return [[rot[0], rot[1], rot[2], tr[0]],
            [rot[3], rot[4], rot[5], tr[1]],
            [rot[6], rot[7], rot[8], tr[2]]]


def lf_is_box(lf):
    return lf["type"] == "TGeoBBox"


def _root_matrix(frame, name):
    import ROOT
    from array import array
    rot = ROOT.TGeoRotation(name + "_r")
    # TGeoRotation::SetMatrix takes the local->master matrix row-major, i.e. the columns are the
    # local frame's basis vectors expressed in the part frame.
    m = array("d", [frame["x"][0], frame["y"][0], frame["z"][0],
                    frame["x"][1], frame["y"][1], frame["z"][1],
                    frame["x"][2], frame["y"][2], frame["z"][2]])
    rot.SetMatrix(m)
    combi = ROOT.TGeoCombiTrans(frame["origin"][0], frame["origin"][1], frame["origin"][2], rot)
    ROOT.SetOwnership(rot, False)
    ROOT.SetOwnership(combi, False)
    return combi


def _root_node_class(op, outside):
    import ROOT
    if op == "union":
        return ROOT.TGeoUnion
    if op == "intersection":
        return ROOT.TGeoSubtraction if outside else ROOT.TGeoIntersection
    raise ValueError(f"unhandled composite op {op!r}")


def _root_composite(name, shapes_and_frames, op, outside=None):
    """Left-fold the leaves into nested boolean nodes, in the leaves' order.

    PyROOT owns no operand, since `TGeoBoolNode` deletes them. Under `intersection` an `outside`
    leaf enters as a `TGeoSubtraction`.
    """
    import ROOT
    flags = list(outside or [False] * len(shapes_and_frames))
    (s0, f0), (s1, f1) = shapes_and_frames[0], shapes_and_frames[1]
    ROOT.SetOwnership(s0, False)
    ROOT.SetOwnership(s1, False)
    node = _root_node_class(op, flags[1])(s0, s1, _root_matrix(f0, f"{name}_m0"),
                                          _root_matrix(f1, f"{name}_m1"))
    ROOT.SetOwnership(node, False)
    comp = ROOT.TGeoCompositeShape(f"{name}_c1", node)
    ROOT.SetOwnership(comp, False)
    for i, (shape, frame) in enumerate(shapes_and_frames[2:], start=2):
        ROOT.SetOwnership(shape, False)
        node = _root_node_class(op, flags[i])(comp, shape, ROOT.nullptr,
                                              _root_matrix(frame, f"{name}_m{i}"))
        ROOT.SetOwnership(node, False)
        comp = ROOT.TGeoCompositeShape(f"{name}_c{i}", node)
        ROOT.SetOwnership(comp, False)
    comp.SetName(name)
    return comp


def _root_leaf(lf, name):
    import ROOT
    kind, p = lf["type"], lf["params"]
    if kind == "TGeoBBox":
        return ROOT.TGeoBBox(name, p["dx"], p["dy"], p["dz"])
    if kind == "TGeoTube":
        return ROOT.TGeoTube(name, p["rmin"], p["rmax"], p["dz"])
    if kind == "TGeoTubeSeg":
        return ROOT.TGeoTubeSeg(name, p["rmin"], p["rmax"], p["dz"], p["phi1"], p["phi2"])
    if kind == "TGeoCone":
        return ROOT.TGeoCone(name, p["dz"], p["rmin1"], p["rmax1"], p["rmin2"], p["rmax2"])
    if kind == "TGeoSphere":
        return ROOT.TGeoSphere(name, p["rmin"], p["rmax"])
    if kind == "TGeoTorus":
        return ROOT.TGeoTorus(name, p["r"], p["rmin"], p["rmax"], p["phi1"], p["dphi"])
    if kind == "TGeoEltu":
        return ROOT.TGeoEltu(name, p["a"], p["b"], p["dz"])
    if kind == "TGeoPcon":
        shape = ROOT.TGeoPcon(name, p["phi1"], p["dphi"], len(p["z"]))
        for i, (zz, r0, r1) in enumerate(zip(p["z"], p["rmin"], p["rmax"])):
            shape.DefineSection(i, zz, r0, r1)
        return shape
    if kind == "TGeoPgon":
        shape = ROOT.TGeoPgon(name, p["phi1"], p["dphi"], int(round(p["nedges"])), len(p["z"]))
        for i, (zz, r0, r1) in enumerate(zip(p["z"], p["rmin"], p["rmax"])):
            shape.DefineSection(i, zz, r0, r1)
        return shape
    if kind == "TGeoTrd1":
        return ROOT.TGeoTrd1(name, p["dx1"], p["dx2"], p["dy"], p["dz"])
    if kind == "TGeoTrd2":
        return ROOT.TGeoTrd2(name, p["dx1"], p["dx2"], p["dy1"], p["dy2"], p["dz"])
    if kind == "TGeoArb8":
        from array import array
        return ROOT.TGeoArb8(name, p["dz"], array("d", [float(v) for v in p["vertices"]]))
    if kind == "TGeoXtru":
        from array import array
        shape = ROOT.TGeoXtru(len(p["z"]))
        shape.SetName(name)
        shape.DefinePolygon(len(p["x"]), array("d", [float(v) for v in p["x"]]),
                            array("d", [float(v) for v in p["y"]]))
        for k in range(len(p["z"])):
            shape.DefineSection(k, p["z"][k], p["xoff"][k], p["yoff"][k], p["scale"][k])
        return shape
    raise ValueError(f"unhandled leaf type {kind!r}")
