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

"""The flat-DNF emitter: a cell's carriers as signed implicit halfspaces for `O2FlatCSG`.

The material side is `sign * Q(x) <= 0`; the sign composes the carrier's own orientation and its
`side`, and an inverted one still gives a solid, so the emitter self-test checks it. A plane is
stored with `2b = n` for a unit outward normal `n`, which keeps `O2FlatCSG`'s accelerated queries
bit-identical to their `_Loop` twins.
"""

import math
import struct

SIDECAR_MAGIC = b"O2FLTCSG"
SIDECAR_VERSION = 1

# The two `FlatCSGHalfspace::Kind` values, as the sidecar spells them.
KIND_QUADRIC = 0
KIND_TORUS = 1

# One quadric block is ten doubles; the record on file carries eleven, the last unused.
QUADRIC_COEFFICIENTS = 10
BLOCK_COEFFICIENTS = 11


def _outer(u, v):
    return [[u[i] * v[j] for j in range(3)] for i in range(3)]


def _quadric(a, b, c):
    """Pack A (3x3 symmetric), b (3) and c into the ten-double block the shape stores."""
    return [a[0][0], a[0][1], a[0][2], a[1][1], a[1][2], a[2][2], b[0], b[1], b[2], c]


def quadric_from_carrier(carrier):
    """`(sign, block)` for a plane, sphere, cylinder or cone carrier; `exterior` flips the sign."""
    from cadsupport import recognise
    sign = -1.0 if carrier["side"] == "exterior" else 1.0
    kind = carrier["kind"]

    if kind == "plane":
        n = carrier["n"]
        p = carrier["p"]
        # Q(x) = n.(x - p); the material side of an outward normal is Q <= 0. `2b = n` for a unit
        # normal is the convention of design section 3.1 and is not free to vary.
        return sign, _quadric([[0.0] * 3 for _ in range(3)],
                              [0.5 * n[0], 0.5 * n[1], 0.5 * n[2]],
                              -(n[0] * p[0] + n[1] * p[1] + n[2] * p[2]))

    if kind == "sphere":
        p = carrier["p"]
        r = carrier["r"]
        identity = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        return sign, _quadric(identity, [-p[0], -p[1], -p[2]],
                              p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - r * r)

    if kind in ("cylinder", "cone"):
        d = carrier["d"]
        p = carrier["p"]
        r = carrier["r"]
        k = 0.0 if kind == "cylinder" else math.tan(carrier["a"])
        scale = 1.0 + k * k
        dd = _outer(d, d)
        a = [[(1.0 if i == j else 0.0) - scale * dd[i][j] for j in range(3)] for i in range(3)]
        ap = [sum(a[i][j] * p[j] for j in range(3)) for i in range(3)]
        pd = sum(p[i] * d[i] for i in range(3))
        b = [-ap[i] - r * k * d[i] for i in range(3)]
        c = sum(p[i] * ap[i] for i in range(3)) + 2.0 * r * k * pd - r * r
        return sign, _quadric(a, b, c)

    raise recognise.Declined(f"a {kind} carrier has no quadric form")


def torus_from_carrier(carrier):
    """`(sign, centre, axis, major, minor)` for a torus carrier.

    The axis is normalised here, as `O2FlatCSG::AddTorus` does, so both evaluate the same torus.
    """
    sign = -1.0 if carrier["side"] == "exterior" else 1.0
    axis = list(carrier["d"])
    length = math.sqrt(sum(v * v for v in axis))
    if length <= 0.0 or not math.isfinite(length):
        raise ValueError(f"a torus carrier's axis {tuple(axis)} has no direction")
    return (sign, list(carrier["p"]), [v / length for v in axis], carrier["r"], carrier["rt"])


# Below this the tangent of a cone's semi-angle is a cylinder's, and the carrier has no apex.
# `recognise._cell_leaf` uses the same floor and declines there rather than build a leaf.
_APEX_SLOPE_FLOOR = 1.0e-30


def cone_apex(carrier):
    """The apex of a cone carrier, or None when its semi-angle is too small for one to exist."""
    k = math.tan(carrier["a"])
    if abs(k) < _APEX_SLOPE_FLOOR:
        return None, k
    p, d = carrier["p"], carrier["d"]
    return tuple(p[i] - (carrier["r"] / k) * d[i] for i in range(3)), k


def cone_apex_plane(carrier):
    """The plane block an INTERIOR cone carrier needs beside its quadric, or None.

    `sign*Q <= 0` is the double cone; `{rho <= r + k u} == {Q <= 0} n {r + k u >= 0}`, and the
    second set is the plane through the apex with unit outward normal `-sign(k) d`. An exterior
    cone gets nothing: `check_cell_box` refuses it.
    """
    if carrier["kind"] != "cone" or carrier["side"] == "exterior":
        return None
    apex, k = cone_apex(carrier)
    if apex is None:
        return None
    d = carrier["d"]
    n = tuple(-math.copysign(1.0, k) * d[i] for i in range(3))
    return {"kind": "quadric", "sign": 1.0,
            "c": _quadric([[0.0] * 3 for _ in range(3)],
                          [0.5 * n[0], 0.5 * n[1], 0.5 * n[2]],
                          -(n[0] * apex[0] + n[1] * apex[1] + n[2] * apex[2])) + [0.0]}


def check_cell_box(carriers, lo, hi):
    """`Declined` when an EXTERIOR cone's quadric carves a mirror cone inside this cell box.

    The caller must pass the same `lo`/`hi` it hands to `O2FlatCSG::SetCellBBox`, per cell.
    """
    from cadsupport import recognise
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    for carrier in carriers:
        if carrier["kind"] != "cone" or carrier["side"] != "exterior":
            continue
        apex, k = cone_apex(carrier)
        if apex is None:
            continue
        d = carrier["d"]
        reach = min(k * sum((corner[i] - apex[i]) * d[i] for i in range(3)) for corner in corners)
        if reach < 0.0:
            raise recognise.Declined(
                "an exterior cone carrier whose cell box reaches past its apex: the quadric's "
                "mirror nappe would remove material that is really there")


def blocks_from_carriers(carriers):
    """The halfspace blocks of one cell, in carrier order, plus an interior cone's apex plane.

    So not one block per carrier: size a cell's `count` with `len(...)` of the result.
    """
    blocks = []
    for carrier in carriers:
        if carrier["kind"] == "torus":
            sign, centre, axis, major, minor = torus_from_carrier(carrier)
            blocks.append({"kind": "torus", "sign": sign,
                           "c": centre + axis + [major, minor, 0.0, 0.0, 0.0]})
        else:
            sign, block = quadric_from_carrier(carrier)
            blocks.append({"kind": "quadric", "sign": sign, "c": block + [0.0]})
        apex_plane = cone_apex_plane(carrier)
        if apex_plane is not None:
            blocks.append(apex_plane)
    return blocks


def eval_block(block, point):
    """`sign * f(point)`, the arithmetic of `O2FlatCSG::EvalHalfspace`; `<= 0` means inside."""
    c = block["c"]
    x, y, z = point
    if block["kind"] == "torus":
        offset = (x - c[0], y - c[1], z - c[2])
        along = offset[0] * c[3] + offset[1] * c[4] + offset[2] * c[5]
        radial = tuple(offset[i] - along * c[3 + i] for i in range(3))
        rho = math.sqrt(sum(v * v for v in radial))
        return block["sign"] * (math.hypot(rho - c[6], along) - c[7])
    quadratic = (c[0] * x * x + c[3] * y * y + c[5] * z * z +
                 2.0 * (c[1] * x * y + c[2] * x * z + c[4] * y * z))
    return block["sign"] * (quadratic + 2.0 * (c[6] * x + c[7] * y + c[8] * z) + c[9])


def flat_contains(blocks, point):
    """True when every block contains the point: one cell's membership test."""
    return all(eval_block(block, point) <= 0.0 for block in blocks)


def plane_scaling_error(block):
    """`| |2b| - 1 |` for a plane block (the `2b = n` convention), or None for another block."""
    if block["kind"] != "quadric":
        return None
    c = block["c"]
    if any(c[index] != 0.0 for index in range(6)):
        return None
    two_b = math.sqrt(4.0 * (c[6] * c[6] + c[7] * c[7] + c[8] * c[8]))
    return abs(two_b - 1.0)


def write_sidecar(path, blocks, cells):
    """Write the version-1 flat-CSG sidecar, byte-compatible with `WriteFlatCSG`.

    Field by field, never as a struct: the record packs at 100 bytes, not the 104-byte C++ layout.
    """
    with open(path, "wb") as handle:
        handle.write(SIDECAR_MAGIC)
        handle.write(struct.pack("<III", SIDECAR_VERSION, len(blocks), len(cells)))
        for block in blocks:
            handle.write(struct.pack("<i", KIND_TORUS if block["kind"] == "torus"
                                     else KIND_QUADRIC))
            handle.write(struct.pack("<d", block["sign"]))
            coefficients = list(block["c"])
            if len(coefficients) > BLOCK_COEFFICIENTS:
                raise ValueError(f"a halfspace block carries {len(coefficients)} coefficients, "
                                 f"more than the {BLOCK_COEFFICIENTS} the sidecar has room for")
            coefficients += [0.0] * (BLOCK_COEFFICIENTS - len(coefficients))
            handle.write(struct.pack("<11d", *coefficients))
        for cell in cells:
            handle.write(struct.pack("<ii", cell["first"], cell["count"]))
            handle.write(struct.pack("<d", cell["volume"]))
            handle.write(struct.pack("<3d", *cell["lo"]))
            handle.write(struct.pack("<3d", *cell["hi"]))
