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

"""Is a part's tessellation the same solid as its exact surfaces?

It is exactly when every face is a planar polygon with no arc or B-spline edge, the condition under
which `LoadSurfaceSolid` builds only `PlanarPolygon` records. It measures; it does not route.
"""

import struct

SIDECAR_MAGIC = b"O2SS"
SIDECAR_VERSION_MIN = 1
SIDECAR_VERSION_MAX = 3

# The sidecar's own surface-type numbering (not BVHSurfaceRecord::Kind, which is decided on read).
TYPE_NAME = {1: "plane", 2: "cylinder", 3: "cone", 4: "sphere", 5: "torus"}
TYPE_PLANE = 1

# Curve types in a wire edge record. Anything that is not a line segment makes a plane curved.
CURVE_LINE = 0


class _Cursor:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def u32(self):
        value = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return value

    def u8(self):
        value = self.data[self.offset]
        self.offset += 1
        return value

    def f64(self):
        value = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return value

    def skip_doubles(self, n):
        self.offset += 8 * n


def surface_census(path):
    """`{'planarPolygon': n, 'curvedPlanar': n, '<curved kind>': n}` for one `surfaces_*.bin`.

    Raises `ValueError` on a file it cannot read, so "not exact" and "could not tell" stay apart.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 16 or data[:4] != SIDECAR_MAGIC:
        raise ValueError(f"{path} is not a surface sidecar (bad magic)")
    cursor = _Cursor(data)
    cursor.offset = 4
    version = cursor.u32()
    n_surfaces = cursor.u32()
    cursor.u32()                                   # reserved
    if not SIDECAR_VERSION_MIN <= version <= SIDECAR_VERSION_MAX:
        raise ValueError(f"{path}: unsupported sidecar version {version}")
    if version >= 2:
        cursor.f64()                               # model tolerance
        if version >= 3:
            cursor.u32()                           # nModelEdges

    counts = {}
    for _ in range(n_surfaces):
        surface_type = cursor.u32()
        cursor.u32()                               # flags
        cursor.skip_doubles(cursor.u32())          # params
        straight = True
        for _ in range(cursor.u32()):              # wires
            cursor.u32()                           # role
            for _ in range(cursor.u32()):          # edges
                if cursor.u32() != CURVE_LINE:
                    straight = False
                cursor.skip_doubles(cursor.u32())  # curve params
        if version >= 3:
            for _ in range(cursor.u32()):          # edge identities
                cursor.u32()
                cursor.u8()
        if surface_type == TYPE_PLANE:
            key = "planarPolygon" if straight else "curvedPlanar"
        else:
            key = TYPE_NAME.get(surface_type, f"unknown{surface_type}")
        counts[key] = counts.get(key, 0) + 1
    return counts


def tessellation_is_exact(path):
    """`(exact, reason, census)` for one part's sidecar; `(None, why, None)` when it cannot be read."""
    try:
        counts = surface_census(path)
    except Exception as error:                      # a file we cannot read is not a verdict
        return None, str(error), None
    if not counts:
        return None, "the sidecar carries no surfaces", counts
    planar = counts.get("planarPolygon", 0)
    if planar == sum(counts.values()):
        return True, (f"all {planar} faces are planar polygons, so triangulating them loses "
                      "nothing"), counts
    curved = {k: v for k, v in counts.items() if k not in ("planarPolygon",)}
    if list(curved) == ["curvedPlanar"]:
        return False, (f"{curved['curvedPlanar']} face(s) are flat but have a curved boundary (an "
                       "arc or a spline), so the face is exact and its outline is not"), counts
    named = ", ".join(f"{v} {k}" for k, v in sorted(curved.items(), key=lambda kv: -kv[1]))
    return False, f"{named} face(s) are not planar polygons", counts
