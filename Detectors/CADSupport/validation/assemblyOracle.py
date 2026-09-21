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

"""Ground truth for ASSEMBLY-level transport: the ordered crossing list per ray, annotated with
WHICH VOLUME the track is in between the crossings.

Companion to `xrayOracle.py` (one leaf solid), for the failure mode it cannot see: a track that
exits volume A and is never reported entering B. Per interval it answers the SET of occupants:

| assembly situation      | what the occupancy sequence looks like                  |
| ----------------------- | ------------------------------------------------------- |
| touching parts          | `{A} -> {B}` at ONE distance: a transition, no vacuum    |
| a genuine gap           | `{A} -> {} -> {B}`, with the vacuum run's length stated  |
| a part nested in another| `{A} -> {A,B} -> {A}`                                    |
| interpenetration        | `{A} -> {A,B} -> {B}` -- occupancy is AMBIGUOUS, and the |
|                         | oracle says so rather than choosing an occupant          |
| a ray starting inside   | segment 0's occupancy is non-empty; it is reported       |

Candidate positions are merged ACROSS parts before the intervals are cut, so touching parts give
one transition `{A} -> {B}`; the merge tolerance is reported, and a vacuum run shorter than
`--thin-vacuum` is counted and flagged. An interval's occupancy comes from
`BRepClass3d_SolidClassifier` at its MIDPOINT, once per part; a midpoint OCCT calls `ON` flags the
ray `amb`.

Units
-----
Ray origins, directions and crossing distances are in the MODEL'S NATIVE UNITS (mm for every STEP
file in this corpus), not cm; `scaleToCm` is carried beside them and a consumer must apply it.

Usage
-----
  assemblyOracle.py --self-test                       # the synthetic assembly, analytic answers
  assemblyOracle.py --step <model>.step --rays N --beams M --out crossings.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cadsupport_path  # noqa: E402,F401  (puts ../tools on sys.path)

from cadsupport.occ_env import ensure_occ

ensure_occ()

from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TCollection import TCollection_AsciiString
from OCC.Core.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt, gp_Trsf

# A ray parameter this close to the origin is the origin itself; the kernel and oracles share it.
_RAY_EPS = 1.0e-9

ASSEMBLY_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------------------------
# Loading a STEP assembly as a flat list of PLACED solids
# ---------------------------------------------------------------------------------------------

class Part:
    """One placed solid in the world frame; `shape` carries its placement as a `TopLoc_Location`."""

    __slots__ = ("name", "definition", "path", "shape", "bbox")

    def __init__(self, name, definition, path, shape):
        self.name = name
        self.definition = definition
        self.path = path
        self.shape = shape
        box = Bnd_Box()
        brepbndlib.Add(shape, box)
        self.bbox = box.Get() if not box.IsVoid() else None

    def __repr__(self):
        return f"Part({self.name})"


def _label_id(label) -> str:
    s = TCollection_AsciiString()
    TDF_Tool.Entry(label, s)
    return s.ToCString()


def _label_name(label) -> str:
    try:
        n = label.GetLabelName()
        return str(n) if n else ""
    except Exception:
        return ""


def detect_step_unit_scale_to_cm(step_path: Path) -> float:
    """Same heuristic `O2_CADtoTGeo.py` uses, kept independent on purpose (this module must not
    import the 200 kB converter to read a header)."""
    data = step_path.open("rb").read(4 * 1024 * 1024).decode("latin-1", errors="ignore").upper()
    if ".MILLI." in data:
        return 0.1
    if ".CENTI." in data:
        return 1.0
    if ".METRE." in data or ".METER." in data:
        return 100.0
    if "INCH" in data:
        return 2.54
    if "FOOT" in data or "FEET" in data:
        return 30.48
    return 0.1


def load_assembly(step_path: Path, explode_solids: bool = True):
    """Every PLACED leaf solid of a STEP assembly, in the world frame: (parts, scale_to_cm).

    The parts are instances, not definitions: a prototype referenced 28 times yields 28 parts.
    """
    doc = TDocStd_Document("assembly")
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetLayerMode(True)
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {step_path}")
    reader.Transfer(doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())

    parts = []
    used = {}

    def emit(label, trsf, path):
        definition = _label_id(label)
        name = _label_name(label) or definition.replace(":", "_")
        shape = shape_tool.GetShape(label).Moved(TopLoc_Location(trsf))
        pieces = []
        if explode_solids:
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while explorer.More():
                pieces.append(topods.Solid(explorer.Current()))
                explorer.Next()
        if not pieces:
            pieces = [shape]
        for k, piece in enumerate(pieces):
            base = name if len(pieces) == 1 else f"{name}.s{k}"
            count = used.get(base, 0)
            used[base] = count + 1
            unique = base if count == 0 else f"{base}#{count}"
            parts.append(Part(unique, definition, path, piece))

    def walk(label, trsf, path):
        children = TDF_LabelSequence()
        shape_tool.GetComponents(label, children)
        if children.Length() > 0 or shape_tool.IsAssembly(label):
            for i in range(children.Length()):
                child = children.Value(i + 1)
                if shape_tool.IsReference(child):
                    referred = TDF_Label()
                    shape_tool.GetReferredShape(child, referred)
                    walk(referred, trsf.Multiplied(shape_tool.GetLocation(child).Transformation()),
                         f"{path}_{i}")
                else:
                    walk(child, trsf, f"{path}_{i}")
            return
        if shape_tool.IsSimpleShape(label):
            emit(label, trsf, path)

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    for i in range(roots.Length()):
        root = roots.Value(i + 1)
        if shape_tool.IsReference(root):
            referred = TDF_Label()
            shape_tool.GetReferredShape(root, referred)
            walk(referred, shape_tool.GetLocation(root).Transformation(), f"r{i}")
        else:
            walk(root, gp_Trsf(), f"r{i}")

    # Pin the XCAF document for the life of the process: a collected one leaves its shapes dangling.
    load_assembly._keepalive = (doc, shape_tool)
    return parts, detect_step_unit_scale_to_cm(step_path)


def assembly_from_shapes(named_shapes):
    """A synthetic assembly from (name, TopoDS_Shape) pairs -- the self-test's entry point."""
    return [Part(name, name, "synthetic", shape) for name, shape in named_shapes]


# ---------------------------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------------------------

def _ray_hits_box(bbox, origin, direction, tmax, pad):
    """Slab test. Conservative: a false positive costs one intersector call, a false negative
    costs a lost wall, so every comparison is inclusive and padded."""
    if bbox is None:
        return False
    lo = 0.0
    hi = tmax
    for axis in range(3):
        omin, omax = bbox[axis] - pad, bbox[axis + 3] + pad
        d = direction[axis]
        o = origin[axis]
        if abs(d) < 1e-300:
            if o < omin or o > omax:
                return False
            continue
        t0 = (omin - o) / d
        t1 = (omax - o) / d
        if t0 > t1:
            t0, t1 = t1, t0
        lo = max(lo, t0)
        hi = min(hi, t1)
        if lo > hi:
            return False
    return True


class AssemblyCrossingOracle:
    """The ordered, occupancy-annotated crossing list for a compound of placed parts."""

    def __init__(self, parts, merge_tolerance=1.0e-9, thin_vacuum=1.0e-6):
        self.parts = list(parts)
        self.merge_tolerance = max(merge_tolerance, _RAY_EPS)
        self.thin_vacuum = thin_vacuum
        self.intersectors = []
        self.classifiers = []
        for part in self.parts:
            intersector = IntCurvesFace_ShapeIntersector()
            intersector.Load(part.shape, _RAY_EPS)
            self.intersectors.append(intersector)
            self.classifiers.append(BRepClass3d_SolidClassifier(part.shape))

    # -- one part, one ray ---------------------------------------------------------------------

    def _candidates(self, index, origin, direction, tmax):
        intersector = self.intersectors[index]
        line = gp_Lin(gp_Pnt(*origin), gp_Dir(*direction))
        intersector.Perform(line, _RAY_EPS, tmax)
        if not intersector.IsDone():
            return None
        out = []
        for k in range(1, intersector.NbPnt() + 1):
            parameter = intersector.WParameter(k)
            if _RAY_EPS < parameter <= tmax:
                out.append(parameter)
        return out

    def _state(self, index, point):
        classifier = self.classifiers[index]
        classifier.Perform(point, _RAY_EPS)
        state = classifier.State()
        if state == TopAbs_IN:
            return 1
        if state == TopAbs_OUT:
            return 0
        if state == TopAbs_ON:
            return -1
        raise RuntimeError(f"unexpected classifier state {state}")

    # -- one ray -------------------------------------------------------------------------------

    def crossings(self, origin, direction, tmax):
        """Returns a dict describing the whole transport along this ray.

        Keys:
          `s0`        occupancy at the ray origin (list of part names; empty = vacuum)
          `seg`       [t0, t1, [occupants]] for every maximal run of constant occupancy
          `x`         the flat ordered crossing list: {t, part, s (+1 enter / -1 exit),
                      occ (occupancy AFTER), g (group: crossings sharing one distance)}
          `amb`       OCCT declined to classify somewhere on this ray
          `ovl`       some segment had two or more occupants -- occupancy is AMBIGUOUS and no
                      single volume id can be assigned
          `ovlClean`  the same, on a ray OCCT did NOT decline anywhere; quote this one, since an
                      inherited ON midpoint can fake a two-occupant segment on a grazing ray
          `thin`      number of vacuum runs shorter than `thin_vacuum`
          `contact`   number of distances at which one part is exited and another entered with no
                      vacuum in between (a touching transition)
        """
        norm = math.sqrt(sum(c * c for c in direction))
        unit = [c / norm for c in direction]

        pad = 10.0 * self.merge_tolerance
        active = [i for i, p in enumerate(self.parts)
                  if _ray_hits_box(p.bbox, origin, unit, tmax, pad)]

        ambiguous = False
        raw = []
        for i in active:
            hits = self._candidates(i, origin, unit, tmax)
            if hits is None:
                ambiguous = True
                continue
            raw.extend(hits)
        raw.sort()

        edges = [0.0]
        for t in raw:
            if t - edges[-1] > self.merge_tolerance:
                edges.append(t)
        if tmax - edges[-1] > self.merge_tolerance:
            edges.append(tmax)
        else:
            edges[-1] = tmax

        occupancy = []
        for k in range(len(edges) - 1):
            mid = 0.5 * (edges[k] + edges[k + 1])
            point = gp_Pnt(*(origin[c] + mid * unit[c] for c in range(3)))
            here = []
            for i in active:
                state = self._state(i, point)
                if state < 0:
                    ambiguous = True
                    # Inherit rather than guess: an ON midpoint is not evidence of either side.
                    if occupancy and self.parts[i].name in occupancy[-1]:
                        here.append(self.parts[i].name)
                elif state == 1:
                    here.append(self.parts[i].name)
            occupancy.append(sorted(here))

        # Maximal runs of constant occupancy.
        segments = []
        for k, occ in enumerate(occupancy):
            if segments and segments[-1][2] == occ:
                segments[-1][1] = edges[k + 1]
            else:
                segments.append([edges[k], edges[k + 1], occ])

        crossings = []
        contact = 0
        for g in range(1, len(segments)):
            before = set(segments[g - 1][2])
            after = set(segments[g][2])
            t = segments[g][0]
            occ_after = segments[g][2]
            exited = sorted(before - after)
            entered = sorted(after - before)
            for name in exited:
                crossings.append({"t": t, "part": name, "s": -1, "occ": occ_after, "g": g - 1})
            for name in entered:
                crossings.append({"t": t, "part": name, "s": +1, "occ": occ_after, "g": g - 1})
            if exited and entered:
                contact += 1

        thin = 0
        for t0, t1, occ in segments:
            if not occ and t0 > 0.0 and t1 < tmax and (t1 - t0) < self.thin_vacuum:
                thin += 1

        overlap = any(len(occ) > 1 for _, _, occ in segments)

        return {
            "o": list(origin), "d": list(unit), "tmax": tmax,
            "s0": segments[0][2] if segments else [],
            "seg": [[t0, t1, occ] for t0, t1, occ in segments],
            "x": crossings,
            "amb": bool(ambiguous),
            "ovl": bool(overlap),
            "ovlClean": bool(overlap and not ambiguous),
            "thin": thin,
            "contact": contact,
        }


# ---------------------------------------------------------------------------------------------
# Ray generation: Fibonacci directions
# ---------------------------------------------------------------------------------------------

def fibonacci_directions(n):
    out = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = golden * i
        out.append((r * math.cos(phi), r * math.sin(phi), z))
    return out


def assembly_bbox(parts):
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for part in parts:
        if part.bbox is None:
            continue
        for a in range(3):
            lo[a] = min(lo[a], part.bbox[a])
            hi[a] = max(hi[a], part.bbox[a + 3])
    return lo, hi


def raster_rays(parts, beams, n, margin_fraction=0.02):
    """`beams` Fibonacci directions x n x n impact parameters, every ray starting outside the
    assembly's bounding sphere and ending outside it."""
    lo, hi = assembly_bbox(parts)
    centre = [(lo[a] + hi[a]) / 2 for a in range(3)]
    radius = 0.5 * math.sqrt(sum((hi[a] - lo[a]) ** 2 for a in range(3)))
    radius *= (1.0 + margin_fraction)
    rays = []
    for b, d in enumerate(fibonacci_directions(beams)):
        # An orthonormal frame with `d` as its third axis.
        helper = (0.0, 0.0, 1.0) if abs(d[2]) < 0.9 else (1.0, 0.0, 0.0)
        u = (d[1] * helper[2] - d[2] * helper[1],
             d[2] * helper[0] - d[0] * helper[2],
             d[0] * helper[1] - d[1] * helper[0])
        un = math.sqrt(sum(c * c for c in u))
        u = tuple(c / un for c in u)
        v = (d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0])
        for i in range(n):
            for j in range(n):
                a = -radius + (i + 0.5) * 2 * radius / n
                c = -radius + (j + 0.5) * 2 * radius / n
                origin = [centre[k] + a * u[k] + c * v[k] - radius * d[k] for k in range(3)]
                rays.append((origin, list(d), 2 * radius, b))
    return rays


# ---------------------------------------------------------------------------------------------
# Self-test: a synthetic assembly whose every answer is known on paper
# ---------------------------------------------------------------------------------------------

def self_test() -> int:
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt as P

    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if not ok else ""))
        if not ok:
            failures.append(name)

    def box(x0, y0, z0, x1, y1, z1):
        return BRepPrimAPI_MakeBox(P(x0, y0, z0), P(x1, y1, z1)).Shape()

    # ---------------------------------------------------------------------------------------
    # The synthetic assembly. Everything is axis aligned so every answer is arithmetic.
    #
    #   x:  0     2 2      4  5      7    7+1e-6   9
    #       |--A--|--B-----|  |---C--|    |---D----|
    #        touching face      1 cm gap    1e-6 cm gap
    #
    #   E = [12,18]^3 with F = [14,16]^3 nested wholly inside it
    #   G = [20,24]x[0,2]x[0,2] and H = [23,27]x[0,2]x[0,2] interpenetrate over [23,24]
    # ---------------------------------------------------------------------------------------
    parts = assembly_from_shapes([
        ("A", box(0, 0, 0, 2, 2, 2)),
        ("B", box(2, 0, 0, 4, 2, 2)),
        ("C", box(5, 0, 0, 7, 2, 2)),
        ("D", box(7 + 1e-6, 0, 0, 9, 2, 2)),
        ("E", box(12, 12, 12, 18, 18, 18)),
        ("F", box(14, 14, 14, 16, 16, 16)),
        ("G", box(20, 0, 0, 24, 2, 2)),
        ("H", box(23, 0, 0, 27, 2, 2)),
    ])
    oracle = AssemblyCrossingOracle(parts, merge_tolerance=1e-9, thin_vacuum=1e-5)

    def names(seq):
        return [s["part"] for s in seq]

    def ts(seq):
        return [s["t"] for s in seq]

    # --- case 1: TOUCHING. One transition at x=2, not two events with a gap in between. -------
    r = oracle.crossings([-1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 6.0)
    x = r["x"]
    check("touching: 4 crossings on the A|B chord", len(x) == 4, str([(c["t"], c["part"], c["s"]) for c in x]))
    check("touching: enter A at 1, exit A and enter B at 3, exit B at 5",
          len(x) == 4 and all(abs(a - b) < 1e-9 for a, b in zip(ts(x), [1.0, 3.0, 3.0, 5.0])), str(ts(x)))
    check("touching: the shared face is ONE transition A->B, no vacuum between",
          r["contact"] == 1 and any(c["s"] == -1 and c["part"] == "A" and c["occ"] == ["B"] for c in x),
          f"contact={r['contact']} occ={[c['occ'] for c in x]}")
    check("touching: no vacuum segment between A and B",
          not any(len(o) == 0 and 1.0 < t0 < 5.0 for t0, t1, o in r["seg"]), str(r["seg"]))
    check("touching: occupancy after each crossing is A, B, B, vacuum",
          [c["occ"] for c in x] == [["A"], ["B"], ["B"], []], str([c["occ"] for c in x]))

    # --- case 2: a 1 cm GAP between B and C ---------------------------------------------------
    r = oracle.crossings([-1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 12.0)
    vac = [(t0, t1) for t0, t1, o in r["seg"] if not o and t0 > 0]
    check("gap: a vacuum run of exactly 1 cm between B and C",
          any(abs(t0 - 5.0) < 1e-9 and abs(t1 - 6.0) < 1e-9 for t0, t1 in vac), str(vac))
    check("gap: the occupancy after exiting B is vacuum",
          any(c["part"] == "B" and c["s"] == -1 and c["occ"] == [] for c in r["x"]),
          str([(c["part"], c["s"], c["occ"]) for c in r["x"]]))

    # --- case 3: a 1e-6 cm gap between C and D, resolved and flagged as thin ------------------
    check("thin gap: the 1e-6 cm vacuum between C and D is RESOLVED, not merged away",
          any(abs(t1 - t0 - 1e-6) < 1e-9 for t0, t1 in vac),
          str([(t0, t1, t1 - t0) for t0, t1 in vac]))
    check("thin gap: it is counted as a thin vacuum run", r["thin"] == 1, str(r["thin"]))
    check("thin gap: D is entered, not skipped",
          any(c["part"] == "D" and c["s"] == +1 for c in r["x"]), str(names(r["x"])))

    # ...and with a merge tolerance COARSER than the gap, C and D must report as TOUCHING.
    coarse = AssemblyCrossingOracle(parts, merge_tolerance=1e-4, thin_vacuum=1e-5)
    rc = coarse.crossings([-1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 12.0)
    coarse_vac = [(t0, t1) for t0, t1, o in rc["seg"] if not o and t0 > 0]
    check("thin-gap CONTROL: at merge tolerance 1e-4 the same 1e-6 gap is merged away, and C|D "
          "becomes a touching transition",
          not any(t1 - t0 < 1e-4 for t0, t1 in coarse_vac) and rc["thin"] == 0
          and rc["contact"] == 2,
          f"vac={coarse_vac} thin={rc['thin']} contact={rc['contact']}")

    # --- case 4: NESTING. F wholly inside E. --------------------------------------------------
    r = oracle.crossings([10.0, 15.0, 15.0], [1.0, 0.0, 0.0], 12.0)
    occ = [o for _, _, o in r["seg"]]
    check("nesting: occupancy runs vacuum, E, E+F, E, vacuum",
          occ == [[], ["E"], ["E", "F"], ["E"], []], str(occ))
    check("nesting: entering F does not exit E",
          [(c["part"], c["s"]) for c in r["x"]] ==
          [("E", 1), ("F", 1), ("F", -1), ("E", -1)], str([(c["part"], c["s"]) for c in r["x"]]))
    check("nesting: crossings at 2, 4, 6, 8",
          all(abs(a - b) < 1e-9 for a, b in zip(ts(r["x"]), [2.0, 4.0, 6.0, 8.0])), str(ts(r["x"])))
    check("nesting: reported as multiply-occupied", r["ovl"] is True, str(r["ovl"]))

    # --- case 5: INTERPENETRATION. G and H share [23,24]. -------------------------------------
    r = oracle.crossings([19.0, 1.0, 1.0], [1.0, 0.0, 0.0], 10.0)
    occ = [o for _, _, o in r["seg"]]
    check("overlap: occupancy runs vacuum, G, G+H, H, vacuum",
          occ == [[], ["G"], ["G", "H"], ["H"], []], str(occ))
    check("overlap: the oracle says AMBIGUOUS rather than choosing an occupant",
          r["ovl"] is True and any(len(o) > 1 for o in occ), str(occ))
    check("overlap: the shared slab is [23,24] i.e. t in [4,5]",
          any(len(o) > 1 and abs(t0 - 4.0) < 1e-9 and abs(t1 - 5.0) < 1e-9 for t0, t1, o in r["seg"]),
          str(r["seg"]))
    check("overlap: it survives the ambiguity filter -- `ovlClean` is the number to quote",
          r["ovlClean"] is True and r["amb"] is False, f"ovlClean={r['ovlClean']} amb={r['amb']}")

    # --- case 6: a ray STARTING INSIDE a part -------------------------------------------------
    r = oracle.crossings([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 6.0)
    check("inside start: origin occupancy is A", r["s0"] == ["A"], str(r["s0"]))
    check("inside start: first crossing is exit A / enter B at t=1",
          len(r["x"]) >= 2 and abs(r["x"][0]["t"] - 1.0) < 1e-9 and r["x"][0]["s"] == -1,
          str([(c["t"], c["part"], c["s"]) for c in r["x"]]))

    # --- case 7: a ray GRAZING the shared edge of A and B -------------------------------------
    # Along x=2 in +y the ray runs in the shared face; no interior crossing may be invented.
    r = oracle.crossings([2.0, -1.0, 1.0], [0.0, 1.0, 0.0], 6.0)
    check("grazing shared face: no interior crossing is invented",
          all(c["s"] in (+1, -1) for c in r["x"]) and len(r["x"]) % 2 == 0,
          str([(c["t"], c["part"], c["s"]) for c in r["x"]]))
    print(f"        grazing shared face x=2: seg={r['seg']} amb={r['amb']}")
    # A ray exactly along the shared EDGE x=2, z=2 of A and B.
    r = oracle.crossings([2.0, -1.0, 2.0], [0.0, 1.0, 0.0], 6.0)
    check("grazing shared edge: the list still alternates per part",
          _alternates_per_part(r["x"]), str([(c["t"], c["part"], c["s"]) for c in r["x"]]))
    print(f"        grazing shared edge x=2,z=2: seg={r['seg']} amb={r['amb']}")
    # An inherited ON midpoint can fake an overlap on a grazing ray, so it must never be CLEAN.
    check("grazing: an OCCT-ambiguous ray never reports a CLEAN overlap",
          r["ovlClean"] is False, f"ovl={r['ovl']} ovlClean={r['ovlClean']} amb={r['amb']}")

    # --- case 8: the alternation invariant, on a Fibonacci fan over the whole assembly --------
    rays = raster_rays(parts, beams=32, n=6)
    bad_alt = 0
    bad_occ = 0
    amb = 0
    with_overlap = 0
    for origin, d, tmax, _ in rays:
        r = oracle.crossings(origin, d, tmax)
        amb += bool(r["amb"])
        with_overlap += bool(r["ovl"])
        if not _alternates_per_part(r["x"]):
            bad_alt += 1
        if not _occupancy_consistent(r):
            bad_occ += 1
    check(f"fan ({len(rays)} rays): every part's crossings alternate enter/exit",
          bad_alt == 0, f"{bad_alt} rays")
    check(f"fan ({len(rays)} rays): occupancy after every crossing equals the segment occupancy",
          bad_occ == 0, f"{bad_occ} rays")
    print(f"        fan: {len(rays)} rays, {amb} ambiguous, {with_overlap} with multiple occupancy")

    # --- case 9: the NEGATIVE control -- the overlap flag must be able to be false ------------
    clean = assembly_from_shapes([("A", box(0, 0, 0, 2, 2, 2)), ("B", box(2, 0, 0, 4, 2, 2))])
    clean_oracle = AssemblyCrossingOracle(clean)
    r = clean_oracle.crossings([-1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 8.0)
    check("negative control: a touching-only pair reports NO multiple occupancy",
          r["ovl"] is False, str(r["seg"]))
    # ...and that the same flag fires when the same two boxes are made to interpenetrate.
    dirty = assembly_from_shapes([("A", box(0, 0, 0, 2, 2, 2)), ("B", box(1.9, 0, 0, 4, 2, 2))])
    dirty_oracle = AssemblyCrossingOracle(dirty)
    r = dirty_oracle.crossings([-1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 8.0)
    check("positive control: nudging one box 0.1 cm into the other DOES fire the flag",
          r["ovl"] is True, str(r["seg"]))

    print(f"\n{'SELF-TEST PASSED' if not failures else 'SELF-TEST FAILED'}: "
          f"{len(failures)} failure(s) of {9}")
    return 0 if not failures else 1


def _alternates_per_part(crossings):
    last = {}
    for c in crossings:
        previous = last.get(c["part"])
        if previous is not None and previous == c["s"]:
            return False
        last[c["part"]] = c["s"]
    return True


def _occupancy_consistent(ray):
    """Every crossing's `occ` must be the occupancy of the segment it opens."""
    occ_by_group = {g: seg[2] for g, seg in enumerate(ray["seg"])}
    for c in ray["x"]:
        if c["occ"] != occ_by_group.get(c["g"] + 1):
            return False
    return True


# ---------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=Path, help="the STEP assembly")
    parser.add_argument("--out", type=Path, help="where to write the crossing lists (JSON)")
    parser.add_argument("--beams", type=int, default=32, help="Fibonacci directions")
    parser.add_argument("--raster", type=int, default=8, help="impact parameters per beam, N x N")
    parser.add_argument("--parts", type=str, default="",
                        help="comma-separated instance names to keep (default: all)")
    parser.add_argument("--max-parts", type=int, default=0, help="keep only the first N parts")
    parser.add_argument("--thin-vacuum", type=float, default=1.0e-6,
                        help="a vacuum run shorter than this (cm) is counted as thin")
    parser.add_argument("--self-test", action="store_true",
                        help="the synthetic assembly: touching, gap, thin gap, nesting, overlap, "
                             "inside start, grazing; needs no model")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.step:
        parser.error("--step is required (unless --self-test)")

    started = time.time()
    parts, scale = load_assembly(args.step)
    if args.parts:
        wanted = set(args.parts.split(","))
        parts = [p for p in parts if p.name in wanted]
    if args.max_parts:
        parts = parts[:args.max_parts]
    print(f"  {args.step.name}: {len(parts)} placed solids, scale {scale} cm/unit "
          f"({time.time() - started:.1f} s)", flush=True)

    oracle = AssemblyCrossingOracle(parts, thin_vacuum=args.thin_vacuum / scale)
    rays = raster_rays(parts, args.beams, args.raster)
    print(f"  {len(rays)} rays ({args.beams} Fibonacci directions x {args.raster}^2)", flush=True)

    answers = []
    stats = {"rays": 0, "crossings": 0, "amb": 0, "ovl": 0, "ovlClean": 0, "thin": 0,
             "contact": 0, "insideStart": 0, "empty": 0}
    t0 = time.time()
    for k, (origin, d, tmax, beam) in enumerate(rays):
        r = oracle.crossings(origin, d, tmax)
        r["beam"] = beam
        answers.append(r)
        stats["rays"] += 1
        stats["crossings"] += len(r["x"])
        stats["amb"] += bool(r["amb"])
        stats["ovl"] += bool(r["ovl"])
        stats["ovlClean"] += bool(r["ovlClean"])
        stats["thin"] += r["thin"]
        stats["contact"] += r["contact"]
        stats["insideStart"] += bool(r["s0"])
        stats["empty"] += (not r["x"])
        if (k + 1) % 200 == 0:
            print(f"    {k + 1}/{len(rays)} rays ({time.time() - t0:.1f} s)", flush=True)

    document = {"version": ASSEMBLY_FORMAT_VERSION, "model": str(args.step),
                "scaleToCm": scale, "mergeTolerance": oracle.merge_tolerance,
                "parts": [p.name for p in parts], "stats": stats,
                "oracleSeconds": time.time() - t0, "rays": answers}
    if args.out:
        args.out.write_text(json.dumps(document))
    print(f"  {stats['rays']} rays, {stats['crossings']} crossings, {stats['contact']} touching "
          f"transitions, {stats['ovlClean']} rays with ambiguous occupancy "
          f"({stats['ovl']} before excluding OCCT-ambiguous rays), {stats['thin']} thin "
          f"vacuum runs, {stats['amb']} ambiguous ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
