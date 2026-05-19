#!/usr/bin/env python3
"""
A Python script, doing a deep STEP/XCAF -> ROOT TGeo conversion.
For now, all CAD solids are simply meshed. The ROOT geometry is build as a C++ ROOT macro
and facet data is stored in binary form to keep disc space minimal.

NEW (03/2026):
  - Optional material/medium emission from a BOM (bill of materials) CSV file.
    The CSV is expected to contain lines like:
      CAD, Mechanical/Part, <PartNumber>, <Rev>, <Name>, <Mass>, <Material>, ...
  - If both a part mass and a CAD volume are available, an effective density is computed
    and used in the emitted TGeoMaterial. Otherwise a reasonable default density is used
    for a few common materials, or 1.0 g/cm^3 as fallback.

Generates (into --output-folder):
  - geom.C (small ROOT macro)
  - facets_<VOLNAME>_<LID>.bin for each leaf logical volume (float32 triangles)

Facet file format (little-endian):
  uint32 nTriangles
  then nTriangles * 9 * float32:
    ax ay az bx by bz cx cy cz

VOLNAME is a filename-safe version of the XCAF label name when available (e.g. "nut"),
and LID is the XCAF label entry (e.g. "0:1:1:7" -> "0_1_1_7") to keep filenames unique.

Naming:
  - C++ variable names stay based on XCAF label entry (e.g. 0:1:1:7) for uniqueness.
  - ROOT object names (TGeoVolume / TGeoTessellated / TGeoVolumeAssembly) use the label's
    human name when available (e.g. "nut", "rod-assembly"), falling back to the entry.

Units:
  - By default, the script tries to detect the STEP LENGTH unit by scanning the STEP file
    header/contents (common patterns like .MILLI. / .CENTI. / .METRE. / INCH / FOOT).
  - You can override with --step-unit {auto,mm,cm,m,in,ft}. TGeo expects cm.

Author:
  - Sandro Wenzel, CERN (02/2026)
  - Material/BOM integration patch (03/2026)
"""

import warnings
warnings.filterwarnings("ignore", message=".*all to deprecated function.*", category=DeprecationWarning)

import argparse
import csv
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Extend.TopologyUtils import TopologyExplorer

from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool
from OCC.Core.TCollection import TCollection_AsciiString
from OCC.Core.gp import gp_Trsf

# volume properties for density calcs (may not be present in older pythonOCC builds)
try:
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties
    _HAS_VOLPROPS = True
except Exception:
    _HAS_VOLPROPS = False


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


# -------------------------------
# Triangulation helpers
# -------------------------------

def _scale_triangles(triangles, s: float):
    if s == 1.0:
        return triangles
    out = []
    for (a, b, c) in triangles:
        out.append((
            (a[0] * s, a[1] * s, a[2] * s),
            (b[0] * s, b[1] * s, b[2] * s),
            (c[0] * s, c[1] * s, c[2] * s),
        ))
    return out


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
    return _scale_triangles(triangles, scale_to_cm)


def triangulate_CAD_solid(my_solid, meshparam, scale_to_cm: float = 1.0):
    lin_defl = float(meshparam.get("lin_defl", 0.1))
    ang_defl = float(meshparam.get("ang_defl", 0.1))

    parallel = True
    try:
        BRepMesh_IncrementalMesh(my_solid, lin_defl, False, ang_defl, bool(parallel))
    except TypeError:
        BRepMesh_IncrementalMesh(my_solid, lin_defl, False, ang_defl)

    triangles = []
    for face in TopologyExplorer(my_solid).faces():
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            continue

        trsf = loc.Transformation()
        reverse = (face.Orientation() == TopAbs_REVERSED)

        for i in range(1, triangulation.NbTriangles() + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()

            p1 = triangulation.Node(n1).Transformed(trsf)
            p2 = triangulation.Node(n2).Transformed(trsf)
            p3 = triangulation.Node(n3).Transformed(trsf)

            if reverse:
                p2, p3 = p3, p2

            triangles.append((
                (p1.X(), p1.Y(), p1.Z()),
                (p2.X(), p2.Y(), p2.Z()),
                (p3.X(), p3.Y(), p3.Z()),
            ))

    return _scale_triangles(triangles, scale_to_cm)


# -------------------------------
# Volume helpers (for density)
# -------------------------------

def volume_cm3_of_shape(shape, scale_to_cm: float) -> float:
    """Compute CAD solid volume in cm^3 (using STEP->cm scale)."""
    if _HAS_VOLPROPS:
        try:
            props = GProp_GProps()
            brepgprop_VolumeProperties(shape, props)
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


# -------------------------------
# Naming helpers
# -------------------------------

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
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(triangles)))
        for (a, b, c) in triangles:
            f.write(struct.pack(
                "<9f",
                float(a[0]), float(a[1]), float(a[2]),
                float(b[0]), float(b[1]), float(b[2]),
                float(c[0]), float(c[1]), float(c[2]),
            ))


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
    s = s.replace("en aw", " ")
    s = s.replace("en-aw", " ")
    s = s.replace("en", " ")
    s = s.replace("aw", " ")
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


def emit_cpp_prelude() -> str:
    return """#include <TGeoManager.h>
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


def emit_materials_cpp(
    used_materials: Dict[str, ResolvedMaterial],
    # key: BOM material string as used in CSV after normalization
) -> Tuple[str, Dict[str, str]]:
    """
    Emits C++ code defining TGeoMaterial/TGeoMixture + TGeoMedium for all used materials.

    - If a material resolved to a Geant4 NIST entry, emit a physically correct mixture
      (element mass fractions) and set RadLen/IntLen (from Geant4) when available.
    - If unresolved/ambiguous, emit a dummy material and annotate with FIXME comments.
    """
    cpp: List[str] = []
    cpp.append("  // Default material/medium (placeholder; can be replaced later)")
    cpp.append("  TGeoMaterial *mat_Default = new TGeoMaterial(\"Default\", 0., 0., 0.);")
    cpp.append("  TGeoMedium   *med_Default = new TGeoMedium(\"Default\", 1, mat_Default);")
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
        cpp.append("")
        medium_var[bom_mat] = f"med_{safe}"
        next_id += 1

    return "\n".join(cpp), medium_var




def emit_tessellated_cpp(lid: str, vol_display_name: str, facet_abspath: str, ntriangles: int, medium_var: str) -> str:
    safe = sanitize_cpp_name(lid)
    shape_name = vol_display_name if vol_display_name else lid

    if ntriangles <= 0:
        out = []
        out.append(f'  TGeoBBox *solid_{safe} = new TGeoBBox("{shape_name}", 0.001, 0.001, 0.001);')
        out.append(f'  TGeoVolume *vol_{safe} = new TGeoVolume("{shape_name}", solid_{safe}, {medium_var});')
        return "\n".join(out)

    out = []
    out.append(f'  TGeoTessellated *solid_{safe} = new TGeoTessellated("{shape_name}", {ntriangles});')
    out.append(f'  LoadFacets("{facet_abspath}", solid_{safe}, check);')
    out.append(f'  TGeoVolume *vol_{safe} = new TGeoVolume("{shape_name}", solid_{safe}, {medium_var});')
    return "\n".join(out)


def emit_assembly_cpp(lid: str, asm_display_name: str) -> str:
    safe = sanitize_cpp_name(lid)
    name = asm_display_name if asm_display_name else lid
    return f'  TGeoVolumeAssembly *asm_{safe} = new TGeoVolumeAssembly("{name}");'


# -------------------------------
# Definition graph extraction
# -------------------------------

logical_volumes: Dict[str, list] = {}     # def_lid -> triangles
def_names: Dict[str, str] = {}           # def_lid -> human display name (may be "")
def_volumes_cm3: Dict[str, float] = {}   # def_lid -> volume in cm^3 (leaf only)
assemblies = set()                       # def_lid
placements = []                          # (parent_def_lid, child_def_lid, gp_Trsf local)
top_defs = set()                         # top definition lids
visited_defs = set()                     # expanded defs


def cpp_var_for_def(lid: str) -> str:
    safe = sanitize_cpp_name(lid)
    return f"asm_{safe}" if lid in assemblies else f"vol_{safe}"


def expand_definition(def_label: TDF_Label, shape_tool, meshparam=None, scale_to_cm: float = 1.0):
    def_lid = label_id(def_label)
    if def_lid in visited_defs:
        return
    visited_defs.add(def_lid)

    nm = label_name(def_label)
    if nm and def_lid not in def_names:
        def_names[def_lid] = nm
    elif def_lid not in def_names:
        def_names[def_lid] = ""

    children = TDF_LabelSequence()
    shape_tool.GetComponents(def_label, children)
    has_children = children.Length() > 0

    if has_children or shape_tool.IsAssembly(def_label):
        assemblies.add(def_lid)

        for i in range(children.Length()):
            child = children.Value(i + 1)
            if shape_tool.IsReference(child):
                referred = TDF_Label()
                shape_tool.GetReferredShape(child, referred)
                child_def_lid = label_id(referred)

                loc = shape_tool.GetLocation(child)
                trsf = loc.Transformation()
                placements.append((def_lid, child_def_lid, trsf))

                expand_definition(referred, shape_tool, meshparam=meshparam, scale_to_cm=scale_to_cm)
            else:
                child_def_lid = label_id(child)
                placements.append((def_lid, child_def_lid, gp_Trsf()))
                expand_definition(child, shape_tool, meshparam=meshparam, scale_to_cm=scale_to_cm)
        return

    if shape_tool.IsSimpleShape(def_label):
        if def_lid not in logical_volumes:
            shape = shape_tool.GetShape(def_label)

            # store volume (for density estimation)
            try:
                def_volumes_cm3[def_lid] = volume_cm3_of_shape(shape, scale_to_cm=scale_to_cm)
            except Exception:
                def_volumes_cm3[def_lid] = 0.0

            do_meshing = (meshparam is not None) and meshparam.get("do_meshing", None) is True
            logical_volumes[def_lid] = triangulate_CAD_solid(shape, meshparam=meshparam, scale_to_cm=scale_to_cm) if do_meshing else triangulate_asbbox(shape, scale_to_cm=scale_to_cm)
        return

    assemblies.add(def_lid)


def extract_graph(step_path: str, meshparam=None, scale_to_cm: float = 1.0):
    global logical_volumes, def_names, def_volumes_cm3, assemblies, placements, top_defs, visited_defs
    logical_volumes = {}
    def_names = {}
    def_volumes_cm3 = {}
    assemblies = set()
    placements = []
    top_defs = set()
    visited_defs = set()

    doc, shape_tool = load_step_with_xcaf(step_path)

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)

    for i in range(roots.Length()):
        root = roots.Value(i + 1)
        if shape_tool.IsReference(root):
            ref = TDF_Label()
            shape_tool.GetReferredShape(root, ref)
            top_defs.add(label_id(ref))
            expand_definition(ref, shape_tool, meshparam=meshparam, scale_to_cm=scale_to_cm)
        else:
            top_defs.add(label_id(root))
            expand_definition(root, shape_tool, meshparam=meshparam, scale_to_cm=scale_to_cm)

    return doc, shape_tool


# -------------------------------
# ROOT macro emission
# -------------------------------

def emit_placement_cpp(parent_def: str, child_def: str, trsf: gp_Trsf, copy_no: int, scale_to_cm: float) -> str:
    parent_cpp = cpp_var_for_def(parent_def)
    child_cpp = cpp_var_for_def(child_def)
    tr_name = f"tr_{sanitize_cpp_name(parent_def)}_{sanitize_cpp_name(child_def)}_{copy_no}"
    return trsf_to_tgeo(trsf, tr_name, scale_to_cm) + f"  {parent_cpp}->AddNode({child_cpp}, {copy_no}, {tr_name});\n"



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
    materials_csv: Optional[str] = None,
    bom_mass_unit: str = "kg",
    g4_nist_json: Optional[str] = None,
    mat_cfg: Optional[MatMatchConfig] = None,
):
    if (step_unit or "auto").lower() == "auto":
        detected = detect_step_length_unit(step_path)
        scale_to_cm = step_unit_scale_to_cm(detected)
        print(f"Detected STEP length unit: {detected} (scale to cm = {scale_to_cm})")
    else:
        scale_to_cm = step_unit_scale_to_cm(step_unit)
        print(f"Using overridden STEP length unit: {step_unit} (scale to cm = {scale_to_cm})")

    extract_graph(step_path, meshparam=meshparam, scale_to_cm=scale_to_cm)

    out_folder = out_folder.expanduser().resolve()
    out_folder.mkdir(parents=True, exist_ok=True)


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
                def_volumes_cm3.get(lid, 0.0),
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

    materials_cpp, medium_var_map = emit_materials_cpp(used_materials)

    # --- emit C++ macro ---
    cpp: List[str] = []
    cpp.append(emit_cpp_prelude())

    cpp.append("TGeoVolume* build(bool check=true) {")
    cpp.append('  if (!gGeoManager) { throw std::runtime_error("gGeoManager is null. Call build_and_export() or create a TGeoManager first."); }')
    cpp.append(materials_cpp)

    for lid in logical_volumes.keys():
        ntriangles = len(logical_volumes[lid])

        # choose medium for this volume
        med = "med_Default"
        if lid in lid_to_bom:
            mat_name = normalize_material_name(lid_to_bom[lid].material)
            med = medium_var_map.get(mat_name, "med_Default")

        cpp.append(emit_tessellated_cpp(lid, def_names.get(lid, ""), facet_files[lid], ntriangles, med))

    for lid in sorted(assemblies):
        cpp.append(emit_assembly_cpp(lid, def_names.get(lid, "")))

    for idx, (parent, child, trsf) in enumerate(placements, start=1):
        cpp.append(emit_placement_cpp(parent, child, trsf, idx, scale_to_cm))

    if len(top_defs) == 1:
        top = next(iter(top_defs))
        cpp.append(f"  return {cpp_var_for_def(top)};")
    else:
        cpp.append('  TGeoVolumeAssembly *asm_WORLD = new TGeoVolumeAssembly("WORLD");')
        for i, node in enumerate(sorted(top_defs), start=1):
            cpp.append(f"  asm_WORLD->AddNode({cpp_var_for_def(node)}, {i});")
        cpp.append("  return asm_WORLD;")

    cpp.append("}")

    # exports a function allowing to export the geometry to TGeo file
    cpp.append('void build_and_export(const char* out_root = "geom.root", bool check=true) {')
    cpp.append('  if (!gGeoManager) { new TGeoManager("geom","geom"); }')
    cpp.append('  TGeoVolume* top = build(check);')
    cpp.append('  gGeoManager->SetTopVolume(top);')
    cpp.append('  gGeoManager->CloseGeometry();')
    cpp.append('  gGeoManager->CheckOverlaps();')
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

def label_entry(label):
    s = TCollection_AsciiString()
    TDF_Tool.Entry(label, s)
    return s.ToCString()


def traverse_print(label, shape_tool, depth=0):
    indent = "  " * depth
    name = label.GetLabelName()
    entry = label_entry(label)
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
    ap.add_argument("step", help="Input STEP file")
    ap.add_argument("-o", "--out", default="geom.C", help="Output ROOT macro file name (default: geom.C)")
    ap.add_argument("--output-folder", default="./", help="Output folder for macro + facet files")
    ap.add_argument("--out-path", default=None, help="(deprecated) Alias for --output-folder")
    ap.add_argument("--mesh", action="store_true", help="Use full BRepMesh triangulation instead of bounding boxes")
    ap.add_argument("--print-tree", action="store_true", help="Just prints the geometry tree")
    ap.add_argument("--mesh-prec", default=0.1, help="meshing precision. lower --> slower")
    ap.add_argument("--step-unit", default="auto", choices=["auto", "mm", "cm", "m", "in", "ft"], help="STEP length unit override (default: auto-detect); TGeo expects cm")

    # NEW: BOM / material support
    ap.add_argument("--materials-csv", default=None, help="BOM CSV file providing material + mass per part (optional)")
    ap.add_argument("--bom-mass-unit", default="kg", choices=["kg", "g"], help="Unit of the BOM mass column (default: kg)")
    ap.add_argument("--g4-nist-json", default=None, help="Path to Geant4 NIST DB JSON dump (from nist_export_all). Enables TGeoMixture emission + RadLen/IntLen.")


    # Material matching scoring knobs (only used if --g4-nist-json is provided)
    ap.add_argument("--mat-min-score", type=float, default=0.35, help="Minimum combined score to accept a G4 NIST material match (default: 0.35)")
    ap.add_argument("--mat-ambiguity-delta", type=float, default=0.05, help="If best-second < delta, treat match as ambiguous/unresolved (default: 0.05)")
    ap.add_argument("--mat-w-token", type=float, default=0.75, help="Weight for token/name similarity score (default: 0.75)")
    ap.add_argument("--mat-w-density", type=float, default=0.25, help="Weight for density proximity score (default: 0.25)")
    ap.add_argument("--mat-max-log-density-diff", type=float, default=0.0, help="Optional hard density filter in log-space (0 disables). Example 0.8 ~ within 2.2x (default: 0.0)")
    ap.add_argument("--mat-compound-penalty", type=float, default=0.25, help="Penalty for matching to oxides/carbides/etc. when BOM doesn't mention them (default: 0.25)")

    args = ap.parse_args()

    step_path = str(_Path(args.step).expanduser().resolve())
    if args.print_tree:
        print_geom(step_path)
        return

    out_folder = _Path(args.output_folder)
    if args.out_path is not None:
        out_folder = _Path(args.out_path)

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
        materials_csv=args.materials_csv,
        bom_mass_unit=args.bom_mass_unit,
        g4_nist_json=args.g4_nist_json,
        mat_cfg=mat_cfg,
    )
    out_macro.write_text(code)

    print(f"Wrote ROOT macro: {out_macro}")
    print(f"Wrote facet files into: {out_folder}")
    print("In ROOT you can do:")
    print(f"  root -l {out_macro}")
    print('  build_and_export("geom.root");')


if __name__ == "__main__":
    main()
