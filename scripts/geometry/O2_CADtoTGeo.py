#!/usr/bin/env python3
"""
A Python script, doing a deep STEP/XCAF -> ROOT TGeo conversion.
For now, all CAD solids are simply meshed. The ROOT geometry is build as a C++ ROOT macro
and facet data is stored in binary form to keep disc space minimal.

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
"""

import warnings
warnings.filterwarnings("ignore", message=".*all to deprecated function.*", category=DeprecationWarning)

import argparse
import re
import struct
from pathlib import Path as _Path

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


def emit_materials_cpp() -> str:
    return """  // Default material/medium (placeholder; can be replaced later)
  TGeoMaterial *mat_Default = new TGeoMaterial("Default", 0., 0., 0.);
  TGeoMedium   *med_Default = new TGeoMedium("Default", 1, mat_Default);
"""


def emit_tessellated_cpp(lid: str, vol_display_name: str, facet_abspath: str, ntriangles: int) -> str:
    safe = sanitize_cpp_name(lid)
    shape_name = vol_display_name if vol_display_name else lid

    if ntriangles <= 0:
        out = []
        out.append(f'  TGeoBBox *solid_{safe} = new TGeoBBox("{shape_name}", 0.001, 0.001, 0.001);')
        out.append(f'  TGeoVolume *vol_{safe} = new TGeoVolume("{shape_name}", solid_{safe}, med_Default);')
        return "\n".join(out)

    out = []
    out.append(f'  TGeoTessellated *solid_{safe} = new TGeoTessellated("{shape_name}", {ntriangles});')
    out.append(f'  LoadFacets("{facet_abspath}", solid_{safe}, check);')
    out.append(f'  TGeoVolume *vol_{safe} = new TGeoVolume("{shape_name}", solid_{safe}, med_Default);')
    return "\n".join(out)


def emit_assembly_cpp(lid: str, asm_display_name: str) -> str:
    safe = sanitize_cpp_name(lid)
    name = asm_display_name if asm_display_name else lid
    return f'  TGeoVolumeAssembly *asm_{safe} = new TGeoVolumeAssembly("{name}");'


# -------------------------------
# Definition graph extraction
# -------------------------------

logical_volumes = {}     # def_lid -> triangles
def_names = {}           # def_lid -> human display name (may be "")
assemblies = set()       # def_lid
placements = []          # (parent_def_lid, child_def_lid, gp_Trsf local)
top_defs = set()         # top definition lids
visited_defs = set()     # expanded defs


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
            do_meshing = (meshparam is not None) and meshparam.get("do_meshing", None) is True
            logical_volumes[def_lid] = triangulate_CAD_solid(shape, meshparam=meshparam, scale_to_cm=scale_to_cm) if do_meshing else triangulate_asbbox(shape, scale_to_cm=scale_to_cm)
        return

    assemblies.add(def_lid)


def extract_graph(step_path: str, meshparam=None, scale_to_cm: float = 1.0):
    global logical_volumes, def_names, assemblies, placements, top_defs, visited_defs
    logical_volumes = {}
    def_names = {}
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


def emit_root_macro(step_path: str, out_folder: _Path, meshparam=None, step_unit: str = "auto"):
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

    facet_files = {}  # def_lid -> absolute path string
    for lid, tris in logical_volumes.items():
        disp = def_names.get(lid, "")
        volname = sanitize_filename(disp) if disp else "vol"
        lidname = sanitize_filename(lid)
        fname = f"facets_{volname}_{lidname}.bin"
        fpath = (out_folder / fname).resolve()
        write_facets_bin(fpath, tris)
        facet_files[lid] = str(fpath).replace("\\", "\\\\")  # C++ string literal safety

    cpp = []
    cpp.append(emit_cpp_prelude())

    cpp.append("TGeoVolume* build(bool check=true) {")
    cpp.append('  if (!gGeoManager) { throw std::runtime_error("gGeoManager is null. Call build_and_export() or create a TGeoManager first."); }')
    cpp.append(emit_materials_cpp())

    for lid in logical_volumes.keys():
        ntriangles = len(logical_volumes[lid])
        cpp.append(emit_tessellated_cpp(lid, def_names.get(lid, ""), facet_files[lid], ntriangles))

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
    ap.add_argument("--step-unit", default="auto", choices=["auto", "mm", "cm", "m", "in", "ft"], help="STEP length unit override (default: auto-detect)")

    args = ap.parse_args()

    step_path = str(_Path(args.step).expanduser().resolve())
    if args.print_tree:
        print_geom(step_path)
        return

    out_folder = _Path(args.output_folder)
    if args.out_path is not None:
        out_folder = _Path(args.out_path)

    meshparam = {"do_meshing": args.mesh, "lin_defl": args.mesh_prec, "ang_defl": args.mesh_prec}

    out_folder = out_folder.expanduser().resolve()
    out_folder.mkdir(parents=True, exist_ok=True)

    out_macro = (out_folder / _Path(args.out).name).resolve()
    code = emit_root_macro(step_path, out_folder, meshparam=meshparam, step_unit=args.step_unit)
    out_macro.write_text(code)

    print(f"Wrote ROOT macro: {out_macro}")
    print(f"Wrote facet files into: {out_folder}")
    print("In ROOT you can do:")
    print(f"  root -l {out_macro}")
    print('  build_and_export("geom.root");')


if __name__ == "__main__":
    main()
