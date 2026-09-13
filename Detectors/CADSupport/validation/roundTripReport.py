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

"""The TGeo -> STEP -> TGeo report: what the round trip does to every part of a geometry.

For every part it gives the source `TGeoShape` (class, and for a boolean its depth and leaf
classes), the representation the cascade emitted with its evidence or decline reason, and the
known-source verdict. The **feature matrix** sets every source shape class in the corpus against
what the round trip made of it. Every number is read from an existing instrument
(`csg_report.json`, `checkKnownSource.py`, the sidecars, `exportSourceShapes.py`); a field it
cannot read is reported as unknown, never as zero.

Usage
-----
    # the whole corpus, one markdown document
    roundTripReport.py --corpus <root> --out report.md

    # the same as a standalone HTML page (print it to PDF from a browser)
    roundTripReport.py --corpus <root> --out report.html --html

    # one part, on the fly
    roundTripReport.py --corpus <root> --part BREF1

`<root>` holds one directory per module, each with `o2sim_geometry.root`,
`<MOD>_writer_report.json` and a conversion subdirectory (`conv/` by default) containing
`csg_report.json`. `--converted-root` points the conversions somewhere else, `--modules` selects a
subset, and `--no-source-shapes` skips the (ROOT-loading) source description when only the
converter's own side is wanted.
"""

import argparse
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cadsupport_path  # noqa: E402,F401  (puts ../tools on sys.path)
from cadsupport import planar  # noqa: E402


# ------------------------------------------------------------------------------------------
# reading a corpus
# ------------------------------------------------------------------------------------------

def _load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def source_descriptions(module, source_dir, conv_dir, refresh=False):
    """`{stem: description}` of the shape each part was made from, cached beside the conversion.

    Delegates to `exportSourceShapes.export_run(write=False)`; cached in `original_report.json`.
    """
    cache = Path(conv_dir) / "original_report.json"
    if cache.exists() and not refresh:
        payload = _load(cache, {})
        return {row["part"]: row for row in payload.get("parts", []) if row.get("part")}
    geometry = Path(source_dir) / "o2sim_geometry.root"
    writer = Path(source_dir) / f"{module}_writer_report.json"
    if not geometry.exists() or not writer.exists():
        return {}
    try:
        import exportSourceShapes
        import ROOT
        ROOT.gErrorIgnoreLevel = ROOT.kWarning     # one Import banner per module is not a finding
        records = exportSourceShapes.export_run(str(geometry), str(writer), str(conv_dir),
                                                verbose=False, write=False,
                                                tiers=("csg", "surface", "mesh"))
    except Exception as error:
        print(f"  {module}: could not describe source shapes ({error})", file=sys.stderr)
        return {}
    cache.write_text(json.dumps({"parts": records}, indent=1))
    return {row["part"]: row for row in records if row.get("part")}


def read_module(module, source_dir, conv_dir, with_sources=True, refresh=False):
    """Every part of one module, joined across the instruments. Returns (rows, module summary)."""
    report = _load(Path(conv_dir) / "csg_report.json")
    if not report:
        return [], {"module": module, "error": f"no csg_report.json in {conv_dir}"}

    known = _load(Path(conv_dir) / "knownsource.json", {})
    known_rows = {}
    for row in (known.get("parts") if isinstance(known, dict) else known) or []:
        if row.get("part"):
            known_rows[row["part"]] = row

    sources = source_descriptions(module, source_dir, conv_dir, refresh) if with_sources else {}
    writer = _load(Path(source_dir) / f"{module}_writer_report.json", {}) or {}

    rows = []
    for part in report.get("parts", []):
        stem = part.get("part")
        evidence = part.get("evidence") or {}
        source = sources.get(stem) or {}
        ks = known_rows.get(stem) or {}
        # A part carried as CSG that also wrote a flat sidecar ships the flat solid, not a tree.
        tier = part.get("representation")
        if tier == "csg" and part.get("flatSidecar"):
            tier = "flatcsg"
        surfaces = Path(conv_dir) / f"surfaces_{stem}.bin"
        # Older conversions lack this field, so it is computed from the sidecar.
        exact, exact_why = part.get("tessellationExact"), part.get("tessellationExactWhy")
        census = part.get("surfaceCensus")
        if exact is None and surfaces.exists():
            exact, exact_why, census = planar.tessellation_is_exact(str(surfaces))
        rows.append({
            "module": module,
            "part": stem,
            "volume": part.get("volume"),
            "sourceVolume": source.get("sourceVolume") or ks.get("source"),
            "sourceClass": source.get("class") or ks.get("sourceClass"),
            "booleanDepth": source.get("booleanDepth"),
            "leaves": source.get("leaves"),
            "leafClasses": source.get("leafClasses") or {},
            "ships": tier,
            "recogniser": evidence.get("recogniser"),
            "structure": evidence.get("description", {}).get("op")
                         if isinstance(evidence.get("description"), dict) else None,
            "dVsym": evidence.get("symmetricDifferenceCm3"),
            "band": evidence.get("bandCm3"),
            "relative": evidence.get("relativeToVolume"),
            "whyNotCSG": part.get("whyNotCSG"),
            "tessellationExact": exact,
            "tessellationExactWhy": exact_why,
            "surfaceCensus": census or {},
            "knownSourceFailures": ks.get("failures"),
            "knownSourceFlags": ks.get("flags"),
            "capacityRelativeDeviation": ks.get("capacityRelativeDeviation"),
            "containsMismatches": (ks.get("contains") or {}).get("mismatches"),
            "containsPoints": (ks.get("contains") or {}).get("points"),
            "sidecarBytes": _size(surfaces),
            "flatSidecarBytes": _size(Path(conv_dir) / f"flatcsg_{stem}.bin"),
            "facetBytes": _size(Path(conv_dir) / f"facets_{stem}.bin"),
        })

    exactness = {"exact": sum(1 for r in rows if r["tessellationExact"])}
    summary = {
        "module": module,
        "leafSolids": report.get("nLeafSolids"),
        "tiers": report.get("tiers") or {},
        # The writer's own coverage: volumes it declined never reach the converter.
        "writerByShapeClass": writer.get("byShapeClass") or {},
        "writerVisited": writer.get("volumesVisited"),
        "writerDefinitions": writer.get("definitions"),
        "writerDeclined": writer.get("declined"),
        "writerDeclinedRows": [v for v in (writer.get("volumes") or [])
                               if not v.get("converted") and not v.get("isAssembly")],
        "writerVolumes": writer.get("volumes") or [],
        "flat": sum(1 for r in rows if r["ships"] == "flatcsg"),
        "tessellationExact": exactness.get("exact"),
        "knownSourceScored": len(known_rows),
        "knownSourceFailed": sum(1 for r in known_rows.values() if r.get("failures")),
        "error": None,
    }
    return rows, summary


def read_corpus(root, converted_root=None, subdir="conv", modules=None,
                with_sources=True, refresh=False):
    root = Path(root)
    names = modules or sorted(p.name for p in root.iterdir() if p.is_dir())
    rows, summaries = [], []
    for module in names:
        source_dir = root / module
        conv_dir = (Path(converted_root) / module) if converted_root else (source_dir / subdir)
        if not (Path(conv_dir) / "csg_report.json").exists():
            summaries.append({"module": module, "error": f"no csg_report.json under {conv_dir}"})
            continue
        print(f"  reading {module} ...", file=sys.stderr)
        module_rows, summary = read_module(module, source_dir, conv_dir, with_sources, refresh)
        rows.extend(module_rows)
        summaries.append(summary)
    return rows, summaries


# ------------------------------------------------------------------------------------------
# the tables
# ------------------------------------------------------------------------------------------

TIERS = ["csg", "flatcsg", "surface", "mesh"]

# A "family" is a volume name without its trailing _<digits> groups; grouping is presentation only.
def family(name):
    return re.sub(r"(_\d+)+$", "", name or "")

TIER_LABEL = {"csg": "CSG", "flatcsg": "FlatCSG", "surface": "Surface", "mesh": "Tessellated"}


def redundancy(summaries):
    """Per module: the solid volumes that duplicate another (same family, class, capacity)."""
    out = []
    for s in summaries:
        rows = [v for v in (s.get("writerVolumes") or []) if not v.get("isAssembly")]
        if not rows:
            continue
        families = defaultdict(list)
        for v in rows:
            families[family(v.get("name"))].append(v)
        signatures = 0
        worst = []
        for name, members in families.items():
            sig = {(m.get("shapeClass"),
                    None if m.get("capacity_cm3") is None else round(m["capacity_cm3"], 10))
                   for m in members}
            signatures += len(sig)
            if len(members) > len(sig):
                worst.append((len(members), len(sig), name, members[0].get("shapeClass")))
        worst.sort(reverse=True)
        out.append({"module": s["module"], "volumes": len(rows), "families": len(families),
                    "signatures": signatures, "redundant": len(rows) - signatures,
                    "worst": worst[:6]})
    return out


def feature_matrix(rows):
    """Every distinct source shape class against what the round trip made of it.

    A `TGeoCompositeShape` is counted once as itself and once per leaf class, in a second table.
    """
    direct = defaultdict(lambda: {"parts": 0, **{t: 0 for t in TIERS},
                                  "knownSourceFailed": 0, "exact": 0})
    inside = Counter()
    inside_parts = defaultdict(set)
    for row in rows:
        cls = row["sourceClass"] or "(source shape unknown)"
        entry = direct[cls]
        entry["parts"] += 1
        if row["ships"] in entry:
            entry[row["ships"]] += 1
        if row["knownSourceFailures"]:
            entry["knownSourceFailed"] += 1
        if row["tessellationExact"]:
            entry["exact"] += 1
        for leaf, count in (row["leafClasses"] or {}).items():
            if cls == "TGeoCompositeShape":
                inside[leaf] += count
                inside_parts[leaf].add((row["module"], row["part"]))
    return direct, inside, inside_parts


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0:
            return "0"
        return f"{value:.{digits}g}"
    return str(value)


def md_table(header, rows, align=None):
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def render_markdown(rows, summaries, per_part=True):
    total = len(rows)
    doc = []
    doc.append("# TGeo &rarr; STEP &rarr; TGeo: what the round trip does to this geometry\n")
    doc.append(
        "Every number below is read from an instrument that produced it: `csg_report.json` for "
        "the cascade's own decision and evidence, `knownsource.json` for the only test that sees "
        "the original `TGeoVolume`, and the sidecars themselves for the face census. Nothing is "
        "re-derived here, and a field that could not be read is `-`, never `0`.\n")

    # --- corpus summary ---------------------------------------------------------------------
    doc.append("## The corpus\n")
    table = []
    tot = Counter()
    for s in summaries:
        if s.get("error"):
            table.append([s["module"], "-", "-", "-", "-", "-", "-", s["error"]])
            continue
        t = s["tiers"]
        tree = t.get("csg", 0) - s["flat"]
        table.append([s["module"], s["leafSolids"], tree, s["flat"], t.get("surface", 0),
                      t.get("mesh", 0), fmt(s["tessellationExact"]),
                      f"{s['knownSourceScored'] - s['knownSourceFailed']}/{s['knownSourceScored']}"])
        tot["leaf"] += s["leafSolids"] or 0
        tot["tree"] += tree
        tot["flat"] += s["flat"]
        tot["surface"] += t.get("surface", 0)
        tot["mesh"] += t.get("mesh", 0)
        tot["exact"] += s["tessellationExact"] or 0
        tot["ks"] += s["knownSourceScored"]
        tot["ksfail"] += s["knownSourceFailed"]
    table.append(["**all**", f"**{tot['leaf']}**", f"**{tot['tree']}**", f"**{tot['flat']}**",
                  f"**{tot['surface']}**", f"**{tot['mesh']}**", f"**{tot['exact']}**",
                  f"**{tot['ks'] - tot['ksfail']}/{tot['ks']}**"])
    doc.append(md_table(
        ["module", "leaf solids", "CSG tree", "FlatCSG", "Surface", "Tessellated",
         "tessellation exact", "agrees with source"],
        table, ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    doc.append("")
    doc.append(
        "*tessellation exact* counts parts whose every face is a planar polygon, for which a "
        "triangulation is the same solid rather than an approximation of it (`cadsupport/planar.py`). It "
        "is an annotation, not a routing rule -- a box is recognised as a `TGeoBBox` and stays "
        "one. *agrees with source* is `checkKnownSource.py`: class, capacity and a seeded "
        "containment cross-check against the original `TGeoShape`.\n")

    # --- writer coverage --------------------------------------------------------------------
    writer_classes = defaultdict(lambda: {"converted": 0, "declined": 0, "pureAssembly": 0,
                                          "reasons": Counter()})
    declined_rows = []
    for s in summaries:
        for cls, e in (s.get("writerByShapeClass") or {}).items():
            entry = writer_classes[cls]
            entry["converted"] += e.get("converted", 0)
            entry["declined"] += e.get("declined", 0)
            entry["pureAssembly"] += e.get("pureAssembly", 0)
            for reason, n in (e.get("reasons") or {}).items():
                entry["reasons"][reason] += n
        for row in s.get("writerDeclinedRows") or []:
            declined_rows.append((s["module"], row))
    if writer_classes:
        total_declined = sum(e["declined"] for e in writer_classes.values())
        doc.append("## Step 1, the writer: what reached the STEP file at all\n")
        doc.append(
            "A volume the writer declines never reaches the converter, so the tiers above are a "
            "fraction of what got *out*, not of the geometry. This is the other half. It counts "
            "**volumes**, where the tables above count leaf solids, and the two denominators are "
            "not the same number -- a volume with daughters contributes a `<name>__body` solid.\n")
        table = []
        for cls, e in sorted(writer_classes.items(), key=lambda kv: -kv[1]["converted"]):
            reasons = "; ".join(f"{r} ({n})" for r, n in e["reasons"].most_common(3))
            table.append([f"`{cls}`", e["converted"], e["declined"] or "",
                          e["pureAssembly"] or "", reasons or ""])
        doc.append(md_table(["source shape", "written", "declined", "pure assembly", "why declined"],
                            table, ["---", "---:", "---:", "---:", "---"]))
        doc.append("")
        doc.append(f"**{total_declined} volume(s) declined by the writer** across this corpus."
                   + (" Every solid-carrying volume was exported." if not total_declined else "")
                   + "\n")
        if declined_rows:
            table = [[m, r.get("name"), f"`{r.get('shapeClass')}`", r.get("reason") or "-"]
                     for m, r in declined_rows[:60]]
            doc.append(md_table(["module", "volume", "shape", "reason"], table))
            doc.append("")

    # --- repeated logical volumes -----------------------------------------------------------
    red = redundancy(summaries)
    heavy = [r for r in red if r["redundant"]]
    if heavy:
        doc.append("## Repeated logical volumes: the same solid, built many times\n")
        doc.append(
            "Some detector geometries give every *instance* of a component its own `TGeoVolume` "
            "and its own `TGeoShape`, where one volume placed many times would do. It is not a "
            "naming artefact: MFT holds **5144 separate `TGeoVolume` objects with 5144 separate "
            "`TGeoBBox` objects** whose parameters are one and the same "
            "`(0.05, 0.025, 0.025)` box, in one medium. A geometry that does this pays for it "
            "everywhere downstream -- the manager's volume table and voxelisation, the STEP file, "
            "this pipeline's per-part acceptance test, and navigation at run time.\n")
        doc.append(
            "*volumes* counts solid-carrying volumes; *distinct solids* counts distinct "
            "(family, shape class, capacity) signatures. The difference is what could be shared.\n")
        table = []
        for r in red:
            worst = "; ".join(f"`{n}` {v}&rarr;{s}" for v, s, n, _ in r["worst"][:3])
            table.append([r["module"], r["volumes"], r["families"], r["signatures"],
                          r["redundant"] or "", worst])
        doc.append(md_table(
            ["module", "volumes", "name families", "distinct solids", "redundant", "worst families"],
            table, ["---", "---:", "---:", "---:", "---:", "---"]))
        doc.append("")

    # --- feature matrix ---------------------------------------------------------------------
    direct, inside, inside_parts = feature_matrix(rows)
    doc.append("## Step 2, the converter: every source shape class, and what became of it\n")
    doc.append(
        "This is the table that says what the pipeline supports, measured over a real geometry "
        "rather than asserted. One row per distinct `TGeoShape` class in the source, and what the "
        "round trip emitted for the parts that used it.\n")
    table = []
    for cls, e in sorted(direct.items(), key=lambda kv: -kv[1]["parts"]):
        table.append([f"`{cls}`", e["parts"], e["csg"], e["flatcsg"], e["surface"], e["mesh"],
                      e["exact"], e["knownSourceFailed"] or ""])
    doc.append(md_table(
        ["source shape", "parts", "CSG tree", "FlatCSG", "Surface", "Tessellated",
         "tess. exact", "source disagreements"],
        table, ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    doc.append("")
    if inside:
        doc.append("### Primitive classes appearing *inside* a `TGeoCompositeShape`\n")
        doc.append(
            "Supporting a boolean means supporting what is in it. This counts leaf occurrences, "
            "and the parts they occur in.\n")
        table = [[f"`{cls}`", n, len(inside_parts[cls])]
                 for cls, n in inside.most_common()]
        doc.append(md_table(["leaf class", "occurrences", "parts"], table,
                            ["---", "---:", "---:"]))
        doc.append("")

    # --- what declined ----------------------------------------------------------------------
    declined = [r for r in rows if r["ships"] in ("surface", "mesh")]
    doc.append(f"## What did not become CSG ({len(declined)} of {total})\n")
    if declined:
        reasons = Counter((r["whyNotCSG"] or "(no reason recorded)").split(";")[0].strip()
                          for r in declined)
        doc.append(md_table(["the recogniser's reason", "parts"],
                            [[r, n] for r, n in reasons.most_common(25)], ["---", "---:"]))
        doc.append("")
    else:
        doc.append("Nothing. Every leaf solid in this corpus round-tripped as native CSG.\n")

    # --- per-part ---------------------------------------------------------------------------
    if per_part:
        doc.append("## Every part\n")
        doc.append(
            "One row per leaf solid, folded per module: a whole geometry is several thousand of "
            "them and an open list that long is not a document anyone reads.\n")
        for module in dict.fromkeys(r["module"] for r in rows):
            mrows = [r for r in rows if r["module"] == module]
            flat = sum(1 for r in mrows if r["ships"] == "flatcsg")
            exact = sum(1 for r in mrows if r["tessellationExact"])
            doc.append(f"<details><summary><strong>{module}</strong> &mdash; {len(mrows)} parts, "
                       f"{flat} FlatCSG, {exact} with an exact tessellation</summary>\n")
            # Rows that say the same thing about the same family are one row with a count.
            groups = {}
            for r in mrows:
                source = f"`{r['sourceClass'] or '?'}`"
                if r["sourceClass"] == "TGeoCompositeShape":
                    source += f" d{fmt(r['booleanDepth'])}/{fmt(r['leaves'])}l"
                faces = ", ".join(f"{v} {k}" for k, v in
                                  sorted((r["surfaceCensus"] or {}).items(), key=lambda kv: -kv[1]))
                key = (family(r["volume"] or r["part"]), source,
                       TIER_LABEL.get(r["ships"], r["ships"] or "-"), r["recogniser"] or "-",
                       "yes" if r["tessellationExact"] else
                       ("no" if r["tessellationExact"] is False else "-"),
                       faces or "-",
                       "FAIL" if r["knownSourceFailures"] else
                       ("ok" if r["containsPoints"] else "-"))
                entry = groups.setdefault(key, {"n": 0, "example": r["volume"] or r["part"],
                                                "dv": r["dVsym"]})
                entry["n"] += 1
                if r["dVsym"] is not None and (entry["dv"] is None or r["dVsym"] > entry["dv"]):
                    entry["dv"] = r["dVsym"]
            table = []
            for key, entry in sorted(groups.items(), key=lambda kv: (-kv[1]["n"], kv[0][0])):
                name = f"`{key[0]}`" + (f" &times;{entry['n']}" if entry["n"] > 1 else "")
                table.append([name, key[1], key[2], key[3], fmt(entry["dv"]),
                              key[4], key[5], key[6]])
            collapsed = len(mrows) - len(table)
            if collapsed:
                doc.append(f"*{len(table)} rows for {len(mrows)} parts; {collapsed} that said the "
                           "same thing about the same name family are folded into a &times;count. "
                           "`dV_sym` is the worst in the group.*\n")
            doc.append(md_table(
                ["part family", "source shape", "ships", "recogniser", "worst dV_sym cm^3",
                 "tess. exact", "faces", "vs source"],
                table, ["---", "---", "---", "---", "---:", "---:", "---", "---:"]))
            doc.append("\n</details>\n")
    return "\n".join(doc)


def render_part(rows, name):
    """One part's full record, for `--part`."""
    matches = [r for r in rows if name in (r["part"], r["volume"])]
    if not matches:
        return f"No part named {name!r} in this corpus.\n"
    out = []
    for r in matches:
        out.append(f"# {r['volume']}  ({r['module']})\n")
        pairs = [
            ("artefact stem", r["part"]),
            ("source volume", r["sourceVolume"]),
            ("source shape", r["sourceClass"]),
            ("boolean depth / leaves", None if r["booleanDepth"] is None
                else f"{r['booleanDepth']} / {r['leaves']}"),
            ("leaf classes", ", ".join(f"{v} {k}" for k, v in (r["leafClasses"] or {}).items()) or None),
            ("ships as", TIER_LABEL.get(r["ships"], r["ships"])),
            ("recogniser", r["recogniser"]),
            ("dV_sym / band", None if r["dVsym"] is None
                else f"{fmt(r['dVsym'])} / {fmt(r['band'])} cm^3"),
            ("declined CSG because", r["whyNotCSG"]),
            ("faces", ", ".join(f"{v} {k}" for k, v in
                                sorted((r["surfaceCensus"] or {}).items(), key=lambda kv: -kv[1])) or None),
            ("tessellation", None if r["tessellationExact"] is None else
                ("EXACT -- " + (r["tessellationExactWhy"] or "") if r["tessellationExact"]
                 else "an approximation -- " + (r["tessellationExactWhy"] or ""))),
            ("sidecar bytes", r["sidecarBytes"]),
            ("flat sidecar bytes", r["flatSidecarBytes"]),
            ("facet bytes", r["facetBytes"]),
            ("agrees with source", None if not r["containsPoints"] else
                (f"FAILED: {'; '.join(r['knownSourceFailures'])}" if r["knownSourceFailures"]
                 else f"{r['containsPoints']} points, {r['containsMismatches']} disagreement(s), "
                      f"capacity {fmt(r['capacityRelativeDeviation'])} relative")),
        ]
        out.append(md_table(["", ""], [[k, fmt(v)] for k, v in pairs if v is not None]))
        out.append("")
    return "\n".join(out)


HTML_HEAD = """<meta charset="utf-8"><title>TGeo &rarr; STEP &rarr; TGeo report</title>
<style>
 body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1200px;margin:2rem auto;
      padding:0 1.5rem;color:#1b1f24;background:#fff}
 h1{font-size:1.7rem;border-bottom:2px solid #d8dee4;padding-bottom:.3rem}
 h2{font-size:1.3rem;margin-top:2.2rem;border-bottom:1px solid #e6eaef;padding-bottom:.2rem}
 h3{font-size:1.05rem;margin-top:1.6rem;color:#39434f}
 table{border-collapse:collapse;margin:.8rem 0 1.4rem;font-size:12.5px;width:100%}
 th,td{border:1px solid #dde3ea;padding:3px 7px;text-align:left;vertical-align:top}
 th{background:#f2f5f8;font-weight:600}
 tr:nth-child(even) td{background:#fafbfc}
 code{background:#f2f5f8;padding:1px 4px;border-radius:3px;font-size:12px}
 em{color:#5a6673}
 details{margin:.6rem 0}summary{cursor:pointer;padding:.35rem .5rem;background:#f2f5f8;
   border:1px solid #dde3ea;border-radius:4px}summary:hover{background:#e9eef4}
 /* Printing is how this becomes a PDF, and a collapsed section must not vanish from it. */
 @media print{body{max-width:none;margin:0}h2{page-break-after:avoid}table{page-break-inside:auto}
   details{display:block}details>*{display:revert}summary{display:none}}
</style>
"""


def markdown_to_html(text):
    """Just enough markdown for this document: headings, tables, code spans, paragraphs."""
    import re
    lines = text.split("\n")
    out, table = [], []

    def flush_table():
        if not table:
            return
        head = [c.strip() for c in table[0].strip("|").split("|")]
        body = [[c.strip() for c in r.strip("|").split("|")] for r in table[2:]]
        out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head)
                   + "</tr></thead><tbody>")
        for row in body:
            out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table>")
        table.clear()

    def inline(s):
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
        return s.replace("&amp;rarr;", "&rarr;")

    for line in lines:
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if line.startswith("<details") or line.startswith("</details"):
            out.append(line)                       # raw HTML the markdown already carries
        elif line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush_table()
    return HTML_HEAD + "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="root holding one directory per module")
    ap.add_argument("--converted-root", help="conversions live here instead of <corpus>/<MOD>/conv")
    ap.add_argument("--subdir", default="conv", help="conversion subdirectory (default: conv)")
    ap.add_argument("--modules", help="comma-separated module names (default: all)")
    ap.add_argument("--part", help="report on this part only, and print it")
    ap.add_argument("--out", help="write the document here (default: stdout)")
    ap.add_argument("--html", action="store_true", help="emit HTML instead of markdown")
    ap.add_argument("--json", help="also write the joined per-part records here")
    ap.add_argument("--no-source-shapes", action="store_true",
                    help="skip the source-shape description (no ROOT, much faster)")
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the cached original_report.json")
    ap.add_argument("--no-per-part", action="store_true", help="summary and matrix only")
    args = ap.parse_args()

    modules = [m for m in (args.modules or "").split(",") if m] or None
    rows, summaries = read_corpus(args.corpus, args.converted_root, args.subdir, modules,
                                  with_sources=not args.no_source_shapes, refresh=args.refresh)
    if not rows:
        print("no parts found; is --corpus right, and has anything been converted?", file=sys.stderr)
        return 1

    text = (render_part(rows, args.part) if args.part
            else render_markdown(rows, summaries, per_part=not args.no_per_part))
    if args.html:
        text = markdown_to_html(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(rows)} parts, {len(summaries)} module(s))", file=sys.stderr)
    else:
        print(text)
    if args.json:
        Path(args.json).write_text(json.dumps({"parts": rows, "modules": summaries}, indent=1))
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
