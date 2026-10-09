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

"""Diff two gate.json reports column by column, with the scale law of each column applied.

Every real field carries a length exponent in `_FIELD_EXPONENT`; the expectation is scaled by
`factor ** exponent` and the residual is reported. Integer columns are compared for equality, real
columns within a stated relative band, a floor on double arithmetic rather than a tolerance on the
geometry. Fields absent from gate.json and points the harness never sampled are out of reach.

Usage
-----
  compareGateRuns.py --baseline base/gate.json --candidate z400/gate.json --label "z+400 cm"
  compareGateRuns.py --baseline base/gate.json --candidate x10/gate.json  --scale 10
  compareGateRuns.py --baseline base/gate.json --self-test
"""

import argparse
import copy
import json
import sys
from pathlib import Path

# Timing and point-derived checksums are not compared; `Seconds` is matched as a substring.
_IGNORED_SUBSTRINGS = ("Seconds", "nsPerCall", "checksum")
_IGNORED_KEYS = {"id", "model", "worstOffenders", "rimDetail",
                 "timingCandidate", "timingReference", "timingCandidateLoop", "timingPruned",
                 "timingUnpruned"}

# The mesh columns move under a scaling on purpose; --gate-columns-only excludes them.
_MESH_DERIVED_PREFIXES = ("contains.", "distout.", "distin.", "safety.")
_MESH_DERIVED_KEYS = {"nTriangles"}

# The exponent of the length scale factor each real-valued field carries. Anything not listed is
# treated as dimensionless (exponent 0) -- counts, fractions, relative deviations, booleans.
_FIELD_EXPONENT = {
    # lengths, cm
    "maxRimIsolation": 1,
    "rimChordResolution": 1,
    "rimMatchTolerance": 1,
    "totalRimLength": 1,
    "unmatchedRimLength": 1,
    "maxSharedEdgeDeviation": 1,
    "worstDeviation": 1,
    "tolerance": 1,
    # volumes, cm^3
    "capacity": 3,
    "capacityCandidate": 3,
}

# Fields whose value is a physical measurement and must agree only to within double arithmetic
# across two independently converted shapes; everything else is required to be equal.
_REAL_BAND = 1.0e-9


def flatten(node, prefix=""):
    """Depth-first flatten of a part's report into {dotted path: scalar}."""
    flat = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _IGNORED_KEYS or any(s in key for s in _IGNORED_SUBSTRINGS):
                continue
            flat.update(flatten(value, f"{prefix}{key}."))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            flat.update(flatten(value, f"{prefix}{i}."))
    else:
        flat[prefix.rstrip(".")] = node
    return flat


def exponent_of(path: str) -> int:
    return _FIELD_EXPONENT.get(path.rsplit(".", 1)[-1], 0)


def is_mesh_derived(path: str) -> bool:
    return (path.startswith(_MESH_DERIVED_PREFIXES) or
            path.rsplit(".", 1)[-1] in _MESH_DERIVED_KEYS)


def compare_part(base: dict, cand: dict, factor: float, gate_only: bool = False):
    """Return (list of differences, number of fields compared)."""
    flat_base = flatten(base)
    flat_cand = flatten(cand)
    if gate_only:
        flat_base = {k: v for k, v in flat_base.items() if not is_mesh_derived(k)}
        flat_cand = {k: v for k, v in flat_cand.items() if not is_mesh_derived(k)}
    differences = []
    for path in sorted(set(flat_base) | set(flat_cand)):
        if path not in flat_base:
            differences.append((path, "<absent>", flat_cand[path], "field only in candidate"))
            continue
        if path not in flat_cand:
            differences.append((path, flat_base[path], "<absent>", "field only in baseline"))
            continue
        b, c = flat_base[path], flat_cand[path]
        if isinstance(b, bool) or isinstance(c, bool) or isinstance(b, str) or isinstance(c, str):
            if b != c:
                differences.append((path, b, c, "differs"))
            continue
        if isinstance(b, int) and isinstance(c, int):
            if b != c:
                differences.append((path, b, c, f"integer differs by {c - b:+d}"))
            continue
        if b is None or c is None:
            # A null leaf is a real value (a non-comparable capacity): null -> number is a change.
            if b != c:
                differences.append((path, b, c, "differs"))
            continue
        expected = b * (factor ** exponent_of(path))
        if expected == c:
            continue
        scale = max(abs(expected), abs(c))
        residual = abs(c - expected) / scale if scale else abs(c - expected)
        if residual > _REAL_BAND:
            differences.append((path, expected, c,
                                f"relative residual {residual:.3g} > {_REAL_BAND:g} "
                                f"(scale law: factor^{exponent_of(path)})"))
    return differences, len(set(flat_base) | set(flat_cand))


def key_reports(baseline, candidate):
    """Pair the two reports' parts up, and say how.

    By full part id, falling back to the leading component only when the ids do not match; every
    part of one CAD model shares that component.
    """
    by_id = ({p["id"]: p for p in baseline}, {p["id"]: p for p in candidate})
    if set(by_id[0]) == set(by_id[1]):
        return by_id[0], by_id[1], "part id"
    by_stem = ({p["id"].split("/", 1)[0]: p for p in baseline},
               {p["id"].split("/", 1)[0]: p for p in candidate})
    collapsed = len(by_stem[0]) < len(baseline) or len(by_stem[1]) < len(candidate)
    if collapsed:
        return by_id[0], by_id[1], "part id (ids differ and the leading component is not unique)"
    return by_stem[0], by_stem[1], "leading id component"


def compare(baseline, candidate, factor, label, gate_only=False):
    base_by_key, cand_by_key, keying = key_reports(baseline, candidate)
    print(f"(paired by {keying})")
    print(f"=== {label} : {len(cand_by_key)} part(s) vs baseline's {len(base_by_key)}, "
          f"length factor {factor:g} ===")
    missing = sorted(set(base_by_key) - set(cand_by_key))
    extra = sorted(set(cand_by_key) - set(base_by_key))
    total_differences = 0
    for key in missing:
        print(f"  [MISSING]  {key}: in baseline, absent from candidate")
        total_differences += 1
    for key in extra:
        print(f"  [EXTRA]    {key}: in candidate, absent from baseline")
        total_differences += 1
    for key in sorted(set(base_by_key) & set(cand_by_key)):
        differences, n_fields = compare_part(base_by_key[key], cand_by_key[key], factor, gate_only)
        total_differences += len(differences)
        if not differences:
            print(f"  [same]     {key}: {n_fields} field(s) identical after the scale law")
            continue
        print(f"  [DIFFERS]  {key}: {len(differences)} of {n_fields} field(s)")
        for path, b, c, why in differences:
            print(f"               {path}: baseline {b!r} -> candidate {c!r}  ({why})")
    print(f"\n{total_differences} difference(s)")
    return total_differences


def _nudge(path):
    """A defect injector that multiplies a real field by 1 + 1e-8, on the first part where that
    field is not exactly zero.
    """
    def apply(report):
        for index, part in enumerate(report):
            node = part
            for key in path[:-1]:
                node = node[key]
            if node.get(path[-1]):
                node[path[-1]] = node[path[-1]] * (1. + 1.e-8)
                return index
        return None
    return apply


def self_test(baseline):
    """Prove the comparison can say "yes": four injected defects, one per code path."""
    print("=== self-test: can this comparison detect a violation? ===")

    def bump_int(report):
        column = report[0]["oracle"]["contains"]
        column["nMismatchUnexplained"] = column["nMismatchUnexplained"] + 1
        return 0

    def downgrade(report):
        report[0]["navigation"]["reliability"] = "openBoundary"
        return 0

    cases = [
        ("integer column (one extra unexplained oracle disagreement)", bump_int),
        ("length column, exponent 1 (maxRimIsolation +1e-8 relative)",
         _nudge(["navigation", "maxRimIsolation"])),
        ("volume column, exponent 3 (capacityCandidate +1e-8 relative)",
         _nudge(["oracle", "capacityCandidate"])),
        ("verdict string (navigation reliability downgraded)", downgrade),
    ]
    caught = 0
    for what, break_it in cases:
        broken = copy.deepcopy(baseline)
        try:
            index = break_it(broken)
        except (KeyError, IndexError) as exc:
            print(f"  [SKIP] {what}: not present in this report ({exc})")
            continue
        if index is None:
            print(f"  [MISSED] {what}: no part carries a non-zero value, nothing was injected")
            continue
        differences, _ = compare_part(baseline[index], broken[index], 1.0)
        ok = bool(differences)
        caught += ok
        print(f"  [{'caught' if ok else 'MISSED'}] {what}  [part {baseline[index]['id']}]")
        for path, b, c, why in differences:
            print(f"             {path}: {b!r} -> {c!r} ({why})")
    print(f"\n{caught}/{len(cases)} injected defect(s) caught")
    return caught == len(cases)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="uniform length factor applied to the candidate's geometry (1 for a pure "
                         "translation). Every real column is compared against baseline * "
                         "factor**exponent, with the exponent declared per field in this file.")
    ap.add_argument("--label", default=None)
    ap.add_argument("--gate-columns-only", action="store_true",
                    help="drop the columns that compare against the tessellated mesh, and the "
                         "triangle count. Under a scaling the mesh is deliberately not the same "
                         "mesh, so those columns move for a reason that is not the kernel's.")
    ap.add_argument("--self-test", action="store_true",
                    help="inject known defects into the baseline and report whether they are "
                         "caught; run this before believing any green comparison")
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text())
    if args.self_test:
        return 0 if self_test(baseline) else 1
    if args.candidate is None:
        ap.error("--candidate is required unless --self-test is given")
    candidate = json.loads(args.candidate.read_text())
    label = args.label or f"{args.baseline} vs {args.candidate}"
    return 0 if compare(baseline, candidate, args.scale, label,
                        args.gate_columns_only) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
