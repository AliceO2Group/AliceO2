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

"""One module through the round trip, anchored where the source geometry put it.

  roundtrip_module.py <studydir> <MODULE> [csg,mesh]

Each module is converted per hall anchor:

  * everything under `barrel` becomes one conversion with `--top barrel`, the hall volume
    hollowed, placed back into the real `barrel` with the identity;
  * anything under `cave` or `caveRB24` is converted from its own subtree root
    and placed with that root's own matrix.

Writes <studydir>/cad/<MODULE>/ and a module_entries.json fragment that
make_configs.py assembles into the o2-sim external-geometry file.
"""

import json
import math
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cadsupport_path  # noqa: E402,F401  (puts ../tools on sys.path)
from cadsupport import occ_env  # noqa: E402

HALL = ("cave", "barrel", "caveRB24")


def sh(cmd, cwd, log):
    """Run one step in its own shell, with its output kept in a log."""
    with open(os.path.join(cwd, log), "w") as fh:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd, stdout=fh,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(open(os.path.join(cwd, log)).read()[-3000:])
        raise SystemExit(f"step failed ({r.returncode}): {cmd[:120]}")


def euler_deg(rot, env_o2):
    """The rotation_deg triple ExternalModule's JSON wants, verified by rebuilding it in ROOT."""
    if all(abs(rot[i] - (1.0 if i in (0, 4, 8) else 0.0)) < 1e-12 for i in range(9)):
        return None                       # identity: omit the rotation entirely
    # ROOT lives in the o2 environment, so the candidate is rebuilt there.
    probe = f"""
import ROOT, json, itertools, sys
target = {list(rot)!r}
for cand in itertools.product((0,90,-90,180),repeat=3):
    c = ROOT.TGeoCombiTrans()
    c.RotateX(cand[0]); c.RotateY(cand[1]); c.RotateZ(cand[2])
    m = c.GetRotationMatrix()
    if all(abs(m[i]-target[i]) < 1e-9 for i in range(9)):
        print(json.dumps(list(cand))); sys.exit(0)
sys.exit(3)
"""
    r = subprocess.run(["bash", "-c", f'{env_o2}; python3 -c {shlex.quote(probe)}'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot express this rotation as rotation_deg: {rot}\n"
                         "ExternalModule's JSON carries Euler angles only; this "
                         "placement needs a full matrix and the loader does not "
                         "take one yet.")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    study, mod = os.path.abspath(sys.argv[1]), sys.argv[2]
    d = os.path.join(study, "cad", mod)
    os.makedirs(d, exist_ok=True)
    ct = os.path.dirname(os.path.abspath(__file__))
    env_o2 = f'source "{study}/env_o2.sh" >/dev/null 2>&1'
    env_cv = f'{env_o2}; source "{study}/env_converter.sh"'
    occ_python = occ_env.occ_python()
    if occ_python is None:
        raise SystemExit(occ_env.UNRESOLVED)
    py = shlex.quote(str(occ_python))

    print(f"=== {mod}: the source geometry")
    sh(f'{env_o2}; o2-sim-serial -n 0 -g boxgen -m {mod} -o o2sim', d, "geom.log")

    print(f"=== {mod}: where does it hang itself?")
    sh(f'{env_o2}; python3 "{ct}/module_anchors.py" o2sim_geometry.root '
       f'--json anchors.json', d, "anchors.log")
    roots = json.load(open(os.path.join(d, "anchors.json")))["roots"]
    in_barrel = [r for r in roots if r["anchor"] == "barrel"]
    elsewhere = [r for r in roots if r["anchor"] != "barrel"]
    print(f"    {len(in_barrel)} subtree(s) under barrel, "
          f"{len(elsewhere)} elsewhere: "
          f"{[(r['volume'], r['anchor']) for r in elsewhere]}")

    entries = {}
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["csg", "mesh"]

    def convert(tag, top, hollow, anchor, placement, variant="csg"):
        """One --top conversion plus its media sidecar, scored.

        `variant` "csg" is the shipped cascade; "mesh" is tessellated-only, as a benchmark.
        """
        out = f"conv_{tag}" if variant == "csg" else f"conv_{variant}_{tag}"
        cascade = ("--csg auto --exact-surfaces auto --mesh" if variant == "csg"
                   else "--mesh")
        hollow_args = " ".join(f"--hollow-volume {h}" for h in hollow)
        tagarg = f'--hollow-tag {mod}' if hollow else ""
        print(f"=== {mod}: --top {top} -> anchor {anchor}")
        # No --carve-mothers: the converter restores the nesting from the sidecar, and carving
        # cannot subtract an assembly daughter.
        sh(f'{env_cv}; {py} "{ct}/../../tools/O2_TGeoToCAD.py" o2sim_geometry.root {tag}.step '
           f'--top {top} --report {tag}_writer_report.json '
           f'--media-json {tag}_media.json {hollow_args} {tagarg}',
           d, f"writer_{tag}.log")
        sh(f'{env_cv}; {py} "{ct}/../../tools/O2_CADtoTGeo.py" {tag}.step -o geom.C '
           f'--output-folder {out} {cascade} '
           f'--media-json {tag}_media.json', d, f"{out}.log")
        for line in open(os.path.join(d, f"{out}.log")):
            if "tiers:" in line or "Media from sidecar" in line or "[WARN]" in line:
                print("   ", line.rstrip())
        sh(f'{env_o2}; python3 "{ct}/check_media.py" --original o2sim_geometry.root '
           f'--macro {out}/geom.C --rtol 1e-6 '
           f'--writer-report {tag}_writer_report.json --json media_{out}.json',
           d, f"media_{out}.log")
        for line in open(os.path.join(d, f"media_{out}.log")):
            if line.startswith(("VERDICT", "  media identical", "  left on")):
                print("   ", line.rstrip())
        e = {"tag": tag, "macro": os.path.join(d, out, "geom.C"), "anchor": anchor}
        if placement:
            e["placement"] = placement
        entries.setdefault(variant, []).append(e)

    # The STEP is written once per anchor; only the back-conversion differs per variant.
    for v in variants:
        if in_barrel:
            convert("barrel", "barrel", ["barrel"], "barrel", None, v)
        for r in elsewhere:
            rot = euler_deg(r["rotation"], env_o2)
            pl = {"translation": [float(x) for x in r["translation"]]}
            if rot:
                pl["rotation_deg"] = rot
            convert(r["volume"], r["volume"], [], r["anchor"], pl, v)

    with open(os.path.join(d, "module_entries.json"), "w") as fh:
        json.dump({"module": mod, "entries": entries}, fh, indent=2)
    print(f"=== {mod}: " + ", ".join(f"{len(v)} {k} placement(s)"
                                     for k, v in entries.items())
          + f" -> {d}/module_entries.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
