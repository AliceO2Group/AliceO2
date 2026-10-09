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

"""Turn the integration-demo run directories into the comparison tables.

Reads each <conv-root>/runs/<tag>/sim.log and reports, per run:
  geometry+engine init time, transport time, peak RSS, total steps, secondaries,
  tracks transported, sensitive steps and hits per external detector, and the
  per-volume step tally that MCStepLogger prints.

Then pairs every <name>_exact with its <name>_tess and prints the differences.

Usage: summarise_runs.py <conv-root> [--per-volume]
"""
import os
import re
import sys

RE_INIT = re.compile(r"Init: Real time ([\d.]+) s, CPU time ([\d.]+)")
RE_TOOK = re.compile(r"Simulation process took ([\d.]+) s")
RE_REAL = re.compile(r"\[INFO\] Real time ([\d.]+) s, CPU time ([\d.]+)s")
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_STEPS = re.compile(r"\[STEPLOGGER\]: did (\d+) steps")
RE_TRACKS = re.compile(r"\[STEPLOGGER\]: transported (\d+) different tracks")
RE_VOL = re.compile(r"\[STEPLOGGER\]: VolName (\S+) COUNT (\d+) SECONDARIES (\d+)")
RE_EOE = re.compile(r"External detector (\S+) EndOfEvent: (\d+) sensitive step\(s\) -> (\d+) hit\(s\)")
RE_BAD = re.compile(r"stuck|Stuck|ABORT|abort|FATAL|not reachable|Navigation", re.I)


def parse(logpath):
    r = {"vol": {}, "sec": {}, "sens": {}, "hits": {}, "bad": [], "real": []}
    with open(logpath, errors="ignore") as f:
        for line in f:
            m = RE_INIT.search(line)
            if m:
                r["init_real"], r["init_cpu"] = float(m.group(1)), float(m.group(2))
            m = RE_TOOK.search(line)
            if m:
                r["total"] = float(m.group(1))
            m = RE_REAL.search(line)
            if m:
                r["real"].append((float(m.group(1)), float(m.group(2))))
            m = RE_RSS.search(line)
            if m:
                r["rss_mb"] = int(m.group(1)) / 1024.0
            m = RE_STEPS.search(line)
            if m:
                # MCStepLogger flushes once per event: sum, do not overwrite
                r["steps"] = r.get("steps", 0) + int(m.group(1))
            m = RE_TRACKS.search(line)
            if m:
                r["tracks"] = r.get("tracks", 0) + int(m.group(1))
            m = RE_VOL.search(line)
            if m:
                r["vol"][m.group(1)] = int(m.group(2))
                r["sec"][m.group(1)] = int(m.group(3))
            m = RE_EOE.search(line)
            if m:
                r["sens"][m.group(1)] = r["sens"].get(m.group(1), 0) + int(m.group(2))
                r["hits"][m.group(1)] = r["hits"].get(m.group(1), 0) + int(m.group(3))
            if RE_BAD.search(line) and "TG4RootNavigator" not in line:
                r["bad"].append(line.strip()[:160])
    # the transport timing is the last "Real time" line the application prints
    if r["real"]:
        r["transport_real"], r["transport_cpu"] = r["real"][-1]
    r["secondaries"] = sum(r["sec"].values())
    return r


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = os.path.join(sys.argv[1], "runs")
    per_volume = "--per-volume" in sys.argv
    runs = {}
    analysis = os.path.join(sys.argv[1], "analysis")
    for tag in sorted(os.listdir(root)):
        log = os.path.join(root, tag, "sim.log")
        if not os.path.exists(log):
            continue
        r = parse(log)
        # With LOG_TTREE set, the step counts come from the analyse_all.sh reduction.
        af = os.path.join(analysis, f"steps_{tag}.txt")
        if os.path.exists(af):
            for line in open(af):
                f = line.split()
                if len(f) >= 2 and f[0] == "STEPS_TOTAL":
                    r["steps"] = int(f[1])
                elif len(f) >= 2 and f[0] == "SECONDARIES":
                    r["secondaries"] = int(f[1])
                elif len(f) >= 4 and f[0] == "VOL":
                    r["vol"][f[1]] = int(f[2])
        runs[tag] = r

    hdr = f"{'run':22s} {'init_s':>8s} {'transp_s':>9s} {'RSS_MB':>8s} {'steps':>9s} {'2nd':>8s} {'tracks':>7s} {'BAGR hits':>9s} {'bad':>4s}"
    print(hdr)
    print("-" * len(hdr))
    for tag, r in runs.items():
        print(f"{tag:22s} {r.get('init_real',0):8.2f} {r.get('transport_real',0):9.3f} "
              f"{r.get('rss_mb',0):8.1f} {r.get('steps',0):9d} {r.get('secondaries',0):8d} "
              f"{r.get('tracks',0):7d} {r['hits'].get('BAGR',0):9d} "
              f"{len(r['bad']):4d}")

    print()
    print("pairwise exact vs tessellated")
    for tag in sorted(runs):
        if not tag.endswith("_exact"):
            continue
        other = tag[:-6] + "_tess"
        if other not in runs:
            continue
        a, b = runs[tag], runs[other]
        name = tag[:-6]
        ds = a.get("steps", 0) - b.get("steps", 0)
        rel = 100.0 * ds / b["steps"] if b.get("steps") else 0.0
        print(f"  {name:12s} steps  exact={a.get('steps',0):8d} tess={b.get('steps',0):8d} "
              f"diff={ds:+7d} ({rel:+.2f}%)")
        print(f"  {'':12s} 2nd    exact={a.get('secondaries',0):8d} tess={b.get('secondaries',0):8d}")
        print(f"  {'':12s} transp exact={a.get('transport_real',0):8.3f}s tess={b.get('transport_real',0):8.3f}s "
              f"ratio={a.get('transport_real',0)/b['transport_real'] if b.get('transport_real') else 0:.2f}")
        print(f"  {'':12s} init   exact={a.get('init_real',0):8.2f}s tess={b.get('init_real',0):8.2f}s")
        print(f"  {'':12s} RSS    exact={a.get('rss_mb',0):8.1f}MB tess={b.get('rss_mb',0):8.1f}MB")
        print(f"  {'':12s} hits   BAGR {a['hits'].get('BAGR',0)} vs {b['hits'].get('BAGR',0)}")
        if per_volume:
            vols = sorted(set(a["vol"]) | set(b["vol"]),
                          key=lambda v: -(a["vol"].get(v, 0) + b["vol"].get(v, 0)))
            print(f"    {'volume':28s} {'exact':>8s} {'tess':>8s} {'diff':>8s}")
            for v in vols:
                x, y = a["vol"].get(v, 0), b["vol"].get(v, 0)
                if x == y == 0:
                    continue
                print(f"    {v:28s} {x:8d} {y:8d} {x-y:+8d}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
