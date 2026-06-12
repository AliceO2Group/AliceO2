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
"""AliHyperloop monitoring MCP server.

Exposes a small set of read-only tools to inspect ongoing Hyperloop train
runs, their resource consumption, and per-wagon breakdowns.  All data is
fetched on demand (no polling, no bulk scraping).

The server talks to the Hyperloop REST API through a local authenticating
proxy (ccdb_proxy.py) that handles GRID certificate auth.

Usage
-----
    python3 hyperloop_server.py [--proxy URL] [--token TOKEN]

Environment variables
    HYPERLOOP_PROXY   proxy base URL  (default: http://localhost:8888)
    HYPERLOOP_TOKEN   bearer token    (default: foo-baz)
"""

from __future__ import annotations

import asyncio
import collections
import datetime
import json
import os
import re
import sys
import time

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hyperloop")

PROXY = os.environ.get("HYPERLOOP_PROXY", "http://localhost:8888")
TOKEN = os.environ.get("HYPERLOOP_TOKEN", "foo-baz")
API = f"{PROXY}/alihyperloop-data"

# --- Write guardrails ---------------------------------------------------------
# Wagon-creating tools are HARD-LOCKED to this one analysis. The destination is a
# baked-in constant, never a caller argument, so the tools physically cannot touch
# any other analysis.
ALLOWED_ANALYSIS = 50446  # "O2 Development"
# Every created wagon is prefixed with this, so test wagons are easy to spot/clean.
WAGON_PREFIX = "Test"
# Writes are inert unless the server is explicitly started with this enabled.
ALLOW_WRITE = os.environ.get("HYPERLOOP_ALLOW_WRITE", "").strip().lower() in ("1", "true", "yes", "on")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


async def _get(path: str, params: dict | None = None) -> any:
    hdrs = _headers()
    hdrs["Accept-Encoding"] = "identity"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{API}/{path}", params=params, headers=hdrs)
        r.raise_for_status()
        return r.json()


async def _get_text(path: str, params: dict | None = None) -> str:
    """GET a JSP endpoint and return the raw response text. Used for mutating
    endpoints (e.g. clone-wagon) that don't always return JSON."""
    hdrs = _headers()
    hdrs["Accept-Encoding"] = "identity"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{API}/{path}", params=params, headers=hdrs)
        r.raise_for_status()
        return r.text


ALIMON = f"{PROXY}/alimonitor"
ALIMON_TOKEN = os.environ.get("HYPERLOOP_ALIMON_TOKEN", "jalien-secret")


async def _get_workdir_json(train_id: int, fname: str):
    """Fetch a file from a test's train-workdir (alimonitor route)."""
    b = f"{train_id // 10000:04d}"
    n = f"{train_id:08d}"
    url = f"{ALIMON}/train-workdir/tests/{b}/{n}/{fname}"
    hdrs = {"Authorization": f"Bearer {ALIMON_TOKEN}", "Accept-Encoding": "identity"}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(url, headers=hdrs)
        r.raise_for_status()
        return r.json()


def _tag_date(tag: str | None) -> str | None:
    """Extract the YYYYMMDD date from a package tag like '…daily-20260604-0400-1'."""
    m = re.search(r"(?:daily|nightly|epn)-?(\d{8})", (tag or "").lower())
    return m.group(1) if m else None


def _series_max(v) -> float | None:
    """Max value of a {timestamp,value} time series (or a scalar). Use for
    cumulative metrics like processed_size (max = final total)."""
    if isinstance(v, list) and v:
        try:
            return max(float(x["value"]) for x in v)
        except Exception:
            return None
    try:
        return float(v)
    except Exception:
        return None


def _series_sum(v) -> float:
    """Sum of a {timestamp,value} time series. `cpuUsedAbsolute` is per-interval
    CPU microseconds (O2 Monitoring ProcessMonitor: Δ getrusage utime+stime per
    sample), so the sum is the *total* CPU time of the run (µs; /1e6 = CPU-s)."""
    if isinstance(v, list):
        try:
            return sum(float(x["value"]) for x in v)
        except Exception:
            return 0.0
    return 0.0


def _fmt_bytes(n: float | None) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _parse_job_status(raw: str | None) -> dict:
    if not raw:
        return {}
    js = json.loads(raw) if isinstance(raw, str) else raw
    done = sum(v for k, v in js.items() if k.startswith("DONE"))
    total = js.get("TOTAL", 0)
    errors = sum(v for k, v in js.items()
                 if k.startswith("ERROR") or k.startswith("EXPIRED")
                 or k.startswith("FAILED") or k.startswith("KILLED"))
    active = sum(v for k, v in js.items()
                 if k.startswith("R") or k.startswith("A") or k.startswith("S"))
    wait = total - done - errors - active
    return {"total": total, "done": done, "errors": errors,
            "active": active, "wait": max(0, wait)}


def _format_train_table(trains: list[dict]) -> str:
    lines = []
    lines.append(f"{'ID':>8}  {'State':<11} {'Done/Total':>12} {'Err%':>5}  "
                 f"{'Dataset':<40} {'Package'}")
    lines.append("-" * 120)

    for t in trains:
        js = _parse_job_status(t.get("job_status"))
        total = js.get("total", 0)
        done = js.get("done", 0)
        errors = js.get("errors", 0)
        err_pct = f"{100 * errors / total:.1f}" if total > 0 else "n/a"
        pkg = (t.get("package_tag") or "").replace("O2Physics::", "")
        ds = t.get("dataset_name", "")
        if len(ds) > 40:
            ds = ds[:37] + "..."
        lines.append(
            f"{t['id']:>8}  {t.get('state', '?'):<11} "
            f"{done:>6}/{total:<6} {err_pct:>5}  "
            f"{ds:<40} {pkg}"
        )
    return "\n".join(lines)


@mcp.tool()
async def list_ongoing_trains() -> str:
    """List all currently running / ready Hyperloop train runs.

    Returns a compact table with train ID, dataset, state, job progress,
    error rate, and package tag.  One API call.
    """
    trains = await _get("trains/all-trains.jsp", {"state": "ready"})
    if not trains:
        return "No ongoing trains."

    trains.sort(key=lambda x: _parse_job_status(
        x.get("job_status")).get("total", 0), reverse=True)

    result = _format_train_table(trains)
    result += f"\n\nTotal: {len(trains)} trains"
    return result


@mcp.tool()
async def search_trains(dataset: str, last_n: int = 10) -> str:
    """Search for recent trains (including finished) on a given dataset.

    Uses the dataset name for server-side coarse filtering, then exact-matches
    client-side.  Returns the most recent `last_n` trains (by ID descending).

    Args:
        dataset: Exact dataset name (e.g. "LHC25ae_pass2_small").
        last_n:  Number of most recent trains to return (default 10).
    """
    raw = await _get("trains/all-trains.jsp", {"dataset_name": dataset})
    if not raw:
        return f"No trains found for dataset '{dataset}'."

    # Server returns fuzzy matches; exact-filter client-side
    exact = [t for t in raw if t.get("dataset_name") == dataset]
    if not exact:
        return f"No trains found with exact dataset name '{dataset}'."

    # Most recent first
    exact.sort(key=lambda t: t.get("id", 0), reverse=True)
    exact = exact[:last_n]

    result = _format_train_table(exact)
    result += f"\n\nShowing {len(exact)} most recent (of {len([t for t in raw if t.get('dataset_name') == dataset])} total)"
    return result


@mcp.tool()
async def train_detail(train_id: int) -> str:
    """Get resource metrics for a specific train run (ongoing or finished).

    Shows CPU time, wall time, memory (PSS), throughput, input/output
    sizes, target, and merge status.  One API call.
    """
    t = await _get("trains/train.jsp", {"train_id": train_id})

    lines = [f"Train {t['id']}: {t.get('dataset_name', '?')}"]
    lines.append(f"  State:       {t.get('state')}")
    lines.append(f"  Package:     {t.get('package_tag')}")
    lines.append(f"  Target:      {t.get('target')}")
    lines.append(f"  CPU cores:   {t.get('cpu_cores')}")
    lines.append(f"  CPU time:    {_fmt_time(t.get('cpu_time'))}")
    lines.append(f"  Wall time:   {_fmt_time(t.get('wall_time'))}")
    lines.append(f"  PSS memory:  {_fmt_bytes(t.get('mem_pss'))} avg, "
                 f"{_fmt_bytes(t.get('mem_pss_max'))} max")
    lines.append(f"  Private mem: {_fmt_bytes(t.get('mem_private'))} avg, "
                 f"{_fmt_bytes(t.get('mem_private_max'))} max")
    lines.append(f"  Input size:  {_fmt_bytes(t.get('input_size'))}")
    lines.append(f"  Output size: {_fmt_bytes(t.get('output_size'))}")

    throughput = t.get("estimated_throughput")
    if throughput:
        lines.append(f"  Throughput:  {_fmt_bytes(throughput)}/s")

    events = t.get("events")
    if events and events > 0:
        lines.append(f"  Events:      {events}")

    lines.append(f"  Created:     {t.get('created')}")
    lines.append(f"  Username:    {t.get('username')}")

    return "\n".join(lines)


@mcp.tool()
async def wagon_stats(train_id: int) -> str:
    """Get per-wagon CPU and memory breakdown for a train (ongoing or finished).

    Fetches wagon IDs from the train, then retrieves grid statistics
    for each wagon.  Typically 10-20 wagons, one API call each.
    """
    # First get train detail for dataset_id and wagons_timestamp
    t = await _get("trains/train.jsp", {"train_id": train_id})
    dataset_id = t.get("dataset_id")
    wagons_ts = t.get("wagons_timestamp") or t.get("dataset_timestamp")

    if not dataset_id or not wagons_ts:
        return f"Cannot determine dataset/timestamp for train {train_id}"

    # Get wagon IDs
    wagons_data = await _get("trains/wagons_derived_data.jsp",
                             {"train_id": train_id,
                              "wagons_timestamp": wagons_ts})
    wagon_ids = list(wagons_data.keys()) if isinstance(wagons_data, dict) else []
    if not wagon_ids:
        return f"No wagons found for train {train_id}"

    # Fetch stats for each wagon concurrently
    async def fetch_one(wid: str) -> dict | None:
        try:
            stats = await _get("analysis/wagon/wagon-dataset-grid-statistics.jsp",
                               {"wagon_id": wid, "dataset_id": dataset_id})
            if isinstance(stats, dict) and str(train_id) in stats:
                return stats[str(train_id)]
        except Exception:
            pass
        return None

    results = await asyncio.gather(*(fetch_one(wid) for wid in wagon_ids))

    rows = []
    for wid, stat in zip(wagon_ids, results):
        if stat is None:
            continue
        rows.append(stat)

    if not rows:
        return f"No wagon statistics available for train {train_id}"

    # Sort by CPU time descending
    rows.sort(key=lambda r: r.get("cpu_time") or 0, reverse=True)

    lines = [f"Wagon stats for train {train_id} "
             f"({t.get('dataset_name', '?')}), {len(rows)} wagons:\n"]
    lines.append(f"{'Wagon':<35} {'CPU time':>10} {'PSS avg':>10} "
                 f"{'PSS max':>10} {'Throughput':>12} {'Done%':>6}")
    lines.append("-" * 90)

    total_cpu = 0
    for r in rows:
        name = r.get("wagon_name", f"id={r.get('wagon_id', '?')}")
        if len(name) > 35:
            name = name[:32] + "..."
        cpu = r.get("cpu_time") or 0
        total_cpu += cpu
        pss_avg = _fmt_bytes(r.get("mem_pss"))
        pss_max = _fmt_bytes(r.get("mem_pss_max"))
        tp = _fmt_bytes(r.get("throughput")) + "/s" if r.get("throughput") else "n/a"
        pct = r.get("percent_done")
        pct_str = f"{pct}%" if pct is not None else "n/a"
        lines.append(f"{name:<35} {_fmt_time(cpu / 1000):>10} {pss_avg:>10} "
                     f"{pss_max:>10} {tp:>12} {pct_str:>6}")

    lines.append("-" * 90)
    lines.append(f"Total CPU: {_fmt_time(total_cpu / 1000)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis / wagon browsing
#
# These mirror the alihyperloop web UI's analysis pages.  Endpoint and param
# names were taken from the frontend bundle (/hyperloop/assets/index-*.js);
# unknown `lists` values silently return an empty array, and the wagon list
# uses the *plural* `analysis_ids`.
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_analyses(username: str) -> str:
    """List a user's Hyperloop analyses (id, name, JIRA, analyzers).

    `username` is the CERN login of an analyzer (e.g. "eulisse").
    """
    rows = await _get("analysis/list-analysis.jsp",
                      {"lists": "analysis-by-username", "username": username})
    if not isinstance(rows, list) or not rows:
        return f"No analyses found for user '{username}'."
    lines = [f"Analyses for {username}:\n",
             f"{'ID':>7}  {'Svc':<3} {'JIRA':<14} Name"]
    lines.append("-" * 70)
    for a in rows:
        svc = "yes" if a.get("service_analysis") else ""
        lines.append(f"{a.get('id'):>7}  {svc:<3} {str(a.get('jira_id') or ''):<14} "
                     f"{a.get('name')}")
    return "\n".join(lines)


@mcp.tool()
async def analysis_wagons(analysis_id: int) -> str:
    """List the wagons of an analysis (wagon id, name, last test train id)."""
    data = await _get("analysis/wagons-by-analyses.jsp",
                      {"analysis_ids": analysis_id})
    if not isinstance(data, dict) or not data:
        return f"No wagons found for analysis {analysis_id}."
    rows = sorted(data.values(), key=lambda w: str(w.get("name", "")).lower())
    lines = [f"{len(rows)} wagons in analysis {analysis_id}:\n",
             f"{'WagonID':>8}  {'TrainID':>8}  Name"]
    lines.append("-" * 70)
    for w in rows:
        lines.append(f"{w.get('id'):>8}  {str(w.get('train_id') or '-'):>8}  "
                     f"{w.get('name')}")
    return "\n".join(lines)


@mcp.tool()
async def wagon_config(wagon_id: int, device: str = "") -> str:
    """Show a wagon's merged configuration (device -> parameters).

    If `device` is given, only devices whose name contains that substring are
    shown (e.g. "pid-tpc-service"); otherwise the device list + sizes is shown.
    """
    cfg = await _get("analysis/wagon/download-configuration.jsp",
                     {"wagon_id": wagon_id})
    if not isinstance(cfg, dict) or not cfg:
        return f"No configuration for wagon {wagon_id}."
    devices = {k: v for k, v in cfg.items() if isinstance(v, dict)}
    if not device:
        lines = [f"Wagon {wagon_id}: {len(devices)} configured devices:\n"]
        for k in sorted(devices):
            lines.append(f"  {k}  ({len(devices[k])} params)")
        lines.append("\nPass device=<substring> to see a device's parameters.")
        return "\n".join(lines)
    matched = {k: v for k, v in devices.items() if device in k}
    if not matched:
        return f"Wagon {wagon_id}: no device matching '{device}'."
    lines = []
    for k in sorted(matched):
        lines.append(f"[{k}]")
        for p in sorted(matched[k]):
            lines.append(f"  {p} = {matched[k][p]}")
        lines.append("")
    return "\n".join(lines).rstrip()


@mcp.tool()
async def wagon_detail(wagon_id: int) -> str:
    """Show a wagon's identity and dependency chain (read-only, any analysis).

    Reports name, owning analysis, workflow name, derived-data limits and the
    dependency wagons with their resolved names and owning analyses — the
    information needed to understand how a train composed from this wagon is
    put together (e.g. which producer provides which workflow). Follow up with
    wagon_detail on a dependency id to walk the chain.
    """
    w = await _get("analysis/wagon/wagon.jsp",
                   {"wagon_id": int(wagon_id), "referenceTime": 0})
    if not isinstance(w, dict) or w.get("id") is None:
        return f"No wagon {wagon_id} (or not accessible)."
    lines = [f"Wagon {w.get('id')} '{w.get('name')}'",
             f"  analysis:   {w.get('analysis_id')} ({w.get('analysis_name')})",
             f"  workflow:   {w.get('work_flow_name')}",
             f"  max_df_size: {w.get('max_df_size')}  "
             f"max_derived_file_size: {w.get('max_derived_file_size')}  "
             f"slim_ready: {w.get('slim_ready')}",
             f"  last change: {w.get('changed_by')}"]
    # Resolve dependency ids to names/analyses via the parallel existing_* arrays.
    dep_info = {}
    ex_ids = str(w.get("existing_dependencies") or "").split(",")
    ex_names = str(w.get("existing_dependencies_name") or "").split(",")
    ex_ana = str(w.get("existing_dependencies_analysis_name") or "").split(",")
    for i, d in enumerate(ex_ids):
        if d:
            dep_info[d] = (ex_names[i] if i < len(ex_names) else "?",
                           ex_ana[i] if i < len(ex_ana) else "?")
    deps = [d for d in str(w.get("dependencies") or "").split(",") if d]
    if not deps:
        lines.append("  dependencies: (none)")
    else:
        lines.append(f"  dependencies ({len(deps)}):")
        for d in deps:
            name, ana = dep_info.get(d, ("?", "?"))
            lines.append(f"    {d:>8}  {name}  [{ana}]")
    return "\n".join(lines)


@mcp.tool()
async def find_wagons_by_config(analysis_id: int, param: str,
                                value: str | None = None) -> str:
    """Find wagons in an analysis whose config sets a given parameter.

    Scans every wagon's merged config for a device parameter whose name
    contains `param` (e.g. "useNetworkCorrection"). If `value` is given, only
    wagons where the parameter equals it are reported. Each hit resolves the
    wagon's dataset name(s) so Run 2 vs Run 3 is visible.

    Example: find wagons running the TPC PID neural network ->
      find_wagons_by_config(50446, "pidTPC.useNetworkCorrection", "1")
    """
    wagons = await _get("analysis/wagons-by-analyses.jsp",
                        {"analysis_ids": analysis_id})
    if not isinstance(wagons, dict) or not wagons:
        return f"No wagons found for analysis {analysis_id}."

    # wagon_id -> [dataset names], via the wagon<->dataset associations.
    assoc = await _get("analysis/wagondataset-by-analyses.jsp",
                       {"analysis_ids": analysis_id})
    train_ids = {a.get("test_train_id") for a in (assoc or [])
                 if a.get("test_train_id")}
    train_ds = {}
    for tid in train_ids:
        try:
            t = await _get("trains/train.jsp", {"train_id": tid})
            t = t[0] if isinstance(t, list) else t
            train_ds[tid] = t.get("dataset_name")
        except Exception:
            pass
    wagon_ds: dict = {}
    for a in (assoc or []):
        ds = train_ds.get(a.get("test_train_id"))
        if ds:
            wagon_ds.setdefault(str(a.get("wagon_id")), set()).add(ds)

    hits = []
    for wid, w in wagons.items():
        try:
            cfg = await _get("analysis/wagon/download-configuration.jsp",
                             {"wagon_id": wid})
        except Exception:
            continue
        for dev, c in cfg.items() if isinstance(cfg, dict) else []:
            if not isinstance(c, dict):
                continue
            for p, v in c.items():
                if param not in p:
                    continue
                if value is not None and str(v) != str(value):
                    continue
                ds = ", ".join(sorted(wagon_ds.get(str(wid), []))) or "?"
                hits.append((w.get("name"), wid, dev, p, str(v), ds))

    if not hits:
        cond = f"{param}={value}" if value is not None else param
        return f"No wagons in analysis {analysis_id} match {cond}."
    lines = [f"Wagons in analysis {analysis_id} matching '{param}'"
             + (f"={value}" if value is not None else "") + ":\n"]
    for name, wid, dev, p, v, ds in hits:
        lines.append(f"  {str(name)[:34]:34} wagon {wid:>6} | {dev} | {p}={v} | {ds}")
    return "\n".join(lines)


@mcp.tool()
async def wagon_status(analysis_id: int, wagon_name: str,
                       dataset: str = "") -> str:
    """Monitor a wagon's latest test run(s) in an analysis, one row per dataset.

    A wagon is tested once per dataset, so the dataset matters: this resolves
    every wagon in `analysis_id` whose name contains `wagon_name`
    (case-insensitive substring), finds its most recent test train per dataset,
    and reports state, job progress (done/total), error rate and package. Pass
    `dataset` to restrict to datasets whose name contains that substring.

    Use to track the progress of a specific wagon, e.g.
      wagon_status(50446, "PIDTPCServiceTests")
      wagon_status(50446, "PIDTPCServiceTests", "PbPb")
    For full per-run metrics (CPU/mem/throughput) follow up with train_detail on
    the reported train ID.
    """
    wagons = await _get("analysis/wagons-by-analyses.jsp",
                        {"analysis_ids": analysis_id})
    if not isinstance(wagons, dict) or not wagons:
        return f"No wagons found for analysis {analysis_id}."
    needle = wagon_name.lower()
    matched = {wid: w for wid, w in wagons.items()
               if needle in str(w.get("name", "")).lower()}
    if not matched:
        return (f"No wagon in analysis {analysis_id} matches '{wagon_name}'. "
                f"Use analysis_wagons({analysis_id}) to list them.")

    # wagon_id -> {test_train_id}: each association is one (wagon, dataset) test.
    assoc = await _get("analysis/wagondataset-by-analyses.jsp",
                       {"analysis_ids": analysis_id})
    tids_by_wagon: dict = {}
    for a in (assoc or []):
        tid = a.get("test_train_id")
        if tid:
            tids_by_wagon.setdefault(str(a.get("wagon_id")), set()).add(tid)

    out = []
    for wid, w in sorted(matched.items(),
                         key=lambda kv: str(kv[1].get("name", "")).lower()):
        out.append(f"Wagon {wid}: {w.get('name')}")
        trains = []
        for tid in sorted(tids_by_wagon.get(str(wid), ()), reverse=True):
            try:
                t = await _get("trains/train.jsp", {"train_id": tid})
                trains.append(t[0] if isinstance(t, list) else t)
            except Exception:
                pass
        if dataset:
            d = dataset.lower()
            trains = [t for t in trains
                      if d in str(t.get("dataset_name", "")).lower()]
        # Keep only the latest test (highest train id) per dataset.
        latest: dict = {}
        for t in trains:
            ds = t.get("dataset_name", "?")
            if t.get("id", 0) > latest.get(ds, {}).get("id", -1):
                latest[ds] = t
        if not latest:
            out.append("  (no test runs"
                       + (f" matching dataset '{dataset}'" if dataset else "")
                       + ")\n")
            continue
        rows = sorted(latest.values(), key=lambda t: t.get("id", 0), reverse=True)
        out.append(_format_train_table(rows) + "\n")
    return "\n".join(out).rstrip()


@mcp.tool()
async def analysis_trains(analysis_id: int, days: int = 14,
                          daily_only: bool = True, dataset: str = "") -> str:
    """Recent test-train history for an analysis — the source for trend analysis.

    Each daily release re-tests an analysis's wagons, producing a test train.
    This lists those trains (id, date, state, dataset, wagons), most recent
    first, filtered to the last `days` (by package date); `daily_only` keeps
    only daily builds and `dataset` filters by substring. Feed the train IDs to
    `test_metrics` to build a per-release time series (CPU / PSS / throughput).
    """
    raw = await _get("analysis/trains-by-analyses.jsp", {"analysis_ids": analysis_id})
    c = raw[0] if isinstance(raw, list) and raw else raw
    trains = c.get("trains", []) if isinstance(c, dict) else []
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y%m%d")
    rows = []
    for t in trains:
        d = _tag_date(t.get("package_tag"))
        if not d or d < cutoff:
            continue
        if daily_only and "daily" not in (t.get("package_tag") or "").lower():
            continue
        if dataset and dataset.lower() not in (t.get("dataset_name") or "").lower():
            continue
        rows.append(t)
    if not rows:
        return f"No matching test trains in analysis {analysis_id} (last {days}d)."
    rows.sort(key=lambda t: (_tag_date(t.get("package_tag")) or "", t.get("id", 0)),
              reverse=True)
    lines = [f"{len(rows)} test trains in analysis {analysis_id} (last {days}d"
             + (", daily" if daily_only else "") + "):\n",
             f"{'date':>8}  {'train':>7}  {'state':<10} {'dataset':<26} wagons"]
    lines.append("-" * 100)
    for t in rows:
        lines.append(f"{_tag_date(t.get('package_tag')):>8}  {t.get('id'):>7}  "
                     f"{str(t.get('state'))[:10]:<10} "
                     f"{str(t.get('dataset_name'))[:26]:<26} "
                     f"{(t.get('wagons_names') or '')[:42]}")
    return "\n".join(lines)


@mcp.tool()
async def test_metrics(train_id: int, per_device: bool = False) -> str:
    """Resource metrics for one test train (from performanceMetrics_processed.json).

    Aggregates per-device CPU (`cpuUsedAbsolute`) and peak PSS, plus the input
    actually processed. With `per_device=True`, lists the heaviest devices — the
    hot spots. Call across the train IDs from `analysis_trains` to build a trend
    (these tests are time-limited, so CPU/PSS move with optimizations while raw
    throughput is often I/O-bound and flat).
    """
    try:
        d = await _get_workdir_json(train_id, "performanceMetrics_processed.json")
    except Exception as e:
        return f"No performance metrics for test {train_id} ({e})."
    devs = []
    tot_cpu = tot_pss = tot_instr = 0.0
    proc = None
    for name, m in d.items():
        if not isinstance(m, dict):
            continue
        cpu = _series_sum(m.get("cpuUsedAbsolute"))
        instr = _series_sum(m.get("cpuInstructions"))
        pss = (m.get("proportionalSetSize_summary") or {}).get("max", 0.0)
        tot_cpu += cpu
        tot_instr += instr
        tot_pss += pss
        if "processed_size" in m:
            proc = _series_max(m["processed_size"]) or proc
        if name.startswith("o2-") and (cpu or pss):
            devs.append((name, cpu, instr, pss, m.get("wagon_id")))
    lines = [f"Test {train_id}: total cpuAbs={tot_cpu:,.0f}"
             + (f"  instr={tot_instr:,.0f}" if tot_instr else "")
             + f"  PSS(sum dev max)={_fmt_bytes(tot_pss)}"
             + (f"  processed={_fmt_bytes(proc)}" if proc else "")]
    if per_device:
        devs.sort(key=lambda x: -x[1])
        lines.append(f"\n{'device':<46} {'cpuAbs':>13} {'instr':>15} {'PSS':>10} {'wagon':>7}")
        for name, cpu, instr, pss, wid in devs[:18]:
            lines.append(f"{name[:46]:<46} {cpu:>13,.0f} {instr:>15,.0f} "
                         f"{_fmt_bytes(pss):>10} {str(wid or ''):>7}")
    return "\n".join(lines)


async def _daily_tests(analysis_ids: list[int], days: int) -> dict:
    """train_id -> (date, dataset) for daily test trains across analyses, last `days`."""
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y%m%d")
    seen: dict = {}
    for aid in analysis_ids:
        try:
            raw = await _get("analysis/trains-by-analyses.jsp", {"analysis_ids": aid})
        except Exception:
            continue
        c = raw[0] if isinstance(raw, list) and raw else raw
        for t in (c.get("trains", []) if isinstance(c, dict) else []):
            d = _tag_date(t.get("package_tag"))
            if not d or d < cutoff or t.get("state") != "done":
                continue
            if "daily" not in (t.get("package_tag") or "").lower():
                continue
            seen[t["id"]] = (d, t.get("dataset_name"))
    return seen


@mcp.tool()
async def wagon_trend(device: str = "", analysis_ids: str = "21674,50446,50462,50570",
                      days: int = 14, metric: str = "throughput") -> str:
    """Optimization-progress trend across recent **daily** test trains (normalized).

    Per (dataset, day), builds a series of the chosen metric and normalizes each
    dataset to its first day (1.00 = start), so you read the relative change as
    fixes land. Daily-only — eulisse-local / non-daily builds excluded.

    metric:
      'instructions_per_gb'  device retired instructions (`cpuInstructions`) / input
                    GB, for devices matching `device`. The cleanest efficiency
                    metric: unlike CPU-time it is invariant to CPU frequency, core
                    contention and the test's wall-clock cap, so a real ↓ is the
                    optimization landing. Falls back to cpu_per_gb on tests run
                    before the instruction counter shipped (no `cpuInstructions`).
      'cpu_per_gb'  device cpuUsedAbsolute / input GB, for devices matching
                    `device`. Efficiency: ↓ means the optimization landed
                    (normalizes out per-run work variation; far cleaner than raw cpu).
      'throughput'  input_size / wall_time (MB/s). The honest "did it get faster"
                    measure — raw CPU can RISE when a faster upstream stage stops
                    starving a downstream one. `device` is ignored.
      'cpu'         raw device cpuUsedAbsolute (noisy — scales with work done).
      'pss'         peak proportionalSetSize for matching devices.

    Pool analyses (default: integration/nightly/MC test analyses) so cross-cutting
    hot spots get many datasets.

    Examples:
      wagon_trend(metric="throughput")
      wagon_trend("tracks-extra-v002-converter", metric="cpu_per_gb")
    """
    aids = [int(x) for x in str(analysis_ids).replace(" ", "").split(",") if x]
    tests = await _daily_tests(aids, days)
    if not tests:
        return f"No daily test trains for analyses {aids} in the last {days}d."
    key = device.lower()
    series: dict = collections.defaultdict(dict)   # dataset -> {date: value}
    matched: set = set()
    need_train = metric in ("throughput", "cpu_per_gb", "instructions_per_gb")
    for tid, (d, ds) in tests.items():
        ins = wall = None
        if need_train:
            try:
                tj = await _get("trains/train.jsp", {"train_id": tid})
                tj = tj[0] if isinstance(tj, list) else tj
                ins, wall = tj.get("input_size"), tj.get("wall_time")
            except Exception:
                continue
        if metric == "throughput":
            if ins and wall:
                series[ds][d] = max(series[ds].get(d, 0.0), ins / wall / 1e6)
            continue
        if metric in ("cpu_per_gb", "instructions_per_gb") and not ins:
            continue
        try:
            pj = await _get_workdir_json(tid, "performanceMetrics_processed.json")
        except Exception:
            continue
        val = 0.0
        hit = False
        for name, m in pj.items():
            if not isinstance(m, dict) or (key and key not in name.lower()):
                continue
            hit = True
            matched.add(name)
            if metric == "pss":
                val += (m.get("proportionalSetSize_summary") or {}).get("max", 0.0)
            elif metric == "instructions_per_gb":
                # Prefer retired instructions; fall back to CPU-µs when the test
                # predates the instruction counter (so old + new days stay comparable).
                instr = _series_sum(m.get("cpuInstructions"))
                val += instr if instr else _series_sum(m.get("cpuUsedAbsolute"))
            else:
                val += _series_sum(m.get("cpuUsedAbsolute"))
        if hit:
            if metric in ("cpu_per_gb", "instructions_per_gb"):
                val = val / (ins / 1e9)
            series[ds][d] = max(series[ds].get(d, 0.0), val)
    if metric != "throughput" and not matched:
        return (f"No device matching '{device}' in the daily tests of {aids}. "
                f"Try test_metrics(<train_id>, per_device=True) to see device names.")
    units = {"throughput": "MB/s", "cpu_per_gb": "CPU/GB", "instructions_per_gb": "instr/GB",
             "cpu": "cpuAbs", "pss": "PSS"}
    out = [f"Trend [{units.get(metric, metric)}, normalized to first day]  "
           + (f"device '{device}'  " if device else "")
           + f"analyses {aids}, last {days}d"]
    if matched:
        out.append(f"matched devices: {', '.join(sorted(matched))}")
    out.append("")
    for ds, ser in sorted(series.items(), key=lambda kv: -len(kv[1])):
        if len(ser) < 2:
            continue
        xs = sorted(ser)
        base = ser[xs[0]]
        if not base:
            continue
        pts = "  ".join(f"{x[4:6]}/{x[6:8]}={ser[x] / base:.2f}" for x in xs)
        chg = 100 * (ser[xs[-1]] / base - 1)
        out.append(f"{ds:24} ({len(xs):2} pts)  first→last {chg:+5.0f}%   {pts}")
    return "\n".join(out)


@mcp.tool()
async def clone_wagon(src_wagon_id: int, name: str) -> str:
    """Clone an existing wagon into the O2 Development analysis (50446).

    WRITE operation — it creates a new wagon. It is HARD-LOCKED to analysis 50446
    ("O2 Development"): the destination is baked in, there is no analysis
    argument, so it physically cannot create or modify wagons anywhere else.
    Inert unless the server was started with HYPERLOOP_ALLOW_WRITE=1.

    `src_wagon_id` may come from any analysis (e.g. a pre-configured creator or
    builder you found with analysis_wagons / find_wagons_by_config). The new
    wagon's name is always prefixed with 'Test' so created wagons are easy to
    spot and clean up; you may pass `name` with or without the prefix.

    Returns the server response and a read-back confirmation. Inspect the result
    with analysis_wagons(50446).
    """
    if not ALLOW_WRITE:
        return ("Refused: writes are disabled. Start the MCP server with "
                "HYPERLOOP_ALLOW_WRITE=1 to enable wagon creation (locked to "
                f"analysis {ALLOWED_ANALYSIS}).")
    name = (name or "").strip()
    if not name:
        return "Refused: a non-empty wagon name is required."
    if not name.startswith(WAGON_PREFIX):
        name = f"{WAGON_PREFIX}{name}"
    # Hard guardrail: destination analysis is the baked-in constant, never a caller arg.
    params = {"wagon_id": int(src_wagon_id), "name": name,
              "to_analysis_id": ALLOWED_ANALYSIS}
    try:
        resp = await _get_text("analysis/clone-wagon.jsp", params)
    except Exception as e:
        return f"Clone of wagon {src_wagon_id} failed ({e})."
    # Read-back guardrail: confirm the new wagon really landed in 50446.
    landed = False
    try:
        back = await _get("analysis/wagons-by-analyses.jsp",
                          {"analysis_ids": ALLOWED_ANALYSIS})
        landed = name in json.dumps(back)
    except Exception:
        pass
    status = ("confirmed in analysis {}".format(ALLOWED_ANALYSIS) if landed
              else "NOT confirmed — check analysis_wagons({})".format(ALLOWED_ANALYSIS))
    return (f"Cloned wagon {src_wagon_id} -> '{name}' into analysis "
            f"{ALLOWED_ANALYSIS} ({status}).\nServer response: {resp.strip()[:400]}")


async def _post_form(path: str, data: dict) -> str:
    """POST application/x-www-form-urlencoded to a JSP endpoint; return raw text."""
    hdrs = _headers()
    hdrs["Accept-Encoding"] = "identity"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{API}/{path}", data=data, headers=hdrs)
        r.raise_for_status()
        return r.text


async def _wagon_in_allowed(wagon_id: int) -> bool:
    """True iff `wagon_id` belongs to the one writable analysis (50446). The
    by-id write tools refuse anything that isn't in this set, so they cannot
    touch a wagon in another analysis."""
    try:
        data = await _get("analysis/wagons-by-analyses.jsp",
                          {"analysis_ids": ALLOWED_ANALYSIS})
    except Exception:
        return False
    ids: set = set()

    def collect(o):
        if isinstance(o, dict):
            if "id" in o and o.get("analysis_id") == ALLOWED_ANALYSIS:
                try:
                    ids.add(int(o["id"]))
                except (TypeError, ValueError):
                    pass
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)

    collect(data)
    return int(wagon_id) in ids


@mcp.tool()
async def set_wagon_config(wagon_id: int, params: dict) -> str:
    """Set configuration parameters on a wagon in O2 Development (50446).

    WRITE operation. Refuses unless the target wagon belongs to analysis 50446,
    so it cannot modify wagons elsewhere. Inert unless the server was started
    with HYPERLOOP_ALLOW_WRITE=1 (or --allow-write).

    `params` maps parameter name -> new value, e.g.
        {"createDplus": 1, "processNoPvRefitWithDCAFitterNCentFT0M": 1, "do3prong": 1}
    If a name is shared by several tasks, disambiguate with "task_name.param".
    Booleans/ints are sent as Hyperloop stores them ("1"/"0"); arrays pass through.

    It reads the wagon's current config (recovering each param's subwagon/id/type/
    kind), applies the new values, and writes them back in one POST. Verify with
    wagon_config(wagon_id).
    """
    if not ALLOW_WRITE:
        return ("Refused: writes are disabled. Start the server with "
                f"HYPERLOOP_ALLOW_WRITE=1 (locked to analysis {ALLOWED_ANALYSIS}).")
    if not await _wagon_in_allowed(wagon_id):
        return (f"Refused: wagon {wagon_id} is not in analysis {ALLOWED_ANALYSIS} "
                "(or could not be verified). Writes are restricted to that analysis.")
    if not isinstance(params, dict) or not params:
        return "Refused: `params` must be a non-empty {name: value} mapping."
    try:
        conf = await _get("analysis/wagon/get-subwagons-configuration.jsp",
                          {"lists": "subwagons_configuration",
                           "wagon_id": int(wagon_id), "referenceTime": 0})
    except Exception as e:
        return f"Could not read current config of wagon {wagon_id} ({e})."
    entries = conf.get("subwagons_conf", []) if isinstance(conf, dict) else []
    if not entries:
        return f"No configuration entries returned for wagon {wagon_id}."
    by_key: dict = {}
    by_name: dict = {}
    subwagon_tasks: dict = {}
    for e in entries:
        tn, nm, sid = e.get("task_name"), e.get("name"), e.get("subwagon_id")
        by_key[(tn, nm)] = e
        by_name.setdefault(nm, []).append(e)
        subwagon_tasks.setdefault(sid, set()).add(tn)
    resolved: list = []
    errors: list = []
    for key, val in params.items():
        task = None
        nm = key
        if "." in key:
            cand_task, cand_name = key.split(".", 1)
            if any(cand_task == t for (t, _) in by_key):
                task, nm = cand_task, cand_name
        matches = ([by_key[(task, nm)]] if (task and (task, nm) in by_key)
                   else by_name.get(nm, []))
        if not matches:
            errors.append(f"'{key}': no such parameter")
        elif len(matches) > 1:
            tasks = sorted({m.get("task_name") for m in matches})
            errors.append(f"'{key}': ambiguous across tasks {tasks}; use 'task.param'")
        else:
            resolved.append((matches[0], val))
    if errors:
        return "Refused (nothing written):\n  " + "\n  ".join(errors)

    def coerce(entry, val):
        ev = entry.get("value")
        if isinstance(val, bool):
            return "1" if val else "0"
        if isinstance(ev, str) and isinstance(val, (int, float)):
            return str(val)
        return val

    subs: dict = {}
    for e, val in resolved:
        sid = e["subwagon_id"]
        sval = coerce(e, val)
        blk = subs.setdefault(sid, {"task": {}, "configuration": {}, "id": str(sid)})
        for t in subwagon_tasks.get(sid, set()):
            blk["task"].setdefault(t, {"configuration": {}})
        nm, tn = e["name"], e["task_name"]
        blk["task"][tn]["configuration"][nm] = {
            "id": e.get("id"), "name": nm, "value": sval, "help": e.get("help"),
            "labels_rows": e.get("labels_rows"), "labels_cols": e.get("labels_cols"),
            "type": e.get("type"), "kind": e.get("kind"), "conf": e.get("conf"),
        }
        blk["configuration"][nm] = {
            "task_name": tn, "value": sval, "type": e.get("type"),
            "labels_rows": e.get("labels_rows"), "labels_cols": e.get("labels_cols"),
            "kind": e.get("kind"), "help": e.get("help"), "id": e.get("id"),
        }
    payload = {str(sid): blk for sid, blk in subs.items()}
    try:
        resp = await _post_form("analysis/wagon/update-subwagon-configuration.jsp",
                                {"subwagons": json.dumps(payload)})
    except Exception as e:
        return f"Config update of wagon {wagon_id} failed ({e})."
    changed = ", ".join(f"{e['task_name']}.{e['name']}={coerce(e, v)}" for e, v in resolved)
    return (f"Updated wagon {wagon_id} in analysis {ALLOWED_ANALYSIS}: {changed}.\n"
            f"Server response: {resp.strip()[:300]}")


@mcp.tool()
async def set_wagon_dependencies(wagon_id: int, dependency_wagon_ids: list) -> str:
    """Set the dependency wagons of a wagon in O2 Development (50446).

    WRITE operation. Refuses unless the target wagon is in analysis 50446, so it
    cannot modify wagons elsewhere. Inert unless the server was started with
    HYPERLOOP_ALLOW_WRITE=1 (or --allow-write).

    `dependency_wagon_ids` REPLACES the wagon's full dependency set (mirroring the
    UI). Pass the complete producer chain, e.g. [564, 3443, 9998]; pass [] to
    clear. The wagon's other fields (name, workflow, max sizes, slim flag) are
    read first and preserved, so only the dependency list changes. Verify with
    analysis_wagons(50446).
    """
    if not ALLOW_WRITE:
        return ("Refused: writes are disabled. Start the server with "
                f"HYPERLOOP_ALLOW_WRITE=1 (locked to analysis {ALLOWED_ANALYSIS}).")
    try:
        w = await _get("analysis/wagon/wagon.jsp",
                       {"wagon_id": int(wagon_id), "referenceTime": 0})
    except Exception as e:
        return f"Could not read wagon {wagon_id} ({e})."
    if not isinstance(w, dict) or w.get("analysis_id") != ALLOWED_ANALYSIS:
        return (f"Refused: wagon {wagon_id} is not in analysis {ALLOWED_ANALYSIS} "
                "(or could not be verified). Writes are restricted to that analysis.")
    try:
        deps = ",".join(str(int(x)) for x in (dependency_wagon_ids or []))
    except (TypeError, ValueError):
        return "Refused: dependency_wagon_ids must be a list of integer wagon ids."
    # Read-modify-write: preserve every other wagon field, change only dependencies.
    params = {
        "id": int(wagon_id),
        "name": w.get("name", ""),
        "work_flow_name": w.get("work_flow_name", ""),
        "dependencies": deps,
        "max_df_size": w.get("max_df_size", 100000000),
        "max_derived_file_size": w.get("max_derived_file_size", 0),
        "slim_ready": "true" if w.get("slim_ready") else "false",
    }
    try:
        resp = await _get_text("analysis/wagon/update-wagon.jsp", params)
    except Exception as e:
        return f"Dependency update of wagon {wagon_id} failed ({e})."
    now = ""
    try:
        w2 = await _get("analysis/wagon/wagon.jsp",
                        {"wagon_id": int(wagon_id), "referenceTime": 0})
        now = w2.get("dependencies", "") if isinstance(w2, dict) else ""
    except Exception:
        pass
    return (f"Set dependencies of wagon {wagon_id} ('{w.get('name')}') to "
            f"[{deps or '(none)'}] in analysis {ALLOWED_ANALYSIS}. "
            f"Now: [{now or '(none)'}].\nServer response: {resp.strip()[:200]}")


async def _resolve_dataset_id(dataset: str):
    """Resolve a dataset NAME (or numeric id) to its numeric id.
    Returns (id, None) on success or (None, error_message)."""
    s = str(dataset).strip()
    if s.isdigit():
        return int(s), None
    try:
        lst = await _get("dataset/list-dataset.jsp", {"lists": "dataset-list"})
    except Exception as e:
        return None, f"could not fetch the dataset list ({e})"
    items = lst if isinstance(lst, list) else []
    matches = [it for it in items if isinstance(it, dict) and it.get("name") == s]
    if not matches:
        return None, f"no dataset named '{s}' found"
    if len(matches) > 1:
        return None, f"'{s}' is ambiguous: ids {[m.get('id') for m in matches]}"
    return int(matches[0]["id"]), None


@mcp.tool()
async def subscribe_dataset(dataset: str) -> str:
    """Subscribe (enable) a dataset to the O2 Development analysis (50446).

    WRITE operation. HARD-LOCKED to analysis 50446 — there is no analysis
    argument, so it can only ever subscribe a dataset to O2 Development. Inert
    unless the server was started with HYPERLOOP_ALLOW_WRITE=1 (or --allow-write).

    `dataset` is the dataset NAME (e.g. "LHC26ac_pass1_Thin_small") or its numeric
    id; names are resolved via the dataset list. Returns a read-back confirmation
    that 50446 now appears among the dataset's subscribed analyses.
    """
    if not ALLOW_WRITE:
        return ("Refused: writes are disabled. Start the server with "
                f"HYPERLOOP_ALLOW_WRITE=1 (locked to analysis {ALLOWED_ANALYSIS}).")
    dsid, err = await _resolve_dataset_id(dataset)
    if dsid is None:
        return f"Refused: {err}."
    try:
        resp = await _get_text("analysis/enable-dataset.jsp",
                               {"dataset_id": dsid, "analysis_id": ALLOWED_ANALYSIS})
    except Exception as e:
        return f"Subscribing dataset '{dataset}' (id {dsid}) failed ({e})."
    # Read-back: confirm analysis 50446 is now among the dataset's analyses.
    subscribed = False
    try:
        lst = await _get("dataset/list-dataset.jsp",
                         {"lists": "dataset-analysis", "dataset_id": dsid})
        if isinstance(lst, list):
            subscribed = any(isinstance(a, dict) and a.get("id") == ALLOWED_ANALYSIS
                             for a in lst)
    except Exception:
        pass
    status = (f"confirmed subscribed to analysis {ALLOWED_ANALYSIS}" if subscribed
              else "NOT confirmed — check the dataset's analyses")
    return (f"Subscribed dataset '{dataset}' (id {dsid}) to analysis "
            f"{ALLOWED_ANALYSIS} ({status}).\nServer response: {resp.strip()[:200]}")


def main():
    import argparse
    global PROXY, TOKEN, API, ALLOW_WRITE

    parser = argparse.ArgumentParser(description="AliHyperloop MCP server")
    parser.add_argument("--proxy", default=PROXY, help="Proxy base URL")
    parser.add_argument("--token", default=TOKEN, help="Bearer token")
    parser.add_argument("--allow-write", action="store_true",
                        help=("Enable the wagon-write tools (clone/configure), "
                              f"hard-locked to analysis {ALLOWED_ANALYSIS}. "
                              "Off by default; HYPERLOOP_ALLOW_WRITE=1 also enables it."))
    args = parser.parse_args()

    PROXY = args.proxy
    TOKEN = args.token
    API = f"{PROXY}/alihyperloop-data"
    if args.allow_write:
        ALLOW_WRITE = True

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
