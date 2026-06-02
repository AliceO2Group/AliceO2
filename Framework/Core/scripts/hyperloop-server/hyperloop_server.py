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
import json
import os
import sys
import time

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hyperloop")

PROXY = os.environ.get("HYPERLOOP_PROXY", "http://localhost:8888")
TOKEN = os.environ.get("HYPERLOOP_TOKEN", "foo-baz")
API = f"{PROXY}/alihyperloop-data"


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


async def _get(path: str, params: dict | None = None) -> any:
    hdrs = _headers()
    hdrs["Accept-Encoding"] = "identity"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{API}/{path}", params=params, headers=hdrs)
        r.raise_for_status()
        return r.json()


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


def main():
    import argparse
    global PROXY, TOKEN, API

    parser = argparse.ArgumentParser(description="AliHyperloop MCP server")
    parser.add_argument("--proxy", default=PROXY, help="Proxy base URL")
    parser.add_argument("--token", default=TOKEN, help="Bearer token")
    args = parser.parse_args()

    PROXY = args.proxy
    TOKEN = args.token
    API = f"{PROXY}/alihyperloop-data"

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
