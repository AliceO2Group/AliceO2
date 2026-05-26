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
"""Hyperloop perf profile MCP server.

Fetches Linux ``perf script`` output files hosted on the Hyperloop web server,
parses and indexes them in memory, and exposes query tools so an AI assistant
can investigate performance hotspots without ever seeing the raw file.

Usage
-----
    python3 perf_mcp_server.py

Multiple profiles can be loaded simultaneously, enabling before/after
comparisons via the ``compare`` tool.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Perf profile data model
# ---------------------------------------------------------------------------

_OFFSET_RE = re.compile(r"\+0x[0-9a-fA-F]+")


def _strip(sym: str) -> str:
    """Remove hex offsets from a symbol name."""
    return _OFFSET_RE.sub("", sym)


@dataclass
class PerfProfile:
    url: str
    total_cycles: int
    processes: set[str]
    # leaf[sym] = cycles where sym was the innermost (hot) frame
    leaf: dict[str, int]
    # inclusive[sym] = cycles where sym appeared anywhere in the stack
    inclusive: dict[str, int]
    # callers[sym][caller] = cycles attributed to this caller→sym edge
    callers: dict[str, dict[str, int]]
    # callees[sym][callee] = cycles attributed to this sym→callee edge
    callees: dict[str, dict[str, int]]
    # raw stacks for chain queries: (cycles, process_name, [sym0, sym1, ...])
    # sym0 is the innermost (leaf) frame; sym[-1] is the outermost caller.
    stacks: list[tuple[int, str, list[str]]] = field(default_factory=list)


def _parse(text: str) -> PerfProfile:
    """Parse ``perf script`` text output into a PerfProfile."""
    leaf: dict[str, int] = defaultdict(int)
    inclusive: dict[str, int] = defaultdict(int)
    callers: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    callees: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    stacks: list[tuple[int, str, list[str]]] = []
    processes: set[str] = set()
    total_cycles = 0

    cycles = 0
    process = ""
    stack: list[str] = []

    def _flush() -> None:
        nonlocal total_cycles
        if not stack or not cycles:
            return
        stacks.append((cycles, process, list(stack)))
        total_cycles += cycles
        leaf[stack[0]] += cycles
        seen: set[str] = set()
        for i, sym in enumerate(stack):
            if sym not in seen:
                inclusive[sym] += cycles
                seen.add(sym)
            # stack[i] is called by stack[i+1]; stack[i] calls stack[i-1]
            if i > 0:
                callers[stack[i - 1]][stack[i]] += cycles
                callees[stack[i]][stack[i - 1]] += cycles

    for line in text.splitlines():
        if not line:
            continue
        if not line[0].isspace():
            _flush()
            m = re.search(r"(\d+) cycles:", line)
            cycles = int(m.group(1)) if m else 0
            parts = line.split()
            process = parts[0] if parts else ""
            processes.add(process)
            stack = []
        elif line[0] == "\t":
            parts = line.split()
            if len(parts) >= 2:
                stack.append(_strip(parts[1]))

    _flush()

    return PerfProfile(
        url="",
        total_cycles=total_cycles,
        processes=processes,
        leaf=dict(leaf),
        inclusive=dict(inclusive),
        callers={k: dict(v) for k, v in callers.items()},
        callees={k: dict(v) for k, v in callees.items()},
        stacks=stacks,
    )


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

_profiles: dict[str, PerfProfile] = {}


def _get(name: str) -> PerfProfile:
    p = _profiles.get(name)
    if p is None:
        available = ", ".join(_profiles.keys()) if _profiles else "(none)"
        raise ValueError(
            f"No profile named '{name}'. Loaded profiles: {available}. "
            f"Use load_profile first."
        )
    return p


def _pct(cycles: int, total: int) -> str:
    return f"{cycles * 100 / total:.2f}%" if total else "N/A"


def _top_table(hot: dict[str, int], total: int, n: int, process_note: str = "") -> str:
    rows = sorted(hot.items(), key=lambda x: -x[1])[:n]
    if not rows:
        return "No data."
    header = f"{'cycles':>16}  {'%':>7}  symbol"
    if process_note:
        header = f"[filtered: {process_note}]\n" + header
    lines = [header]
    for sym, cyc in rows:
        lines.append(f"{cyc:>16,}  {_pct(cyc, total):>7}  {sym}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("Hyperloop Perf")


@mcp.tool()
async def load_profile(url: str, name: str = "", token: str = "", proxy_token: str = "") -> str:
    """Fetch a ``perf script`` text file and index it for querying.

    The file is downloaded once and kept in memory.  Subsequent tool calls
    use the in-memory index and do not re-fetch.

    Args:
        url:         Direct URL to the perf script text file.
        name:        Human-friendly label (defaults to the filename portion of the URL).
        token:       Hyperloop auth token. Falls back to HYPERLOOP_TOKEN env var.
        proxy_token: Bearer token for the local proxy. Falls back to PROXY_TOKEN env var,
                     then to token.
    """
    token = token or os.environ.get("HYPERLOOP_TOKEN", "")
    proxy_token = proxy_token or os.environ.get("PROXY_TOKEN", "") or token

    # Rewrite alimonitor.cern.ch URLs through the local proxy (same pattern as
    # connect_hyperloop).  The proxy must have a route like:
    #   {"prefix": "/alimonitor/", "upstream": "https://alimonitor.cern.ch", "token": "..."}
    fetch_url = url
    if "alimonitor.cern.ch" in url:
        path = url.split("alimonitor.cern.ch", 1)[1].lstrip("/")
        fetch_url = f"http://localhost:8888/alimonitor/{path}"

    headers = {"Authorization": f"Bearer {proxy_token}"} if proxy_token else {}
    headers["Accept-Encoding"] = "identity"

    async with httpx.AsyncClient(verify=False) as client:
        for attempt in range(3):
            try:
                r = await client.get(fetch_url, headers=headers, timeout=300.0, follow_redirects=True)
                r.raise_for_status()
                break
            except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
                if attempt == 2:
                    raise
        text = r.content.decode("utf-8", errors="replace")

    profile = await asyncio.get_event_loop().run_in_executor(None, _parse, text)
    profile.url = url

    pname = name or url.rstrip("/").split("/")[-1]
    _profiles[pname] = profile

    return (
        f"Loaded '{pname}': {len(profile.stacks):,} samples, "
        f"{profile.total_cycles:,} total cycles, "
        f"processes: {', '.join(sorted(profile.processes))}"
    )


@mcp.tool()
def list_profiles() -> str:
    """List all loaded perf profiles with their basic stats."""
    if not _profiles:
        return "No profiles loaded. Use load_profile first."
    lines = []
    for name, p in _profiles.items():
        lines.append(
            f"{name}: {len(p.stacks):,} samples, {p.total_cycles:,} cycles, "
            f"url={p.url}"
        )
    return "\n".join(lines)


@mcp.tool()
def drop_profile(name: str) -> str:
    """Free a loaded profile from memory.

    Args:
        name: Profile name as returned by load_profile.
    """
    _get(name)
    del _profiles[name]
    return f"Dropped profile '{name}'."


@mcp.tool()
def top_functions(profile_name: str, n: int = 40, process: str = "") -> str:
    """Show the hottest leaf functions (where CPU was actually executing).

    Args:
        profile_name: Profile name as returned by load_profile.
        n:            Number of entries to return (default 40).
        process:      Optional process name filter (exact match).
    """
    p = _get(profile_name)
    if process:
        hot: dict[str, int] = defaultdict(int)
        for cyc, proc, stack in p.stacks:
            if proc == process and stack:
                hot[stack[0]] += cyc
        total = sum(hot.values())
        return _top_table(dict(hot), total, n, process)
    return _top_table(p.leaf, p.total_cycles, n)


@mcp.tool()
def top_inclusive(profile_name: str, n: int = 40, process: str = "") -> str:
    """Show the hottest functions by inclusive cycles (appears anywhere in stack).

    Args:
        profile_name: Profile name as returned by load_profile.
        n:            Number of entries to return (default 40).
        process:      Optional process name filter (exact match).
    """
    p = _get(profile_name)
    if process:
        hot: dict[str, int] = defaultdict(int)
        for cyc, proc, stack in p.stacks:
            if proc != process:
                continue
            for sym in set(stack):
                hot[sym] += cyc
        total = sum(cyc for cyc, proc, _ in p.stacks if proc == process)
        return _top_table(dict(hot), total, n, process)
    return _top_table(p.inclusive, p.total_cycles, n)


@mcp.tool()
def callers_of(profile_name: str, sym: str, n: int = 20) -> str:
    """Show what calls a given function, weighted by cycles.

    Uses substring matching on the symbol name.

    Args:
        profile_name: Profile name as returned by load_profile.
        sym:          Symbol name (or substring) to look up.
        n:            Number of callers to return (default 20).
    """
    p = _get(profile_name)
    merged: dict[str, int] = defaultdict(int)
    for fn, caller_map in p.callers.items():
        if sym in fn:
            for caller, cyc in caller_map.items():
                merged[caller] += cyc
    if not merged:
        return f"No callers found for '{sym}'."
    return _top_table(dict(merged), p.total_cycles, n)


@mcp.tool()
def callees_of(profile_name: str, sym: str, n: int = 20) -> str:
    """Show what a given function calls, weighted by cycles.

    Uses substring matching on the symbol name.

    Args:
        profile_name: Profile name as returned by load_profile.
        sym:          Symbol name (or substring) to look up.
        n:            Number of callees to return (default 20).
    """
    p = _get(profile_name)
    merged: dict[str, int] = defaultdict(int)
    for fn, callee_map in p.callees.items():
        if sym in fn:
            for callee, cyc in callee_map.items():
                merged[callee] += cyc
    if not merged:
        return f"No callees found for '{sym}'."
    return _top_table(dict(merged), p.total_cycles, n)


@mcp.tool()
def chains_through(profile_name: str, sym: str, depth: int = 3, n: int = 20, process: str = "") -> str:
    """Show call chain windows centered on a function.

    Finds all samples where ``sym`` appears in the stack and returns the
    most common surrounding call chains, each showing ``depth`` frames on
    either side.

    Args:
        profile_name: Profile name as returned by load_profile.
        sym:          Symbol name (or substring) to search for.
        depth:        Number of frames to show on each side (default 3).
        n:            Number of top chains to return (default 20).
        process:      Optional process name filter (exact match).
    """
    p = _get(profile_name)
    chains: dict[str, int] = defaultdict(int)

    for cyc, proc, stack in p.stacks:
        if process and proc != process:
            continue
        for i, frame in enumerate(stack):
            if sym in frame:
                start = max(0, i - depth)
                end = min(len(stack), i + depth + 1)
                chain = " <- ".join(stack[start:end])
                chains[chain] += cyc
                break

    if not chains:
        return f"No samples found containing '{sym}'."

    lines = [f"Call chains through '{sym}' (depth={depth}):"]
    lines.append(f"{'cycles':>16}  {'%':>7}  chain")
    for chain, cyc in sorted(chains.items(), key=lambda x: -x[1])[:n]:
        lines.append(f"{cyc:>16,}  {_pct(cyc, p.total_cycles):>7}  {chain}")
    return "\n".join(lines)


@mcp.tool()
def compare(name_a: str, name_b: str, n: int = 40, mode: str = "leaf") -> str:
    """Compare two profiles and show which functions changed the most.

    Normalises each profile's cycles to a fraction of its total so that
    differences in overall run length do not skew the comparison.
    Positive Δ means the function got *heavier* in B relative to A.

    Args:
        name_a: Baseline profile name.
        name_b: Comparison profile name.
        n:      Number of entries to show (default 40).
        mode:   "leaf" (default) or "inclusive".
    """
    a = _get(name_a)
    b = _get(name_b)

    hot_a = a.leaf if mode == "leaf" else a.inclusive
    hot_b = b.leaf if mode == "leaf" else b.inclusive
    total_a = a.total_cycles or 1
    total_b = b.total_cycles or 1

    all_syms = set(hot_a) | set(hot_b)
    diffs = []
    for sym in all_syms:
        fa = hot_a.get(sym, 0) / total_a
        fb = hot_b.get(sym, 0) / total_b
        diffs.append((fb - fa, sym, fa, fb))

    diffs.sort(key=lambda x: -abs(x[0]))

    lines = [
        f"Comparing '{name_a}' (A) vs '{name_b}' (B) [{mode}]",
        f"A total: {a.total_cycles:,} cycles   B total: {b.total_cycles:,} cycles",
        f"",
        f"{'Δ%':>8}  {'A%':>7}  {'B%':>7}  symbol",
    ]
    for delta, sym, fa, fb in diffs[:n]:
        lines.append(f"{delta*100:>+8.2f}  {fa*100:>7.2f}  {fb*100:>7.2f}  {sym}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
