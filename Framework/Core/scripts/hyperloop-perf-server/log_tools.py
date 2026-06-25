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
"""Log tools for the Hyperloop perf MCP server.

A train/device log (e.g. ``stdout.log``) is fetched once through the alimonitor
proxy and cached on disk; subsequent ``grep_log`` calls run regex queries over
the cached copy and return at most ``max_results`` matches (with optional
context), so a multi-MB log never has to come back over the wire — or into the
model's context — in full.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
from dataclasses import dataclass

from hl_common import fetch_bytes

_CACHE_DIR = os.path.expanduser(os.environ.get("LOG_MCP_CACHE", "~/.cache/log-mcp"))
_MAX_LINE = 2000  # truncate individual lines in the output to keep results bounded


@dataclass
class LogReport:
    url: str
    name: str
    path: str
    n_lines: int
    n_bytes: int


_logs: dict[str, LogReport] = {}


def _get(name: str) -> LogReport:
    r = _logs.get(name)
    if r is None:
        avail = ", ".join(_logs) if _logs else "(none)"
        raise ValueError(f"No log '{name}'. Loaded: {avail}. Use load_log first.")
    return r


def _clip(line: str) -> str:
    return line if len(line) <= _MAX_LINE else line[:_MAX_LINE] + " …[truncated]"


async def load_log(url: str, name: str = "", proxy_token: str = "") -> str:
    """Fetch a log file and cache it for regex querying with grep_log.

    The file is downloaded (via the alimonitor proxy for ``alimonitor.cern.ch``
    URLs), decompressed if gzip'd, and cached on disk; grep_log then reads that
    cached copy and never re-fetches.

    Args:
        url:         Direct URL to a log file (e.g. .../stdout.log or a .gz log).
        name:        Label (defaults to the filename portion of the URL).
        proxy_token: Bearer token for the local proxy (else PROXY_TOKEN env).
    """
    raw = await fetch_bytes(url, proxy_token=proxy_token)
    data = gzip.decompress(raw) if (url.endswith(".gz") or raw[:2] == b"\x1f\x8b") else raw
    text = data.decode("utf-8", errors="replace")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    h = hashlib.sha1(url.encode()).hexdigest()[:12]
    path = os.path.join(_CACHE_DIR, f"{h}.log")
    with open(path, "w", errors="replace") as f:
        f.write(text)
    n_lines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    pname = name or url.rstrip("/").split("/")[-1]
    _logs[pname] = LogReport(url, pname, path, n_lines, len(data))
    return f"Loaded log '{pname}': {n_lines:,} lines, {len(data):,} bytes."


def grep_log(
    name: str,
    pattern: str,
    max_results: int = 50,
    ignore_case: bool = False,
    invert: bool = False,
    context: int = 0,
) -> str:
    """Regex-search a cached log and return at most max_results matching lines.

    Args:
        name:        Log name as returned by load_log.
        pattern:     Python regex (re.search semantics, matches anywhere in a line).
        max_results: Maximum number of matching lines to return (default 50).
        ignore_case: Case-insensitive match.
        invert:      Return non-matching lines instead.
        context:     Lines of context to show before and after each match (like grep -C).
    """
    r = _get(name)
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return f"bad regex: {e}"
    if max_results < 1:
        return "max_results must be >= 1"

    with open(r.path, errors="replace") as f:
        lines = f.read().splitlines()

    total = 0
    hits: list[int] = []  # line indices of the first max_results matches
    for i, line in enumerate(lines):
        matched = bool(rx.search(line))
        if invert:
            matched = not matched
        if matched:
            total += 1
            if len(hits) < max_results:
                hits.append(i)

    if total == 0:
        return f"[{name}] no matches for /{pattern}/ in {r.n_lines:,} lines"

    ctx = max(0, context)
    out: list[str] = []
    prev_end = -1  # last printed line index, to insert separators / avoid dup
    for idx in hits:
        lo, hi = max(0, idx - ctx), min(len(lines) - 1, idx + ctx)
        if lo <= prev_end:  # overlap with previous block: continue from there
            lo = prev_end + 1
        elif prev_end >= 0:
            out.append("--")
        for j in range(lo, hi + 1):
            mark = ":" if j == idx else "-"  # ':' = the match line, '-' = context
            out.append(f"{j + 1}{mark} {_clip(lines[j])}")
        prev_end = hi

    shown = min(total, max_results)
    header = f"[{name}] {total} match(es) for /{pattern}/" + (
        f"; showing first {shown}" if total > shown else ""
    )
    return header + "\n" + "\n".join(out)


def list_logs() -> str:
    """List loaded logs."""
    if not _logs:
        return "No logs loaded. Use load_log first."
    return "\n".join(
        f"{n}: {r.n_lines:,} lines, {r.n_bytes:,} bytes, url={r.url}" for n, r in _logs.items()
    )


def drop_log(name: str) -> str:
    """Free a log and delete its cached copy.

    Args:
        name: Log name as returned by load_log.
    """
    r = _get(name)
    if os.path.exists(r.path):
        os.remove(r.path)
    del _logs[name]
    return f"Dropped log '{name}'."


def register(mcp) -> None:
    """Register the log tools on a shared FastMCP instance."""
    for fn in (load_log, grep_log, list_logs, drop_log):
        mcp.tool()(fn)
