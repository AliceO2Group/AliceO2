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
"""IgProf memory-profile tools for the Hyperloop perf MCP server.

IgProf heap dumps are huge pre-order call trees. Rather than parse them in
Python, these tools delegate every query to the ``igprof-query`` C tool (a fast
streaming reader): the dump is fetched + decompressed once and cached on disk,
then ``igprof-query`` is run per query (~100 ms even on a 600k-node dump), so
only the answer's symbols are ever demangled.

Counters in a MEM dump and how they aggregate:
  MEM_TOTAL  total bytes allocated over the run        (summed)
  MEM_MAX    largest single allocation                 (reduced by max)
  MEM_LIVE   bytes still live at dump time = footprint (summed net-of-free)

The ``igprof-query`` binary is located via ``IGPROF_QUERY_BIN`` or ``PATH``.
Build it (with readable names) from ~/src/IgProf:
  cmake -DIGPROF_VIEWER_ONLY=ON -DCMAKE_C_FLAGS=-DIGPROF_DEMANGLE … && make
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from hl_common import fetch_bytes

# ---------------------------------------------------------------------------
# Binary + cache
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.expanduser(os.environ.get("IGPROF_MCP_CACHE", "~/.cache/igprof-mcp"))

_COUNTER_DOC = {
    "MEM_TOTAL": "total bytes allocated over the run (summed)",
    "MEM_MAX": "largest single allocation (reduced by max)",
    "MEM_LIVE": "bytes still live at dump time — footprint / leak (summed net-of-free)",
}


def _bin() -> str:
    b = os.environ.get("IGPROF_QUERY_BIN") or shutil.which("igprof-query")
    if not b:
        raise RuntimeError(
            "igprof-query not found. Set IGPROF_QUERY_BIN or put it on PATH. "
            "Build it from ~/src/IgProf: "
            "cmake -DIGPROF_VIEWER_ONLY=ON -DCMAKE_C_FLAGS=-DIGPROF_DEMANGLE . && make"
        )
    return b


@dataclass
class IgProfReport:
    url: str
    name: str
    dump_path: str
    sidecar_path: str
    counters: list[str]
    default_counter: str


_reports: dict[str, IgProfReport] = {}


def _get(name: str) -> IgProfReport:
    r = _reports.get(name)
    if r is None:
        avail = ", ".join(_reports) if _reports else "(none)"
        raise ValueError(f"No igprof report '{name}'. Loaded: {avail}. Use load_igprof first.")
    return r


def _run(report: IgProfReport, args: list[str]) -> tuple[str, str]:
    cmd = [_bin(), *args]
    if report.sidecar_path:
        cmd += ["-S", report.sidecar_path]
    cmd += [report.dump_path]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise RuntimeError(f"igprof-query failed: {(p.stderr or p.stdout).strip()}")
    return p.stdout, p.stderr


def _enumerate_counters(dump_path: str) -> list[str]:
    """Counters are define-on-first-use (``V<id>=(NAME)``) in the first nodes."""
    seen: list[str] = []
    with open(dump_path, "r", errors="replace") as f:
        for _ in range(400):
            line = f.readline()
            if not line:
                break
            for m in re.finditer(r"V\d+=\(([A-Z_][A-Z0-9_]*)\)", line):
                if m.group(1) not in seen:
                    seen.append(m.group(1))
    return seen


_TOP_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+?)\s*$")


def _parse_top(text: str) -> dict[str, tuple[int, int, int]]:
    """symbol -> (cumulative, self, self_count) from `igprof-query top` output."""
    rows: dict[str, tuple[int, int, int]] = {}
    for line in text.splitlines():
        m = _TOP_ROW.match(line)
        if m:
            # groups: 1=rank 2=cumulative 3=self 4=self-count 5=symbol
            rows[m.group(5)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return rows


def _limit_show(text: str, n: int) -> str:
    """Keep at most `n` edge rows under each `== callers/callees ==` section."""
    out: list[str] = []
    count = 0
    in_edges = False
    for line in text.splitlines():
        if line.startswith("=="):
            in_edges = line.startswith("== callers") or line.startswith("== callees")
            count = 0
            out.append(line)
            continue
        if in_edges and line.strip():
            count += 1
            if count <= n:
                out.append(line)
            elif count == n + 1:
                out.append("  … (more rows; raise n)")
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Tools (registered on the shared FastMCP instance by register())
# ---------------------------------------------------------------------------


async def load_igprof(
    url: str,
    name: str = "",
    counter: str = "MEM_TOTAL",
    sidecar_url: str = "",
    proxy_token: str = "",
) -> str:
    """Fetch an IgProf heap dump and register it for querying.

    The ``.gz`` dump is downloaded (via the alimonitor proxy for
    ``alimonitor.cern.ch`` URLs), decompressed once, and cached on disk;
    subsequent tools re-read that file. No in-memory index.

    Args:
        url:         Direct URL to an ``igprof.<device>.<...>.gz`` dump.
        name:        Label (defaults to the filename portion of the URL).
        counter:     Default counter for this report (MEM_TOTAL/MEM_MAX/MEM_LIVE).
        sidecar_url: Optional ``igprof.*.syms.gz`` resolving ``@?0x…`` addresses.
        proxy_token: Bearer token for the local proxy (else PROXY_TOKEN env).
    """
    raw = await fetch_bytes(url, proxy_token=proxy_token)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    h = hashlib.sha1(url.encode()).hexdigest()[:12]
    dump_path = os.path.join(_CACHE_DIR, f"{h}.dump")
    data = gzip.decompress(raw) if (url.endswith(".gz") or raw[:2] == b"\x1f\x8b") else raw
    with open(dump_path, "wb") as f:
        f.write(data)

    sidecar_path = ""
    if sidecar_url:
        sc = await fetch_bytes(sidecar_url, proxy_token=proxy_token)
        sidecar_path = os.path.join(_CACHE_DIR, f"{h}.syms.gz")
        with open(sidecar_path, "wb") as f:
            f.write(sc)

    counters = _enumerate_counters(dump_path)
    if counters and counter not in counters:
        counter = counters[0]

    pname = name or url.rstrip("/").split("/")[-1]
    report = IgProfReport(url, pname, dump_path, sidecar_path, counters, counter)
    _reports[pname] = report

    nsym = ""
    try:
        _, err = _run(report, ["top", "-k", counter, "-n", "0"])
        m = re.search(r"symbols=(\d+)", err)
        if m:
            nsym = f", {int(m.group(1)):,} symbols"
    except Exception:
        pass

    return (
        f"Loaded igprof '{pname}': {len(data):,} bytes uncompressed{nsym}. "
        f"counters={counters or '(none detected)'}, default={counter}"
        + (", side-car attached" if sidecar_path else "")
    )


def list_igprof() -> str:
    """List loaded IgProf reports."""
    if not _reports:
        return "No igprof reports loaded. Use load_igprof first."
    return "\n".join(
        f"{n}: default={r.default_counter}, counters={r.counters}, url={r.url}"
        for n, r in _reports.items()
    )


def drop_igprof(name: str) -> str:
    """Free a report and delete its cached dump.

    Args:
        name: Report name as returned by load_igprof.
    """
    r = _get(name)
    for p in (r.dump_path, r.sidecar_path):
        if p and os.path.exists(p):
            os.remove(p)
    del _reports[name]
    return f"Dropped igprof report '{name}'."


def igprof_counters(name: str) -> str:
    """List the counters available in a report and what they mean.

    Args:
        name: Report name as returned by load_igprof.
    """
    r = _get(name)
    return "\n".join(
        f"{c}: {_COUNTER_DOC.get(c, 'profiler counter')}"
        + ("  (default)" if c == r.default_counter else "")
        for c in r.counters
    )


def igprof_top(name: str, counter: str = "", n: int = 40) -> str:
    """Top allocators by a counter (cumulative + self, already merged by name).

    Args:
        name:    Report name as returned by load_igprof.
        counter: MEM_TOTAL/MEM_MAX/MEM_LIVE (defaults to the report's default).
        n:       Number of rows (default 40).
    """
    r = _get(name)
    out, _ = _run(r, ["top", "-k", counter or r.default_counter, "-n", str(n)])
    return out


def igprof_show(name: str, symbol: str, counter: str = "", n: int = 40) -> str:
    """Callers and callees of a symbol (POSIX-extended regex), merged by name.

    Args:
        name:    Report name as returned by load_igprof.
        symbol:  Regex matched against the (resolved) symbol name, e.g. ``^_Znwm$``.
        counter: MEM_TOTAL/MEM_MAX/MEM_LIVE (defaults to the report's default).
        n:       Max caller/callee rows to show per side (default 40).
    """
    r = _get(name)
    out, _ = _run(r, ["show", "-s", symbol, "-k", counter or r.default_counter])
    return _limit_show(out, n)


def igprof_show_rank(name: str, rank: int, counter: str = "", n: int = 40) -> str:
    """Drill into the RANK-th heaviest symbol (by `igprof_top`) — callers + callees.

    Args:
        name:    Report name as returned by load_igprof.
        rank:    1-based rank in the `igprof_top` ranking for `counter`.
        counter: MEM_TOTAL/MEM_MAX/MEM_LIVE (defaults to the report's default).
        n:       Max caller/callee rows to show per side (default 40).
    """
    r = _get(name)
    out, _ = _run(r, ["show", "-r", str(rank), "-k", counter or r.default_counter])
    return _limit_show(out, n)


def igprof_compare(name_a: str, name_b: str, counter: str = "", n: int = 40) -> str:
    """Diff two reports' allocators, normalised to each report's total `self`.

    Positive Δ means the symbol takes a larger share of allocations in B than A.

    Args:
        name_a:  Baseline report name.
        name_b:  Comparison report name.
        counter: Counter to compare (defaults to A's default).
        n:       Number of rows (default 40).
    """
    a, b = _get(name_a), _get(name_b)
    c = counter or a.default_counter
    ta, _ = _run(a, ["top", "-k", c, "-n", "100000"])
    tb, _ = _run(b, ["top", "-k", c, "-n", "100000"])
    ra, rb = _parse_top(ta), _parse_top(tb)
    sa = sum(v[1] for v in ra.values()) or 1
    sb = sum(v[1] for v in rb.values()) or 1
    diffs = []
    for sym in set(ra) | set(rb):
        fa = ra.get(sym, (0, 0, 0))[1] / sa
        fb = rb.get(sym, (0, 0, 0))[1] / sb
        diffs.append((fb - fa, sym, fa, fb))
    diffs.sort(key=lambda x: -abs(x[0]))
    lines = [
        f"Comparing '{name_a}' (A) vs '{name_b}' (B)  counter={c}, self-share",
        f"{'Δ%':>8}  {'A%':>7}  {'B%':>7}  symbol",
    ]
    for d, sym, fa, fb in diffs[:n]:
        lines.append(f"{d*100:>+8.2f}  {fa*100:>7.2f}  {fb*100:>7.2f}  {sym}")
    return "\n".join(lines)


def register(mcp) -> None:
    """Register the igprof tools on a shared FastMCP instance."""
    for fn in (
        load_igprof,
        list_igprof,
        drop_igprof,
        igprof_counters,
        igprof_top,
        igprof_show,
        igprof_show_rank,
        igprof_compare,
    ):
        mcp.tool()(fn)
