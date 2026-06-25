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
"""Shared helpers for the Hyperloop perf / igprof MCP tools."""

from __future__ import annotations

import json
import os
import socket
import time

import httpx

# security-proxy (see ~/src/ali-bot/security-proxy): a localhost credential proxy
# that binds a RANDOM port and mints a per-service, daily-rotating gate token,
# both handed out over a per-user UNIX socket. Replaces the old fixed
# localhost:8888 + static-bearer ccdb-proxy. Every alimonitor.cern.ch artefact is
# routed through the "/alimonitor/" route (upstream = alimonitor.cern.ch root), so a
# single "alimonitor" gate token covers train-workdir / hyperloop / alihyperloop-data.
_AGENT_SOCK = os.path.expanduser(
    os.environ.get("SECURITY_PROXY_AGENT_SOCK", "~/.security-proxy/agent.sock")
)
_PROXY_SERVICE = os.environ.get("SECURITY_PROXY_SERVICE", "alimonitor")
_creds_cache: dict[str, tuple[int, str, float]] = {}


def _proxy_creds(service: str) -> tuple[int, str]:
    """Return (port, gate_token) for ``service`` from the security-proxy agent socket.

    Cached ~5 min; the proxy accepts the current and previous token, so a slightly
    stale cached token still works across the daily rotation. Raises with a clear
    hint if the proxy isn't running.
    """
    now = time.time()
    hit = _creds_cache.get(service)
    if hit and now - hit[2] < 300:
        return hit[0], hit[1]
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(_AGENT_SOCK)
        s.sendall((service + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        s.close()
        data = json.loads(buf.decode())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"security-proxy agent not reachable at {_AGENT_SOCK} ({exc}); "
            "is the proxy running? (see ~/src/ali-bot/security-proxy)"
        ) from exc
    if "error" in data:
        raise RuntimeError(
            f"security-proxy: {data['error']}; known services: {data.get('services', [])}"
        )
    port, token = int(data["port"]), data.get("token", "")
    _creds_cache[service] = (port, token, now)
    return port, token


async def fetch_bytes(url: str, proxy_token: str = "", token: str = "") -> bytes:
    """Fetch a workdir artefact, routing alimonitor URLs through the security-proxy.

    ``alimonitor.cern.ch/<path>`` is rewritten to
    ``http://127.0.0.1:<port>/alimonitor/<path>``: the random port and a per-service,
    daily-rotating gate token come from the security-proxy agent socket
    (``~/.security-proxy/agent.sock``; override with ``SECURITY_PROXY_AGENT_SOCK``),
    and the token is sent as ``Authorization: Bearer``. ``Accept-Encoding: identity``
    is required (otherwise the proxy returns a gzip Content-Length mismatch). Retries
    transient protocol/read errors up to 3×.

    Args:
        url:         Direct artefact URL, a local path, or a ``file://`` URL.
        proxy_token: Accepted for backward compatibility but ignored — the gate token
                     is minted from the agent socket.
        token:       Ditto (ignored).
    """
    # Local file (a path or a file:// URL) — read directly, no HTTP. Lets a
    # locally-generated side-car (igprof-demangle-symbols output) be attached
    # via load_igprof(sidecar_url=/path/to/...syms.gz) without a web server.
    if url.startswith("file://") or os.path.isfile(url):
        path = url[len("file://"):] if url.startswith("file://") else url
        with open(path, "rb") as f:
            return f.read()

    fetch_url = url
    headers = {"Accept-Encoding": "identity"}
    if "alimonitor.cern.ch" in url:
        path = url.split("alimonitor.cern.ch", 1)[1].lstrip("/")
        port, gate = _proxy_creds(_PROXY_SERVICE)
        fetch_url = f"http://127.0.0.1:{port}/{_PROXY_SERVICE}/{path}"
        if gate:
            headers["Authorization"] = f"Bearer {gate}"

    async with httpx.AsyncClient(verify=False) as client:
        for attempt in range(3):
            try:
                r = await client.get(
                    fetch_url, headers=headers, timeout=300.0, follow_redirects=True
                )
                r.raise_for_status()
                return r.content
            except (httpx.RemoteProtocolError, httpx.ReadError):
                if attempt == 2:
                    raise
    raise RuntimeError("unreachable")
