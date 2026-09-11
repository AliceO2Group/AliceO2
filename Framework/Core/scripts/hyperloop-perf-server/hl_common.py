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

import os
import sys

import httpx

# The security-proxy client is shared with the sibling MCP servers; it lives one
# directory up so all of them import the same copy (it used to be duplicated, and
# the copies drifted). See security_proxy_client.__doc__.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import security_proxy_client as _spc  # noqa: E402

_AGENT_SOCK = _spc.AGENT_SOCK
_PROXY_SERVICE = _spc.DEFAULT_SERVICE
_proxy_creds = _spc.proxy_creds


async def fetch_bytes(url: str, proxy_token: str = "", token: str = "") -> bytes:
    """Fetch a workdir artefact, routing alimonitor URLs through the security-proxy.

    ``alimonitor.cern.ch/<path>`` is rewritten to
    ``http://127.0.0.1:<port>/alimonitor/<path>``: the random port and a per-service,
    daily-rotating gate token come from the security-proxy agent socket
    (resolved by ``security_proxy_client``; override with ``SECURITY_PROXY_AGENT_SOCK``),
    and the token is sent as ``Authorization: Bearer``. ``Accept-Encoding: identity``
    is required (otherwise the proxy returns a gzip Content-Length mismatch). Retries
    transient protocol/read errors up to 3×.

    Args:
        url:         Direct artefact URL, a local path, or a ``file://`` URL.
        proxy_token: Gate token to use when ``url`` ALREADY points at the security-proxy
                     (``http://127.0.0.1:<port>/<service>/...``), which carries no
                     ``alimonitor.cern.ch`` host to trigger the rewrite above. Ignored
                     for alimonitor URLs, where the token is minted from the agent socket.
        token:       Fallback for ``proxy_token``.
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
    elif url.startswith(("http://127.0.0.1:", "http://localhost:")):
        # Already a security-proxy URL (pasted from a browser, or built by a caller
        # that resolved the random port itself). The rewrite above does not fire, but
        # the proxy still demands the gate token — without it every route answers 401.
        gate = proxy_token or token
        if not gate:
            try:
                gate = _proxy_creds(_PROXY_SERVICE)[1]
            except RuntimeError:
                gate = ""
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
