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

import httpx


async def fetch_bytes(url: str, proxy_token: str = "", token: str = "") -> bytes:
    """Fetch a workdir artefact, routing alimonitor URLs through the local proxy.

    Mirrors the grid-cert proxy convention used across the Hyperloop tooling:
    ``alimonitor.cern.ch/<path>`` is rewritten to
    ``http://localhost:8888/alimonitor/<path>`` with a bearer token, and
    ``Accept-Encoding: identity`` is required (otherwise the proxy returns a gzip
    Content-Length mismatch). Retries transient protocol/read errors up to 3×.

    Args:
        url:         Direct artefact URL (perf script, igprof dump, side-car, ...).
        proxy_token: Bearer token for the local proxy. Falls back to PROXY_TOKEN,
                     then HYPERLOOP_TOKEN, then ``token``.
        token:       Hyperloop auth token fallback.
    """
    proxy_token = (
        proxy_token
        or os.environ.get("PROXY_TOKEN", "")
        or token
        or os.environ.get("HYPERLOOP_TOKEN", "")
    )

    fetch_url = url
    if "alimonitor.cern.ch" in url:
        path = url.split("alimonitor.cern.ch", 1)[1].lstrip("/")
        fetch_url = f"http://localhost:8888/alimonitor/{path}"

    headers = {"Authorization": f"Bearer {proxy_token}"} if proxy_token else {}
    headers["Accept-Encoding"] = "identity"

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
