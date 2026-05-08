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
"""DPL status MCP server.

Bridges the DPL driver /status WebSocket endpoint to MCP tools so that an
AI assistant (e.g. Claude) can inspect and monitor a running DPL workflow.

Usage
-----
    python3 dpl_mcp_server.py --port 8080
    python3 dpl_mcp_server.py --pid 12345       # port derived as 8080 + pid % 30000
    DPL_STATUS_PORT=8080 python3 dpl_mcp_server.py

Wire protocol (client → driver)
--------------------------------
    {"cmd":"list_metrics","device":"<name>"}
    {"cmd":"subscribe","device":"<name>","metrics":["m1","m2"]}
    {"cmd":"unsubscribe","device":"<name>","metrics":["m1"]}

Wire protocol (driver → client)
--------------------------------
    {"type":"snapshot","devices":[{"name","pid","active","streamingState","deviceState"},...]}
    {"type":"update","device":<idx>,"name":"<name>","metrics":{<name:value,...>}}
    {"type":"metrics_list","device":"<name>","metrics":["m1","m2",...]}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import websockets
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Global connection state (all access from the single asyncio event loop)
# ---------------------------------------------------------------------------
_port: int = 8080
_ws: Any = None
_reader_task: asyncio.Task | None = None
_snapshot: dict = {}
_updates: list[dict] = []
_logs: list[dict] = []
_metrics_lists: dict[str, list[str]] = {}


async def _ensure_connected() -> None:
    """Connect (or reconnect) to the driver's /status WebSocket."""
    global _ws, _reader_task

    # Check liveness of existing connection.
    if _ws is not None:
        try:
            pong = await asyncio.wait_for(_ws.ping(), timeout=2.0)
            await pong
            return
        except Exception:
            _ws = None
            if _reader_task is not None and not _reader_task.done():
                _reader_task.cancel()
            _reader_task = None

    url = f"ws://localhost:{_port}/status"
    _ws = await websockets.connect(url, subprotocols=["dpl"])
    if _reader_task is None or _reader_task.done():
        _reader_task = asyncio.create_task(_reader())


async def _reader() -> None:
    """Background task: read frames from the driver and buffer them."""
    global _ws, _snapshot, _updates, _logs, _metrics_lists
    try:
        async for raw in _ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "snapshot":
                _snapshot = msg
                # Clear stale metric lists from a previous driver instance.
                _metrics_lists.clear()
            elif t == "update":
                _updates.append(msg)
            elif t == "log":
                _logs.append(msg)
            elif t == "metrics_list":
                device = msg.get("device", "")
                _metrics_lists[device] = msg.get("metrics", [])
    except Exception:
        pass
    finally:
        _ws = None


async def _send(obj: dict) -> None:
    await _ensure_connected()
    await _ws.send(json.dumps(obj, separators=(",", ":")))


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------
mcp = FastMCP("DPL Status")


@mcp.tool()
async def list_devices() -> str:
    """List all DPL devices with their current status.

    Returns each device's name, PID, active flag, streaming state, and device
    state as reported by the driver snapshot.
    """
    await _ensure_connected()
    if not _snapshot:
        return "No snapshot received yet — the driver may still be starting."
    devices = _snapshot.get("devices", [])
    if not devices:
        return "No devices in snapshot."
    lines = []
    for d in devices:
        lines.append(
            f"{d['name']}: pid={d['pid']} active={d['active']} "
            f"streaming={d['streamingState']} state={d['deviceState']}"
        )
    return "\n".join(lines)


@mcp.tool()
async def list_metrics(device: str) -> str:
    """List the available numeric metrics for a DPL device.

    Sends a list_metrics command to the driver and waits up to 3 seconds for
    the reply.  Only numeric metrics (int, float, uint64) are included; string
    and enum metrics are excluded.

    Args:
        device: Device name exactly as shown by list_devices.
    """
    # Remove any stale cached result so we can detect the fresh reply.
    _metrics_lists.pop(device, None)
    await _send({"cmd": "list_metrics", "device": device})
    for _ in range(60):          # up to 3 s
        await asyncio.sleep(0.05)
        if device in _metrics_lists:
            names = _metrics_lists[device]
            if not names:
                return f"Device '{device}' has no numeric metrics yet."
            return f"{len(names)} metric(s): " + ", ".join(names)
    return f"No reply from driver for device '{device}' (timeout)."


@mcp.tool()
async def subscribe(device: str, metrics: list[str]) -> str:
    """Subscribe to one or more metrics for a DPL device.

    After subscribing, the driver will push update frames for the device
    whenever any of the subscribed metrics change.  Use get_updates to drain
    the buffer.

    Args:
        device: Device name exactly as shown by list_devices.
        metrics: List of metric names to subscribe to (from list_metrics).
    """
    await _send({"cmd": "subscribe", "device": device, "metrics": metrics})
    return f"Subscribed to {len(metrics)} metric(s) for '{device}': {', '.join(metrics)}"


@mcp.tool()
async def unsubscribe(device: str, metrics: list[str]) -> str:
    """Stop receiving updates for specific metrics of a DPL device.

    Args:
        device: Device name exactly as shown by list_devices.
        metrics: List of metric names to unsubscribe from.
    """
    await _send({"cmd": "unsubscribe", "device": device, "metrics": metrics})
    return f"Unsubscribed from {len(metrics)} metric(s) for '{device}'."


@mcp.tool()
async def subscribe_logs(device: str) -> str:
    """Subscribe to log output for a DPL device.

    After subscribing, new log lines from the device will be buffered and
    can be retrieved with get_logs().

    Args:
        device: Device name exactly as shown by list_devices.
    """
    await _send({"cmd": "subscribe_logs", "device": device})
    return f"Subscribed to logs for '{device}'."


@mcp.tool()
async def unsubscribe_logs(device: str) -> str:
    """Stop receiving log output for a DPL device.

    Args:
        device: Device name exactly as shown by list_devices.
    """
    await _send({"cmd": "unsubscribe_logs", "device": device})
    return f"Unsubscribed from logs for '{device}'."


@mcp.tool()
async def get_logs(max_lines: int = 100) -> str:
    """Drain and return buffered log lines received since the last call.

    Args:
        max_lines: Maximum number of log lines to return (default 100).
    """
    await _ensure_connected()
    batch = _logs[:max_lines]
    del _logs[:max_lines]
    if not batch:
        return "No buffered log lines."
    lines = []
    for entry in batch:
        device = entry.get("device", "?")
        level = entry.get("level", "?")
        line = entry.get("line", "")
        lines.append(f"[{device}][{level}] {line}")
    return "\n".join(lines)


@mcp.tool()
async def start_devices() -> str:
    """Resume all stopped DPL devices (send SIGCONT).

    Use this when the workflow was started with -s (all devices paused).
    """
    await _send({"cmd": "start_devices"})
    return "Sent SIGCONT to all active devices."


@mcp.tool()
async def enable_signpost(device: str, streams: list[str]) -> str:
    """Enable one or more signpost log streams for a DPL device.

    Signpost streams produce detailed trace output visible in the device logs.
    Use get_logs() after subscribing to see the output.

    Known stream names (full form): ch.cern.aliceo2.device,
    ch.cern.aliceo2.completion, ch.cern.aliceo2.monitoring_service,
    ch.cern.aliceo2.data_processor_context, ch.cern.aliceo2.stream_context.

    Args:
        device: Device name as shown by list_devices, or "" for the driver.
        streams: List of full signpost log names to enable.
    """
    await _send({"cmd": "enable_signpost", "device": device, "streams": streams})
    return f"Enabled {len(streams)} signpost stream(s) for '{device or 'driver'}': {', '.join(streams)}"


@mcp.tool()
async def disable_signpost(device: str, streams: list[str]) -> str:
    """Disable one or more signpost log streams for a DPL device.

    Args:
        device: Device name as shown by list_devices, or "" for the driver.
        streams: List of full signpost log names to disable.
    """
    await _send({"cmd": "disable_signpost", "device": device, "streams": streams})
    return f"Disabled {len(streams)} signpost stream(s) for '{device or 'driver'}': {', '.join(streams)}"


@mcp.tool()
async def get_updates(max_updates: int = 50) -> str:
    """Drain and return buffered metric update frames received since the last call.

    Each frame contains the latest values of all subscribed metrics that
    changed during that processing cycle.  Calling this repeatedly gives a
    time-ordered view of metric evolution.

    Args:
        max_updates: Maximum number of update frames to return (default 50).
    """
    await _ensure_connected()
    batch = _updates[:max_updates]
    del _updates[:max_updates]
    if not batch:
        return "No buffered updates."
    lines = []
    for upd in batch:
        name = upd.get("name") or f"device[{upd.get('device', '?')}]"
        metrics = upd.get("metrics", {})
        if metrics:
            parts = ", ".join(f"{k}={v}" for k, v in metrics.items())
            lines.append(f"{name}: {parts}")
        else:
            lines.append(f"{name}: (empty update)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    global _port

    parser = argparse.ArgumentParser(
        description="DPL status MCP server — expose DPL driver metrics via MCP tools"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port of the DPL driver status WebSocket (default: 8080 or DPL_STATUS_PORT env var)",
    )
    group.add_argument(
        "--pid",
        type=int,
        default=None,
        help="PID of the DPL driver process; port is derived as 8080 + pid %% 30000",
    )
    args = parser.parse_args()

    if args.pid is not None:
        _port = 8080 + args.pid % 30000
    elif args.port is not None:
        _port = args.port
    elif "DPL_STATUS_PORT" in os.environ:
        _port = int(os.environ["DPL_STATUS_PORT"])
    # else leave _port at the default 8080

    mcp.run()


if __name__ == "__main__":
    main()
