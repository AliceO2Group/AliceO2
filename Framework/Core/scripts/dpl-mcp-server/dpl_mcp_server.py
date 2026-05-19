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

Supports multiple concurrent workflows.  Use the ``connect`` tool to attach
to a running topology by port or PID, then pass the returned workflow name
to every other tool.

Usage
-----
    python3 dpl_mcp_server.py

Wire protocol (client -> driver)
--------------------------------
    {"cmd":"list_metrics","device":"<name>"}
    {"cmd":"subscribe","device":"<name>","metrics":["m1","m2"]}
    {"cmd":"unsubscribe","device":"<name>","metrics":["m1"]}

Wire protocol (driver -> client)
--------------------------------
    {"type":"snapshot","devices":[{"name","pid","active","streamingState","deviceState"},...]}
    {"type":"update","device":<idx>,"name":"<name>","metrics":{<name:value,...>}}
    {"type":"metrics_list","device":"<name>","metrics":["m1","m2",...]}
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets
from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Per-workflow connection state
# ---------------------------------------------------------------------------
class WorkflowConnection:
    """Holds WebSocket connection and buffered state for one DPL workflow."""

    def __init__(self, port: int, name: str):
        self.port = port
        self.name = name
        self.ws: Any = None
        self.reader_task: asyncio.Task | None = None
        self.snapshot: dict = {}
        self.updates: list[dict] = []
        self.logs: list[dict] = []
        self.metrics_lists: dict[str, list[str]] = {}

    async def ensure_connected(self) -> None:
        """Connect (or reconnect) to the driver's /status WebSocket."""
        if self.ws is not None:
            try:
                pong = await asyncio.wait_for(self.ws.ping(), timeout=2.0)
                await pong
                return
            except Exception:
                old_ws = self.ws
                self.ws = None
                if self.reader_task is not None and not self.reader_task.done():
                    self.reader_task.cancel()
                    try:
                        await self.reader_task
                    except (asyncio.CancelledError, Exception):
                        pass
                self.reader_task = None
                try:
                    await old_ws.close()
                except Exception:
                    pass

        url = f"ws://localhost:{self.port}/status"
        self.ws = await websockets.connect(url, subprotocols=["dpl"])
        if self.reader_task is None or self.reader_task.done():
            self.reader_task = asyncio.create_task(self._reader())

    async def _reader(self) -> None:
        """Background task: read frames from the driver and buffer them."""
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type")
                if t == "snapshot":
                    self.snapshot = msg
                    self.metrics_lists.clear()
                elif t == "update":
                    self.updates.append(msg)
                elif t == "log":
                    self.logs.append(msg)
                elif t == "metrics_list":
                    device = msg.get("device", "")
                    self.metrics_lists[device] = msg.get("metrics", [])
        except Exception:
            pass
        finally:
            self.ws = None

    async def send(self, obj: dict) -> None:
        await self.ensure_connected()
        await self.ws.send(json.dumps(obj, separators=(",", ":")))

    async def close(self) -> None:
        ws = self.ws
        self.ws = None
        if self.reader_task is not None and not self.reader_task.done():
            self.reader_task.cancel()
            try:
                await self.reader_task
            except (asyncio.CancelledError, Exception):
                pass
        self.reader_task = None
        if ws is not None:
            await ws.close()


# ---------------------------------------------------------------------------
# Workflow registry
# ---------------------------------------------------------------------------
_workflows: dict[str, WorkflowConnection] = {}


def _get(workflow: str) -> WorkflowConnection:
    """Look up a workflow by name, raising a clear error if not found."""
    conn = _workflows.get(workflow)
    if conn is None:
        available = ", ".join(_workflows.keys()) if _workflows else "(none)"
        raise ValueError(
            f"No workflow named '{workflow}'. Connected workflows: {available}. "
            f"Use the connect tool first."
        )
    return conn


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------
mcp = FastMCP("DPL Status")


@mcp.tool()
async def connect(port: int = 0, pid: int = 0, name: str = "") -> str:
    """Connect to a running DPL workflow.

    Provide either ``port`` (the driver's WebSocket port) or ``pid`` (the
    driver PID, port derived as 8080 + pid % 30000).  An optional ``name``
    gives the workflow a human-friendly label; if omitted the port number is
    used.

    Args:
        port: TCP port of the DPL driver status WebSocket.
        pid:  PID of the DPL driver process (alternative to port).
        name: Optional human-friendly name for this workflow.
    """
    if pid:
        port = 8080 + pid % 30000
    if not port:
        return "Provide either port or pid."

    wf_name = name or str(port)
    if wf_name in _workflows:
        old = _workflows[wf_name]
        await old.close()

    conn = WorkflowConnection(port, wf_name)
    await conn.ensure_connected()
    _workflows[wf_name] = conn

    devices = conn.snapshot.get("devices", [])
    return (
        f"Connected to workflow '{wf_name}' on port {port} "
        f"({len(devices)} device(s))."
    )


@mcp.tool()
async def disconnect(workflow: str) -> str:
    """Disconnect from a DPL workflow and release its resources.

    Args:
        workflow: Workflow name as returned by connect.
    """
    conn = _get(workflow)
    await conn.close()
    del _workflows[workflow]
    return f"Disconnected from workflow '{workflow}'."


@mcp.tool()
async def list_workflows() -> str:
    """List all currently connected DPL workflows."""
    if not _workflows:
        return "No workflows connected. Use the connect tool first."
    lines = []
    for wf_name, conn in _workflows.items():
        n = len(conn.snapshot.get("devices", []))
        status = "connected" if conn.ws is not None else "disconnected"
        lines.append(f"{wf_name}: port={conn.port} devices={n} status={status}")
    return "\n".join(lines)


@mcp.tool()
async def list_devices(workflow: str) -> str:
    """List all DPL devices with their current status.

    Returns each device's name, PID, active flag, streaming state, and device
    state as reported by the driver snapshot.

    Args:
        workflow: Workflow name as returned by connect.
    """
    conn = _get(workflow)
    await conn.ensure_connected()
    if not conn.snapshot:
        return "No snapshot received yet -- the driver may still be starting."
    devices = conn.snapshot.get("devices", [])
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
async def list_metrics(workflow: str, device: str) -> str:
    """List the available numeric metrics for a DPL device.

    Sends a list_metrics command to the driver and waits up to 3 seconds for
    the reply.  Only numeric metrics (int, float, uint64) are included; string
    and enum metrics are excluded.

    Args:
        workflow: Workflow name as returned by connect.
        device: Device name exactly as shown by list_devices.
    """
    conn = _get(workflow)
    conn.metrics_lists.pop(device, None)
    await conn.send({"cmd": "list_metrics", "device": device})
    for _ in range(60):          # up to 3 s
        await asyncio.sleep(0.05)
        if device in conn.metrics_lists:
            names = conn.metrics_lists[device]
            if not names:
                return f"Device '{device}' has no numeric metrics yet."
            return f"{len(names)} metric(s): " + ", ".join(names)
    return f"No reply from driver for device '{device}' (timeout)."


@mcp.tool()
async def subscribe(workflow: str, device: str, metrics: list[str]) -> str:
    """Subscribe to one or more metrics for a DPL device.

    After subscribing, the driver will push update frames for the device
    whenever any of the subscribed metrics change.  Use get_updates to drain
    the buffer.

    Args:
        workflow: Workflow name as returned by connect.
        device: Device name exactly as shown by list_devices.
        metrics: List of metric names to subscribe to (from list_metrics).
    """
    conn = _get(workflow)
    await conn.send({"cmd": "subscribe", "device": device, "metrics": metrics})
    return f"Subscribed to {len(metrics)} metric(s) for '{device}': {', '.join(metrics)}"


@mcp.tool()
async def unsubscribe(workflow: str, device: str, metrics: list[str]) -> str:
    """Stop receiving updates for specific metrics of a DPL device.

    Args:
        workflow: Workflow name as returned by connect.
        device: Device name exactly as shown by list_devices.
        metrics: List of metric names to unsubscribe from.
    """
    conn = _get(workflow)
    await conn.send({"cmd": "unsubscribe", "device": device, "metrics": metrics})
    return f"Unsubscribed from {len(metrics)} metric(s) for '{device}'."


@mcp.tool()
async def subscribe_logs(workflow: str, device: str) -> str:
    """Subscribe to log output for a DPL device.

    After subscribing, new log lines from the device will be buffered and
    can be retrieved with get_logs().

    Args:
        workflow: Workflow name as returned by connect.
        device: Device name exactly as shown by list_devices.
    """
    conn = _get(workflow)
    await conn.send({"cmd": "subscribe_logs", "device": device})
    return f"Subscribed to logs for '{device}'."


@mcp.tool()
async def unsubscribe_logs(workflow: str, device: str) -> str:
    """Stop receiving log output for a DPL device.

    Args:
        workflow: Workflow name as returned by connect.
        device: Device name exactly as shown by list_devices.
    """
    conn = _get(workflow)
    await conn.send({"cmd": "unsubscribe_logs", "device": device})
    return f"Unsubscribed from logs for '{device}'."


@mcp.tool()
async def get_logs(workflow: str, max_lines: int = 100) -> str:
    """Drain and return buffered log lines received since the last call.

    Args:
        workflow: Workflow name as returned by connect.
        max_lines: Maximum number of log lines to return (default 100).
    """
    conn = _get(workflow)
    await conn.ensure_connected()
    batch = conn.logs[:max_lines]
    del conn.logs[:max_lines]
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
async def start_devices(workflow: str) -> str:
    """Resume all stopped DPL devices (send SIGCONT).

    Use this when the workflow was started with -s (all devices paused).

    Args:
        workflow: Workflow name as returned by connect.
    """
    conn = _get(workflow)
    await conn.send({"cmd": "start_devices"})
    return "Sent SIGCONT to all active devices."


@mcp.tool()
async def enable_signpost(workflow: str, device: str, streams: list[str]) -> str:
    """Enable one or more signpost log streams for a DPL device.

    Signpost streams produce detailed trace output visible in the device logs.
    Use get_logs() after subscribing to see the output.

    Known stream names (full form): ch.cern.aliceo2.device,
    ch.cern.aliceo2.completion, ch.cern.aliceo2.monitoring_service,
    ch.cern.aliceo2.data_processor_context, ch.cern.aliceo2.stream_context.

    Args:
        workflow: Workflow name as returned by connect.
        device: Device name as shown by list_devices, or "" for the driver.
        streams: List of full signpost log names to enable.
    """
    conn = _get(workflow)
    await conn.send({"cmd": "enable_signpost", "device": device, "streams": streams})
    return f"Enabled {len(streams)} signpost stream(s) for '{device or 'driver'}': {', '.join(streams)}"


@mcp.tool()
async def disable_signpost(workflow: str, device: str, streams: list[str]) -> str:
    """Disable one or more signpost log streams for a DPL device.

    Args:
        workflow: Workflow name as returned by connect.
        device: Device name as shown by list_devices, or "" for the driver.
        streams: List of full signpost log names to disable.
    """
    conn = _get(workflow)
    await conn.send({"cmd": "disable_signpost", "device": device, "streams": streams})
    return f"Disabled {len(streams)} signpost stream(s) for '{device or 'driver'}': {', '.join(streams)}"


@mcp.tool()
async def get_updates(workflow: str, max_updates: int = 50) -> str:
    """Drain and return buffered metric update frames received since the last call.

    Each frame contains the latest values of all subscribed metrics that
    changed during that processing cycle.  Calling this repeatedly gives a
    time-ordered view of metric evolution.

    Args:
        workflow: Workflow name as returned by connect.
        max_updates: Maximum number of update frames to return (default 50).
    """
    conn = _get(workflow)
    await conn.ensure_connected()
    batch = conn.updates[:max_updates]
    del conn.updates[:max_updates]
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
    mcp.run()


if __name__ == "__main__":
    main()
