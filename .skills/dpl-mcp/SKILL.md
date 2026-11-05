---
name: dpl-mcp
description: Inspect a running DPL workflow through the DPL MCP server (Framework/Core/scripts/dpl-mcp-server) — devices and their state, live metrics, logs, signposts, per-mapping memory (smaps), resuming a workflow started with -s. Use when asked to look at, debug or measure a workflow while it runs, locally or on Hyperloop.
---

# DPL MCP server

`Framework/Core/scripts/dpl-mcp-server/dpl_mcp_server.py` is an MCP server
(`FastMCP("DPL Status")`) that talks to the `/status` WebSocket of a running
DPL **driver**. Everything it reports comes from the driver, which already
holds the devices' metrics, logs and pids, so nothing has to be enabled in the
workflow: every driver listens, `-b` or not.

The driver side is `Framework/Core/src/StatusWebSocketHandler.{h,cxx}`; the
header documents the wire protocol. A tool and its driver command are added
together: a command the driver does not know is silently ignored, which the
tool reports as a timeout.

## Setup

Dependencies are `mcp` and `websockets` (`pyproject.toml`, Python ≥ 3.11).
Use a virtualenv next to the script and register the script itself; the server
takes no command-line arguments, workflows are attached at run time with
`connect`:

```sh
cd Framework/Core/scripts/dpl-mcp-server
python3 -m venv .venv && .venv/bin/pip install mcp websockets
claude mcp add dpl -- "$PWD/.venv/bin/python3" "$PWD/dpl_mcp_server.py"
```

Several workflows can be connected at once; each tool takes the `workflow`
name that `connect` returned.

## Attaching to a workflow

- **Local**: `connect(pid=<driver pid>)` or `connect(port=<port>)`.
  The driver binds `8080 + pid % 30000` (or `DPL_REMOTE_GUI_PORT`), but moves
  to the next free port if that one is taken, so when `pid=` fails take the
  port from the driver log: `Driver listening on port: <n>`. The driver pid is
  the one without `--id` on its command line.
- **Hyperloop**: `connect_hyperloop(url=https://alimonitor.cern.ch/train-workdir/remote-gui/remote_proxy.html?<token>/<port>)`
  for a test that exposes the remote GUI. NOTE: this still targets the old
  `ws://localhost:8888` proxy with `HYPERLOOP_TOKEN`; it has to be moved to
  the security-proxy before it works again.

## Tools

| Tool | What it does |
|------|--------------|
| `connect`, `connect_hyperloop`, `disconnect`, `list_workflows` | attach / detach workflows |
| `list_devices(workflow)` | name, pid, active, streaming state, device state of every device |
| `list_metrics(workflow, device)` | numeric metrics a device publishes (string/enum ones are left out) |
| `subscribe` / `unsubscribe(workflow, device, metrics)` | the driver pushes a subscribed metric each cycle in which it changed |
| `get_updates(workflow, max_updates)` | drain the buffered metric updates, time-ordered |
| `subscribe_logs` / `unsubscribe_logs(workflow, device)` | stream a device's log lines |
| `get_logs(workflow, max_lines)` | drain the buffered log lines |
| `enable_signpost` / `disable_signpost(workflow, device, streams)` | switch signpost streams on a device (`device=""` for the driver), e.g. `ch.cern.aliceo2.device`, `ch.cern.aliceo2.completion`, `ch.cern.aliceo2.monitoring_service`, `ch.cern.aliceo2.data_processor_context`, `ch.cern.aliceo2.stream_context`; the output arrives in the logs |
| `start_devices(workflow)` | SIGCONT every device of a workflow started with `-s` (stop before device start) |
| `smaps(workflow, device, top)` | memory per mapping, read by the driver from `/proc/<pid>/smaps` (Linux only) |

Updates and logs are buffered by the server between calls and drained by
`get_updates` / `get_logs`, so poll them rather than expecting a push.

## Memory per mapping (`smaps`)

The driver reads `/proc/<pid>/smaps` of the device and returns, per path, kB of
rss, pss, shared/private clean/dirty and swap (unnamed mappings as `[anon]`).
The tool groups them into O2Physics, O2, ROOT, arrow and other libraries,
heap, anon and shm, and lists the `top` paths by PSS.

- Compare **private-dirty**: it is what the device itself costs. PSS splits
  shared pages among every process mapping them, so it moves with how many
  other jobs on the node use the same libraries.
- It works where an outside reader cannot: Hyperloop tests run as `alitrain`,
  and only a process of that user (the driver) may read the devices' smaps.
- Take it while the workflow is processing, to see the working set rather than
  what is left after end of stream.

## Typical uses

- Why is a workflow stuck: `list_devices` for the states, then
  `subscribe_logs` + `get_logs` on the device whose streaming or device state lags the others
  and the `completion` signpost stream to see which inputs a
  device is waiting for.
- Throughput or backpressure: `list_metrics`, then `subscribe` to the rate,
  size or relayer metrics (e.g. `inputs/relayed/*`, `total_rate_in_mb_s`) and
  read `get_updates` over time.
- Where the memory of a device goes: `smaps`.
- Debugging from the start: launch with `-s`, attach, enable signposts /
  subscribe to logs, then `start_devices`.
