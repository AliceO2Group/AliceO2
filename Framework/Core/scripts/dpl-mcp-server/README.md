# DPL Status MCP Server

An MCP server that connects to a running DPL driver's `/status` WebSocket endpoint and exposes its device state and metrics as tools for an AI assistant (e.g. Claude).

## Requirements

```bash
pip install mcp websockets
# or install the package directly:
pip install ./Framework/Core/scripts/dpl-mcp-server/
```

## Running

The driver port defaults to `8080`. Override with `--port`, `--pid`, or `DPL_STATUS_PORT`:

```bash
python3 dpl_mcp_server.py --port 8080
python3 dpl_mcp_server.py --pid 12345   # port = 8080 + pid % 30000
DPL_STATUS_PORT=8080 python3 dpl_mcp_server.py
```

If installed as a package:

```bash
dpl-mcp-server --pid $(pgrep -f diamond-workflow | head -1)
```

## Claude Code integration

Add to `.mcp.json` in your project (or `~/.claude.json` for global use):

```json
{
  "mcpServers": {
    "dpl": {
      "command": "dpl-mcp-server",
      "args": ["--pid", "12345"]
    }
  }
}
```

Or with `claude mcp add`:

```bash
claude mcp add dpl -- dpl-mcp-server --pid 12345
```

## Available tools

| Tool | Description |
|------|-------------|
| `list_devices` | List all devices with pid, active flag, streaming and device state |
| `list_metrics(device)` | List numeric metrics available for a device |
| `subscribe(device, metrics)` | Subscribe to metrics; driver will push updates when they change |
| `unsubscribe(device, metrics)` | Stop receiving updates for specific metrics |
| `get_updates(max_updates)` | Drain buffered update frames (default: up to 50) |

## Protocol

The driver sends a snapshot on connect, then pushes updates only for subscribed metrics that changed each processing cycle.  There is no polling — updates arrive in real time as the workflow runs.

```
connect  →  {"type":"snapshot","devices":[{"name","pid","active","streamingState","deviceState"},...]}

client   →  {"cmd":"list_metrics","device":"producer"}
driver   →  {"type":"metrics_list","device":"producer","metrics":["input-parts","output-bytes",...]}

client   →  {"cmd":"subscribe","device":"producer","metrics":["output-bytes"]}
driver   →  {"type":"update","device":0,"name":"producer","metrics":{"output-bytes":1048576}}
             (pushed every cycle in which output-bytes changed)

client   →  {"cmd":"unsubscribe","device":"producer","metrics":["output-bytes"]}
```
