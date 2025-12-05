# -*- coding: utf-8 -*-
# tools/debug_monitor.py

import asyncio
import json
import sys
import time
from pathlib import Path

import aiocoap
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()

# ------------------------------------------------------------
# Config-Handling: suche nach config.json ODER config/nodes.json
# ------------------------------------------------------------

CONFIG_CANDIDATES = [
    Path("config.json"),
    Path("config/nodes.json"),
    Path("nodes.json"),
]


def load_config():
    for path in CONFIG_CANDIDATES:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            console.print(f"[green]Using config:[/green] {path}")
            return cfg, path
    console.print("[red]Config not found:[/red] tried " +
                  ", ".join(str(p) for p in CONFIG_CANDIDATES))
    sys.exit(1)


def extract_nodes(cfg: dict) -> dict:
    """
    Erzeugt ein dict {node_id: ip} aus deinem nodes.json Layout.

    Erwartet:
      - ein "sync"-Block oben
      - dann A, B, C, D... mit jeweils "ip"
    """
    nodes = {}
    for key, value in cfg.items():
        if key == "sync":
            continue
        if isinstance(value, dict) and "ip" in value:
            nodes[key] = value["ip"]
    return nodes


# ------------------------------------------------------------
# CoAP Status holen
# ------------------------------------------------------------

async def fetch_status(client_ctx: aiocoap.Context, node_id: str, ip: str, timeout: float = 0.7):
    uri = f"coap://{ip}/status"
    req = aiocoap.Message(code=aiocoap.GET, uri=uri)
    try:
        req_ctx = client_ctx.request(req)
        resp = await asyncio.wait_for(req_ctx.response, timeout=timeout)
        data = json.loads(resp.payload.decode("utf-8"))
        data["_ok"] = True
        data["_ip"] = ip
        return node_id, data
    except Exception as e:
        return node_id, {
            "_ok": False,
            "_ip": ip,
            "error": str(e),
        }


# ------------------------------------------------------------
# Darstellung als Tabelle
# ------------------------------------------------------------

def build_table(status_by_node: dict) -> Table:
    table = Table(title="Mesh Time Sync Monitor", expand=True)

    table.add_column("Node", style="bold")
    table.add_column("IP")
    table.add_column("OK?")
    table.add_column("mesh_time [s]")
    table.add_column("offset [ms]")
    table.add_column("Δmono-mesh [ms]")
    table.add_column("peers θ [ms]")
    table.add_column("σ [ms]")

    now = time.time()

    for node_id, st in sorted(status_by_node.items()):
        ip = st.get("_ip", "?")

        if not st.get("_ok", False):
            table.add_row(
                node_id,
                ip,
                "[red]NO[/red]",
                "-", "-", "-", "-",
                st.get("error", "?"),
            )
            continue

        mesh_time = st.get("mesh_time", 0.0)
        offset = st.get("offset_estimate", 0.0)
        mono_now = st.get("monotonic_now", 0.0)
        delta_mono_mesh_ms = (mesh_time - mono_now) * 1000.0

        peers = st.get("peer_offsets_ms") or {}
        sigmas = st.get("peer_sigma_ms") or {}

        peers_str = ", ".join(f"{k}:{v:.1f}" for k, v in peers.items()) or "-"
        sigma_str = ", ".join(f"{k}:{v:.2f}" for k, v in sigmas.items()) or "-"

        table.add_row(
            node_id,
            ip,
            "[green]OK[/green]",
            f"{mesh_time:10.3f}",
            f"{offset*1000.0:8.1f}",
            f"{delta_mono_mesh_ms:8.1f}",
            peers_str,
            sigma_str,
        )

    return table


# ------------------------------------------------------------
# Main-Loop
# ------------------------------------------------------------

async def monitor_loop():
    cfg, cfg_path = load_config()
    nodes = extract_nodes(cfg)

    if not nodes:
        console.print("[red]No nodes with 'ip' found in config.[/red]")
        sys.exit(1)

    console.print("[cyan]Monitoring nodes:[/cyan] " +
                  ", ".join(f"{nid}({ip})" for nid, ip in nodes.items()))

    refresh_interval = 1.0

    # Create client context ONCE (old aiocoap does not support async with)
    try:
        client_ctx = await aiocoap.Context.create_client_context()
    except Exception as e:
        console.print(f"[red]Failed to create CoAP client context:[/red] {e}")
        sys.exit(1)

    status_by_node = {}

    # Live table runs forever, even if nodes are offline initially
    with Live(refresh_per_second=4, console=console) as live:
        try:
            while True:
                tasks = [
                    fetch_status(client_ctx, nid, ip)
                    for nid, ip in nodes.items()
                ]
                results = await asyncio.gather(*tasks)

                for nid, st in results:
                    status_by_node[nid] = st

                table = build_table(status_by_node)
                live.update(table)

                await asyncio.sleep(refresh_interval)
        finally:
            # Best-effort shutdown
            try:
                await client_ctx.shutdown()
            except Exception:
                pass


def main():
    try:
        asyncio.run(monitor_loop())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user.[/yellow]")


if __name__ == "__main__":
    main()
