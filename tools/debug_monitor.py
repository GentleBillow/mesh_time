# -*- coding: utf-8 -*-
# tools/debug_monitor.py

import asyncio
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import aiocoap

from rich.console import Console
from rich.table import Table
from rich.live import Live


console = Console()


async def fetch_status(
    client_ctx: aiocoap.Context,
    node_id: str,
    ip: str,
    timeout: float,
) -> Dict[str, Any]:
    """
    Hole /status von einem Node.
    Gibt immer ein Dict zurück, mit _ok Flag und optional _error.
    """
    uri = f"coap://{ip}/status"
    req = aiocoap.Message(code=aiocoap.GET, uri=uri)

    try:
        req_ctx = client_ctx.request(req)
        resp = await asyncio.wait_for(req_ctx.response, timeout=timeout)
        data = json.loads(resp.payload.decode("utf-8"))
        data["_ok"] = True
        data["_ip"] = ip
        return data
    except Exception as e:
        return {
            "node_id": node_id,
            "_ok": False,
            "_ip": ip,
            "_error": str(e),
        }


def build_table(snapshots):
    """
    Baut eine hübsche Tabelle aus den Snapshots.
    """
    table = Table(title="Mesh Time Debug Monitor", show_lines=False)

    table.add_column("Node", style="bold")
    table.add_column("IP")
    table.add_column("mesh_time", justify="right")
    table.add_column("Δmesh [ms]", justify="right")
    table.add_column("offset [ms]", justify="right")
    table.add_column("peers θ [ms]", justify="left")
    table.add_column("peers σ [ms]", justify="left")
    table.add_column("status", justify="left")

    # Nur erfolgreiche Snapshots für Referenz-Δ
    valid = [s for s in snapshots if s.get("_ok")]
    if valid:
        min_mesh = min(s.get("mesh_time", 0.0) for s in valid)
    else:
        min_mesh = 0.0

    for snap in snapshots:
        node_id = snap.get("node_id", "?")
        ip = snap.get("_ip", "?")

        if not snap.get("_ok", False):
            table.add_row(
                node_id,
                ip,
                "-",
                "-",
                "-",
                "-",
                "-",
                f"[red]ERROR: {snap.get('_error', 'no response')}[/red]",
            )
            continue

        mesh_time = snap.get("mesh_time", 0.0)
        offset_s = snap.get("offset_estimate", 0.0)
        offset_ms = snap.get("offset_estimate_ms", offset_s * 1000.0)

        # Δmesh in ms relativ zum minimalen mesh_time
        delta_mesh_ms = (mesh_time - min_mesh) * 1000.0

        # Peer-Offsets & Sigma (ms)
        peer_offsets_ms = snap.get("peer_offsets_ms")
        if peer_offsets_ms is None:
            # Fallback: aus Sekunden rechnen, falls nur peer_offsets vorhanden
            peer_offsets = snap.get("peer_offsets", {})
            peer_offsets_ms = {k: v * 1000.0 for k, v in peer_offsets.items()}

        peer_sigma_ms = snap.get("peer_sigma_ms")
        if peer_sigma_ms is None:
            peer_sigma = snap.get("peer_sigma", {})
            peer_sigma_ms = {k: v * 1000.0 for k, v in peer_sigma.items()}

        if peer_offsets_ms:
            peers_theta_str = ", ".join(
                f"{peer}:{val:.2f}"
                for peer, val in sorted(peer_offsets_ms.items())
            )
        else:
            peers_theta_str = "-"

        if peer_sigma_ms:
            peers_sigma_str = ", ".join(
                f"{peer}:{val:.2f}"
                for peer, val in sorted(peer_sigma_ms.items())
            )
        else:
            peers_sigma_str = "-"

        status_str = "[green]OK[/green]"

        table.add_row(
            node_id,
            ip,
            f"{mesh_time:.3f}",
            f"{delta_mesh_ms:.2f}",
            f"{offset_ms:.2f}",
            peers_theta_str,
            peers_sigma_str,
            status_str,
        )

    return table


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Lädt die globale Mesh-Config (die gleiche, die du für run_node.py nutzt).
    Erwartet eine JSON-Map { "A": {...}, "B": {...}, "sync": {...}, ... }.
    """
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return data


def extract_nodes(global_cfg: Dict[str, Any], only_nodes=None) -> Dict[str, str]:
    """
    Extrahiert Node→IP Map aus der globalen Config.
    expected:
      {
        "A": {"ip": "172.16.32.x", ...},
        "B": {"ip": "172.16.32.y", ...},
        "sync": {...}
      }
    """
    nodes = {}
    for nid, cfg in global_cfg.items():
        if nid == "sync":
            continue
        if only_nodes is not None and nid not in only_nodes:
            continue
        if isinstance(cfg, dict) and "ip" in cfg:
            nodes[nid] = cfg["ip"]

    return nodes


async def main_async(args):
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        console.print(f"[red]Config not found:[/red] {config_path}")
        return

    global_cfg = load_config(config_path)

    only_nodes = None
    if args.nodes:
        only_nodes = args.nodes.split(",")

    nodes = extract_nodes(global_cfg, only_nodes=only_nodes)
    if not nodes:
        console.print("[red]No nodes with 'ip' found in config.[/red]")
        return

    console.print(
        f"[bold]Monitoring nodes:[/bold] "
        + ", ".join(f"{nid}@{ip}" for nid, ip in nodes.items())
    )
    console.print(f"Interval: {args.interval:.2f}s, Timeout: {args.timeout:.2f}s")
    console.print("Press Ctrl+C to stop.\n")

    client_ctx = await aiocoap.Context.create_client_context()

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                tasks = [
                    fetch_status(client_ctx, nid, ip, args.timeout)
                    for nid, ip in nodes.items()
                ]
                snapshots = await asyncio.gather(*tasks, return_exceptions=False)
                table = build_table(snapshots)
                live.update(table)
                await asyncio.sleep(args.interval)
    except KeyboardInterrupt:
        console.print("\n[bold]Stopping monitor (KeyboardInterrupt).[/bold]")
    finally:
        try:
            await client_ctx.shutdown()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Live Mesh Time Debug Monitor (CoAP /status)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.json",
        help="Path to global mesh config JSON (default: config.json)",
    )
    parser.add_argument(
        "--nodes",
        "-n",
        type=str,
        default="",
        help="Comma-separated list of node IDs to monitor (default: all in config)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=0.5,
        help="CoAP timeout per request in seconds (default: 0.5)",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
