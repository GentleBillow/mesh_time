# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
run_node.py — drop-in

- liest config/nodes.json (utf-8)
- validiert Node-IDs dynamisch (alle Keys außer "sync")
- startet MeshNode
- setzt EventLoopPolicy sauber (Windows/Linux kompatibel)

Usage:
  python run_node.py --id A
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
from pathlib import Path
from typing import Any, Dict, Tuple

from mesh.node import MeshNode


def _set_event_loop_policy() -> None:
    """
    aiocoap/asyncio: auf Windows ist der Proactor-Loop manchmal problematisch.
    Auf Linux/RPi egal, aber wir halten's korrekt & explicit.
    """
    if platform.system() == "Windows":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            # fallback: Default policy
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    else:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


def load_config(node_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config_path = Path(__file__).parent / "config" / "nodes.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)

    valid_nodes = sorted([k for k in cfg.keys() if k != "sync"])
    if node_id not in valid_nodes:
        raise ValueError(
            "Unknown node id '{}'. Must be one of: {}".format(node_id, ", ".join(valid_nodes))
        )

    node_cfg = cfg[node_id] or {}
    return node_cfg, cfg


def main() -> None:
    _set_event_loop_policy()

    parser = argparse.ArgumentParser(description="Run a mesh_time node (IDs come from config/nodes.json).")
    parser.add_argument("--id", required=True, help="Logical node ID (e.g. A/B/C/D)")
    args = parser.parse_args()

    node_cfg, global_cfg = load_config(args.id)

    node = MeshNode(node_id=args.id, node_cfg=node_cfg, global_cfg=global_cfg)
    node.run()


if __name__ == "__main__":
    main()
