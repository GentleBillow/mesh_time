#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from mesh.node import MeshNode


def load_config(node_id: str):
    config_path = Path(__file__).parent / "config" / "nodes.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if node_id not in cfg:
        raise ValueError(f"Unknown node id {node_id!r}. "
                         f"Must be one of: {', '.join(cfg.keys())}")

    node_cfg = cfg[node_id]
    return node_cfg, cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run a mesh_time node (A/B/C/D)."
    )
    parser.add_argument(
        "--id",
        required=True,
        choices=["A", "B", "C", "D"],
        help="Logical node ID"
    )
    args = parser.parse_args()

    node_cfg, global_cfg = load_config(args.id)
    node = MeshNode(node_id=args.id, node_cfg=node_cfg, global_cfg=global_cfg)
    node.run()


if __name__ == "__main__":
    main()
