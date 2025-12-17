# -*- coding: utf-8 -*-
# web_ui3.py - MeshTime Convergence Dashboard

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Config
DB_PATH = "/home/pi/mesh_time/mesh_data.sqlite"
DEFAULT_WINDOW_S = 600
NODE_IDS = ["A", "B", "C"]

NODE_COLORS = {
    'A': {'line': 'rgb(255, 0, 0)', 'fill': 'rgba(255, 0, 0, 0.6)'},  # Red
    'B': {'line': 'rgb(0, 255, 0)', 'fill': 'rgba(0, 255, 0, 0.6)'},  # Green
    'C': {'line': 'rgb(0, 0, 255)', 'fill': 'rgba(0, 0, 255, 0.6)'},  # Blue
}

LINK_COLORS = {
    'A-B': {'line': 'rgb(255, 255, 0)', 'fill': 'rgba(255, 255, 0, 0.6)'},  # Yellow (Red + Green)
    'A-C': {'line': 'rgb(255, 0, 255)', 'fill': 'rgba(255, 0, 255, 0.6)'},  # Magenta (Red + Blue)
    'B-C': {'line': 'rgb(0, 255, 255)', 'fill': 'rgba(0, 255, 255, 0.6)'},  # Cyan (Green + Blue)
}


# ============================================================================
# Database Loading
# ============================================================================

def load_timeseries_from_db(db_path: str, window_s: float) -> Dict[str, List[Tuple[float, float]]]:
    """
    Load offset timeseries from DB for all nodes.

    Returns:
        {'A': [(t_wall, offset_ms), ...], 'B': [...], 'C': [...]}
    """
    cutoff = time.time() - window_s

    node_data = {nid: [] for nid in NODE_IDS}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Convert offset from seconds to ms in the query
        query = """
            SELECT node_id, t_wall, offset * 1000.0 as offset_ms
            FROM ntp_reference
            WHERE t_wall > ?
            ORDER BY t_wall ASC
        """

        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()

        for node_id, t_wall, offset_ms in rows:
            if node_id in node_data:
                node_data[node_id].append((float(t_wall), float(offset_ms)))

        conn.close()

    except Exception as e:
        print(f"[web_ui3] DB load error: {e}")
        return {nid: [] for nid in NODE_IDS}

    return node_data


# ============================================================================
# Interpolation
# ============================================================================

def linear_interp(data: List[Tuple[float, float]], t: float, max_gap_s: float = 5.0) -> Optional[float]:
    """
    Linear interpolation of offset at time t.

    Args:
        data: [(t, offset_ms), ...] sorted by t
        t: target timestamp
        max_gap_s: maximum extrapolation distance

    Returns:
        Interpolated offset_ms or None if out of range
    """
    if not data:
        return None

    # Find bracketing points
    before = None
    after = None

    for i, (ti, vi) in enumerate(data):
        if ti <= t:
            before = (ti, vi)
        if ti >= t and after is None:
            after = (ti, vi)
            break

    # Exact match
    if before and before[0] == t:
        return before[1]
    if after and after[0] == t:
        return after[1]

    # Interpolate
    if before and after:
        t1, v1 = before
        t2, v2 = after
        if t2 == t1:
            return v1
        alpha = (t - t1) / (t2 - t1)
        return v1 + alpha * (v2 - v1)

    # Extrapolate (constant) if within max_gap
    if before and (t - before[0]) <= max_gap_s:
        return before[1]
    if after and (after[0] - t) <= max_gap_s:
        return after[1]

    return None


def interpolate_to_common_timeline(
        node_data: Dict[str, List[Tuple[float, float]]]
) -> Tuple[List[float], Dict[str, List[Optional[float]]]]:
    """
    Interpolate all nodes to common timeline and compute deviations from median.

    Returns:
        timestamps: [t1, t2, ...]
        deviations: {'A': [dev_ms, None, ...], 'B': [...], 'C': [...]}
    """
    # Merge all timestamps
    all_t = set()
    for data in node_data.values():
        for t, _ in data:
            all_t.add(t)

    if not all_t:
        return [], {nid: [] for nid in NODE_IDS}

    timestamps = sorted(all_t)

    # Interpolate each node
    interpolated = {nid: [] for nid in NODE_IDS}
    for nid in NODE_IDS:
        for t in timestamps:
            val = linear_interp(node_data[nid], t)
            interpolated[nid].append(val)

    # Compute median and deviations
    deviations = {nid: [] for nid in NODE_IDS}

    for i, t in enumerate(timestamps):
        offsets = []
        for nid in NODE_IDS:
            val = interpolated[nid][i]
            if val is not None:
                offsets.append(val)

        if len(offsets) >= 2:
            median = float(np.median(offsets))
            for nid in NODE_IDS:
                val = interpolated[nid][i]
                deviations[nid].append((val - median) if val is not None else None)
        else:
            # Not enough nodes available
            for nid in NODE_IDS:
                deviations[nid].append(None)

    # Filter out timestamps where all nodes are None
    filtered_timestamps = []
    filtered_deviations = {nid: [] for nid in NODE_IDS}

    for i, t in enumerate(timestamps):
        has_data = any(deviations[nid][i] is not None for nid in NODE_IDS)
        if has_data:
            filtered_timestamps.append(t)
            for nid in NODE_IDS:
                filtered_deviations[nid].append(deviations[nid][i])

    # Reduce data points if too many (keep every Nth point)
    MAX_POINTS = 500
    if len(filtered_timestamps) > MAX_POINTS:
        step = len(filtered_timestamps) // MAX_POINTS
        reduced_timestamps = filtered_timestamps[::step]
        reduced_deviations = {nid: filtered_deviations[nid][::step] for nid in NODE_IDS}
        return reduced_timestamps, reduced_deviations

    return filtered_timestamps, filtered_deviations


# ============================================================================
# Histogram
# ============================================================================

def compute_histogram(
        deviations: Dict[str, List[Optional[float]]],
        n_bins: int = 50
) -> Tuple[List[float], Dict[str, List[int]]]:
    """
    Compute histogram of deviations per node.

    Returns:
        bin_edges: [edge0, edge1, ..., edge_n]
        counts: {'A': [count0, count1, ...], 'B': [...], 'C': [...]}
    """
    # Collect all non-None deviations
    all_devs = []
    for nid in NODE_IDS:
        all_devs.extend([d for d in deviations[nid] if d is not None])

    if not all_devs:
        return [0, 1], {nid: [0] for nid in NODE_IDS}

    # Compute bin edges
    min_dev = float(np.min(all_devs))
    max_dev = float(np.max(all_devs))

    # Add padding
    range_dev = max_dev - min_dev
    if range_dev < 0.01:  # Very tight sync
        range_dev = 0.01

    bin_min = min_dev - 0.1 * range_dev
    bin_max = max_dev + 0.1 * range_dev

    bin_edges = np.linspace(bin_min, bin_max, n_bins + 1).tolist()

    # Compute counts per node
    counts = {}
    for nid in NODE_IDS:
        node_devs = [d for d in deviations[nid] if d is not None]
        if node_devs:
            hist, _ = np.histogram(node_devs, bins=bin_edges)
            counts[nid] = hist.tolist()
        else:
            counts[nid] = [0] * n_bins

    return bin_edges, counts


# ============================================================================
# Link Quality Data
# ============================================================================

def load_link_data_from_db(db_path: str, window_s: float) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    Load link quality metrics from obs_link table.

    Returns:
        {'A-B': [(t, rtt_ms, theta_ms, sigma_ms), ...], 'A-C': [...], 'B-C': [...]}
    """
    cutoff = time.time() - window_s

    # All possible links in sorted order
    links = ['A-B', 'A-C', 'B-C']
    link_data = {link: [] for link in links}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
            SELECT created_at_s, node_id, peer_id, rtt_ms, theta_ms, sigma_ms,
                   t1_s, t2_s, t3_s, t4_s, accepted
            FROM obs_link
            WHERE created_at_s > ? AND accepted = 1
            ORDER BY created_at_s ASC
        """

        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()

        for row in rows:
            t, node_id, peer_id, rtt_ms, theta_ms, sigma_ms, t1, t2, t3, t4, _ = row

            # Normalize link identifier (always alphabetical)
            link_pair = tuple(sorted([node_id, peer_id]))
            link_id = f"{link_pair[0]}-{link_pair[1]}"

            if link_id in link_data:
                # Calculate asymmetry: (t3-t2) - (t4-t1)
                asymmetry_ms = None
                if all(x is not None for x in [t1, t2, t3, t4]):
                    forward = (t3 - t2) * 1000  # ms
                    backward = (t4 - t1) * 1000  # ms
                    asymmetry_ms = forward - backward

                link_data[link_id].append((
                    float(t),
                    float(rtt_ms) if rtt_ms is not None else None,
                    float(asymmetry_ms) if asymmetry_ms is not None else None,
                    float(sigma_ms) if sigma_ms is not None else None
                ))

        conn.close()

    except Exception as e:
        print(f"[web_ui3] Link data load error: {e}")
        return {link: [] for link in links}

    return link_data


def compute_link_metrics(link_data: Dict[str, List[Tuple[float, float, float, float]]]) -> Dict:
    """
    Compute aggregate link quality metrics.

    Returns:
        {
            'timeseries': {'A-B': {timestamps: [...], rtt: [...], asymmetry: [...], sigma: [...]}, ...},
            'jitter': {'A-B': {timestamps: [...], jitter: [...]}, ...},
            'acceptance': {'A-B': 0.95, 'A-C': 0.98, ...}
        }
    """
    timeseries = {}
    jitter = {}

    for link_id, data in link_data.items():
        if not data:
            timeseries[link_id] = {'timestamps': [], 'rtt': [], 'asymmetry': [], 'sigma': []}
            jitter[link_id] = {'timestamps': [], 'jitter': []}
            continue

        timestamps = [d[0] for d in data]
        rtt_values = [d[1] for d in data]
        asym_values = [d[2] for d in data]
        sigma_values = [d[3] for d in data]

        # Compute rolling jitter (std dev of RTT over window)
        jitter_timestamps = []
        jitter_values = []
        window_size = 20  # samples

        for i in range(len(rtt_values)):
            if i >= window_size - 1:
                window_rtt = [r for r in rtt_values[i - window_size + 1:i + 1] if r is not None]
                if len(window_rtt) >= 5:
                    jitter_val = float(np.std(window_rtt))
                    jitter_timestamps.append(timestamps[i])
                    jitter_values.append(jitter_val)

        timeseries[link_id] = {
            'timestamps': timestamps,
            'rtt': rtt_values,
            'asymmetry': asym_values,
            'sigma': sigma_values
        }

        jitter[link_id] = {
            'timestamps': jitter_timestamps,
            'jitter': jitter_values
        }

    # Note: acceptance rate would need rejected measurements too, skip for now
    # We can add it later by querying both accepted=1 and accepted=0

    return {
        'timeseries': timeseries,
        'jitter': jitter
    }


# ============================================================================
# Statistics
# ============================================================================

def compute_stats(
        deviations: Dict[str, List[Optional[float]]],
        node_data: Dict[str, List[Tuple[float, float]]]
) -> Dict:
    """Compute summary statistics."""
    all_devs = []
    for nid in NODE_IDS:
        all_devs.extend([d for d in deviations[nid] if d is not None])

    if not all_devs:
        return {
            "n_samples": 0,
            "std_dev_ms": 0,
            "max_abs_ms": 0,
            "nodes": {nid: {"n_samples": 0, "current_dev_ms": None} for nid in NODE_IDS}
        }

    std_dev_ms = float(np.std(all_devs))
    max_abs_ms = float(np.max(np.abs(all_devs)))

    node_stats = {}
    for nid in NODE_IDS:
        node_devs = [d for d in deviations[nid] if d is not None]
        current_dev = node_devs[-1] if node_devs else None
        node_stats[nid] = {
            "n_samples": len(node_data[nid]),
            "current_dev_ms": float(current_dev) if current_dev is not None else None
        }

    return {
        "n_samples": len(all_devs),
        "std_dev_ms": std_dev_ms,
        "max_abs_ms": max_abs_ms,
        "nodes": node_stats
    }


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    return render_template('convergence.html')


@app.route('/api/convergence')
def api_convergence():
    """Main API endpoint for convergence and link quality data."""

    # Load convergence data
    node_data = load_timeseries_from_db(DB_PATH, DEFAULT_WINDOW_S)

    # Check if we have data
    total_samples = sum(len(data) for data in node_data.values())
    if total_samples == 0:
        return jsonify({
            "error": "No data available",
            "timeseries": {"timestamps": [], "deviations": {nid: [] for nid in NODE_IDS}},
            "histogram": {"bin_edges": [0, 1], "counts": {nid: [0] for nid in NODE_IDS}},
            "stats": {"n_samples": 0, "std_dev_ms": 0, "max_abs_ms": 0, "nodes": {}},
            "colors": NODE_COLORS,
            "link_colors": LINK_COLORS,
            "links": {"timeseries": {}, "jitter": {}}
        })

    # Interpolate to common timeline
    timestamps, deviations = interpolate_to_common_timeline(node_data)

    # Compute histogram
    bin_edges, hist_counts = compute_histogram(deviations, n_bins=50)

    # Compute stats
    stats = compute_stats(deviations, node_data)

    # Load link quality data
    link_data = load_link_data_from_db(DB_PATH, DEFAULT_WINDOW_S)
    link_metrics = compute_link_metrics(link_data)

    # Return JSON
    return jsonify({
        "timeseries": {
            "timestamps": timestamps,
            "deviations": deviations
        },
        "histogram": {
            "bin_edges": bin_edges,
            "counts": hist_counts
        },
        "stats": stats,
        "colors": NODE_COLORS,
        "link_colors": LINK_COLORS,
        "links": link_metrics,
        "window_s": DEFAULT_WINDOW_S
    })


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f"[web_ui3] Starting dashboard on http://0.0.0.0:5000")
    print(f"[web_ui3] DB path: {DB_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=True)