# -*- coding: utf-8 -*-
# web_ui3_full.py - Complete MeshTime Dashboard with Sync + Links + Controller
# Split-API pattern for fast loading

from __future__ import annotations

import json
import numpy as np
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template

# ============================================================================
# Configuration
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "mesh_data.sqlite"
CFG_PATH = ROOT_DIR / "config" / "nodes.json"

DEFAULT_WINDOW_S = 600.0
NODE_IDS = ['A', 'B', 'C']

NODE_COLORS = {
    'A': {'line': 'rgb(255, 0, 0)', 'fill': 'rgba(255, 0, 0, 0.6)'},
    'B': {'line': 'rgb(0, 255, 0)', 'fill': 'rgba(0, 255, 0, 0.6)'},
    'C': {'line': 'rgb(0, 0, 255)', 'fill': 'rgba(0, 0, 255, 0.6)'},
}

LINK_COLORS = {
    'A-B': {'line': 'rgb(255, 255, 0)', 'fill': 'rgba(255, 255, 0, 0.6)'},
    'A-C': {'line': 'rgb(255, 0, 255)', 'fill': 'rgba(255, 0, 255, 0.6)'},
    'B-C': {'line': 'rgb(0, 255, 255)', 'fill': 'rgba(0, 255, 255, 0.6)'},
}

app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"))

# ============================================================================
# Helper Functions
# ============================================================================

def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if (n % 2 == 1) else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ============================================================================
# Category 1: Synchronization Quality
# ============================================================================

def load_sync_data(db_path: str, window_s: float) -> Dict:
    """Load mesh_clock data for convergence analysis."""
    cutoff = time.time() - window_s
    node_data = {nid: [] for nid in NODE_IDS}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT created_at_s, node_id, offset_s
            FROM mesh_clock
            WHERE created_at_s > ?
            ORDER BY created_at_s ASC
        """
        
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        
        for row in rows:
            t, node_id, offset_s = row
            if node_id in node_data and offset_s is not None:
                node_data[node_id].append((float(t), float(offset_s) * 1000.0))  # to ms
        
        conn.close()
    except Exception as e:
        print(f"[sync_data] Error: {e}")
    
    return node_data


def interpolate_sync_timeline(node_data: Dict[str, List[Tuple[float, float]]]) -> Tuple[List[float], Dict]:
    """Interpolate all nodes to common timeline and compute deviations from median."""
    # Merge all timestamps
    all_timestamps = set()
    for data in node_data.values():
        all_timestamps.update([t for t, _ in data])
    
    if not all_timestamps:
        return [], {nid: [] for nid in NODE_IDS}
    
    timestamps = sorted(all_timestamps)
    
    # Limit to max 500 points
    if len(timestamps) > 500:
        step = len(timestamps) // 500
        timestamps = [timestamps[i] for i in range(0, len(timestamps), step)]
    
    # Interpolate each node
    interpolated = {nid: [] for nid in NODE_IDS}
    
    for nid in NODE_IDS:
        data = node_data[nid]
        if not data:
            interpolated[nid] = [None] * len(timestamps)
            continue
        
        data_t = [t for t, _ in data]
        data_v = [v for _, v in data]
        
        result = []
        for t in timestamps:
            # Find bracketing points
            if t < data_t[0] or t > data_t[-1]:
                result.append(None)
                continue
            
            # Linear interpolation
            for i in range(len(data_t) - 1):
                if data_t[i] <= t <= data_t[i + 1]:
                    dt = data_t[i + 1] - data_t[i]
                    if dt > 0:
                        alpha = (t - data_t[i]) / dt
                        v = data_v[i] + alpha * (data_v[i + 1] - data_v[i])
                        result.append(float(v))
                    else:
                        result.append(float(data_v[i]))
                    break
            else:
                result.append(None)
        
        interpolated[nid] = result
    
    # Compute median and deviations
    deviations = {nid: [] for nid in NODE_IDS}
    
    for i in range(len(timestamps)):
        values = [interpolated[nid][i] for nid in NODE_IDS if interpolated[nid][i] is not None]
        
        if len(values) >= 2:
            median_val = _median(values)
            for nid in NODE_IDS:
                if interpolated[nid][i] is not None:
                    deviations[nid].append(interpolated[nid][i] - median_val)
                else:
                    deviations[nid].append(None)
        else:
            for nid in NODE_IDS:
                deviations[nid].append(None)
    
    return timestamps, deviations


def compute_sync_histogram(deviations: Dict[str, List[Optional[float]]], n_bins: int = 50) -> Dict:
    """Compute histogram of deviations."""
    all_values = []
    for vals in deviations.values():
        all_values.extend([v for v in vals if v is not None])
    
    if not all_values:
        return {'bin_edges': [0, 1], 'counts': {nid: [0] for nid in NODE_IDS}}
    
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Add 10% padding
    range_val = max_val - min_val
    min_val -= 0.1 * range_val
    max_val += 0.1 * range_val
    
    bin_edges = list(np.linspace(min_val, max_val, n_bins + 1))
    
    counts = {}
    for nid in NODE_IDS:
        vals = [v for v in deviations[nid] if v is not None]
        if vals:
            hist, _ = np.histogram(vals, bins=bin_edges)
            counts[nid] = hist.tolist()
        else:
            counts[nid] = [0] * n_bins
    
    return {'bin_edges': bin_edges, 'counts': counts}


def compute_sync_stats(deviations: Dict[str, List[Optional[float]]], node_data: Dict) -> Dict:
    """Compute sync statistics."""
    all_values = []
    for vals in deviations.values():
        all_values.extend([v for v in vals if v is not None])
    
    stats = {
        'n_samples': len(all_values),
        'std_dev_ms': float(np.std(all_values)) if all_values else 0.0,
        'max_abs_ms': float(max([abs(v) for v in all_values])) if all_values else 0.0,
        'nodes': {}
    }
    
    for nid in NODE_IDS:
        vals = [v for v in deviations[nid] if v is not None]
        stats['nodes'][nid] = {
            'current_ms': vals[-1] if vals else None,
            'std_ms': float(np.std(vals)) if vals else 0.0,
            'n_samples': len(node_data.get(nid, []))
        }
    
    return stats


@app.route('/api/sync')
def api_sync():
    """API endpoint for synchronization quality data."""
    node_data = load_sync_data(str(DB_PATH), DEFAULT_WINDOW_S)
    timestamps, deviations = interpolate_sync_timeline(node_data)
    histogram = compute_sync_histogram(deviations)
    stats = compute_sync_stats(deviations, node_data)
    
    return jsonify({
        'timeseries': {'timestamps': timestamps, 'deviations': deviations},
        'histogram': histogram,
        'stats': stats,
        'colors': NODE_COLORS
    })


# ============================================================================
# Category 2: Link Quality
# ============================================================================

def load_link_data(db_path: str, window_s: float) -> Dict:
    """Load link quality data from obs_link table."""
    cutoff = time.time() - window_s
    links = ['A-B', 'A-C', 'B-C']
    link_data = {link: [] for link in links}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT created_at_s, node_id, peer_id, rtt_ms, theta_ms, sigma_ms,
                   t1_s, t2_s, t3_s, t4_s
            FROM obs_link
            WHERE created_at_s > ? AND accepted = 1
            ORDER BY created_at_s ASC
        """
        
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        
        for row in rows:
            t, node_id, peer_id, rtt_ms, theta_ms, sigma_ms, t1, t2, t3, t4 = row
            
            link_pair = tuple(sorted([node_id, peer_id]))
            link_id = f"{link_pair[0]}-{link_pair[1]}"
            
            if link_id in link_data:
                asymmetry_ms = None
                if all(x is not None for x in [t1, t2, t3, t4]):
                    forward = (t3 - t2) * 1000
                    backward = (t4 - t1) * 1000
                    asymmetry_ms = forward - backward
                
                link_data[link_id].append({
                    't': float(t),
                    'rtt': float(rtt_ms) if rtt_ms is not None else None,
                    'asymmetry': float(asymmetry_ms) if asymmetry_ms is not None else None,
                    'sigma': float(sigma_ms) if sigma_ms is not None else None
                })
        
        conn.close()
    except Exception as e:
        print(f"[link_data] Error: {e}")
    
    return link_data


def compute_link_jitter(link_data: Dict, window_size: int = 20) -> Dict:
    """Compute rolling jitter (std dev of RTT)."""
    jitter = {}
    
    for link_id, data in link_data.items():
        if not data:
            jitter[link_id] = {'timestamps': [], 'jitter': []}
            continue
        
        rtt_values = [d['rtt'] for d in data if d['rtt'] is not None]
        timestamps = [d['t'] for d in data]
        
        jitter_timestamps = []
        jitter_values = []
        
        for i in range(len(rtt_values)):
            if i >= window_size - 1:
                window_rtt = rtt_values[i-window_size+1:i+1]
                if len(window_rtt) >= 5:
                    jitter_val = float(np.std(window_rtt))
                    jitter_timestamps.append(timestamps[i])
                    jitter_values.append(jitter_val)
        
        jitter[link_id] = {'timestamps': jitter_timestamps, 'jitter': jitter_values}
    
    return jitter


def compute_link_stats(link_data: Dict) -> Dict:
    """Compute aggregate link statistics."""
    stats = {}
    
    for link_id, data in link_data.items():
        if not data:
            stats[link_id] = {'avg_rtt': 0.0, 'avg_asymmetry': 0.0, 'n_samples': 0}
            continue
        
        rtt_vals = [d['rtt'] for d in data if d['rtt'] is not None]
        asym_vals = [d['asymmetry'] for d in data if d['asymmetry'] is not None]
        
        stats[link_id] = {
            'avg_rtt': float(np.mean(rtt_vals)) if rtt_vals else 0.0,
            'avg_asymmetry': float(np.mean([abs(a) for a in asym_vals])) if asym_vals else 0.0,
            'n_samples': len(data)
        }
    
    return stats


@app.route('/api/links')
def api_links():
    """API endpoint for link quality data."""
    link_data = load_link_data(str(DB_PATH), DEFAULT_WINDOW_S)
    jitter = compute_link_jitter(link_data)
    stats = compute_link_stats(link_data)
    
    # Format for frontend
    timeseries = {}
    for link_id, data in link_data.items():
        timeseries[link_id] = {
            'timestamps': [d['t'] for d in data],
            'rtt': [d['rtt'] for d in data],
            'asymmetry': [d['asymmetry'] for d in data],
            'sigma': [d['sigma'] for d in data]
        }
    
    return jsonify({
        'timeseries': timeseries,
        'jitter': jitter,
        'stats': stats,
        'colors': LINK_COLORS
    })


# ============================================================================
# Category 3: Controller Diagnostics
# ============================================================================

def load_controller_data(db_path: str, window_s: float) -> Dict:
    """Load controller diagnostic data."""
    cutoff = time.time() - window_s
    node_data = {nid: [] for nid in NODE_IDS}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT created_at_s, node_id, delta_desired_ms, delta_applied_ms, slew_clipped
            FROM diag_controller
            WHERE created_at_s > ?
            ORDER BY created_at_s ASC
        """
        
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        
        for row in rows:
            t, node_id, desired, applied, clipped = row
            if node_id in node_data:
                node_data[node_id].append({
                    't': float(t),
                    'desired': float(desired) if desired is not None else None,
                    'applied': float(applied) if applied is not None else None,
                    'clipped': int(clipped) if clipped is not None else 0
                })
        
        conn.close()
    except Exception as e:
        print(f"[controller_data] Error: {e}")
    
    return node_data


def compute_controller_stats(node_data: Dict) -> Dict:
    """Compute controller statistics."""
    stats = {}
    
    for nid in NODE_IDS:
        data = node_data[nid]
        if not data:
            stats[nid] = {'clip_rate': 0.0, 'n_samples': 0}
            continue
        
        clipped_count = sum(1 for d in data if d['clipped'] == 1)
        
        stats[nid] = {
            'clip_rate': float(clipped_count / len(data)) if data else 0.0,
            'n_samples': len(data)
        }
    
    return stats


@app.route('/api/controller')
def api_controller():
    """API endpoint for controller diagnostic data."""
    node_data = load_controller_data(str(DB_PATH), DEFAULT_WINDOW_S)
    stats = compute_controller_stats(node_data)
    
    # Format for frontend
    timeseries = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        timeseries[nid] = {
            'timestamps': [d['t'] for d in data],
            'desired': [d['desired'] for d in data],
            'applied': [d['applied'] for d in data],
            'clipped': [d['clipped'] for d in data]
        }
    
    return jsonify({
        'timeseries': timeseries,
        'stats': stats,
        'colors': NODE_COLORS
    })


# ============================================================================
# Category 4: Kalman Filter Internals (Detailed Diagnostics)
# ============================================================================

def load_kalman_diagnostics(db_path: str, window_s: float) -> Dict:
    """Load detailed Kalman filter diagnostic data."""
    cutoff = time.time() - window_s
    node_data = {nid: [] for nid in NODE_IDS}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT created_at_s, node_id,
                   x_offset_ms, x_drift_ppm,
                   p_offset_ms2, p_drift_ppm2,
                   innov_med_ms, innov_p95_ms,
                   nis_med, nis_p95,
                   r_eff_ms2
            FROM diag_kalman
            WHERE created_at_s > ?
            ORDER BY created_at_s ASC
        """
        
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        
        for row in rows:
            t, node_id, x_offset, x_drift, p_offset, p_drift, \
                innov_med, innov_p95, nis_med, nis_p95, r_eff = row
            
            if node_id in node_data:
                node_data[node_id].append({
                    't': float(t),
                    'x_offset_ms': float(x_offset) if x_offset is not None else None,
                    'x_drift_ppm': float(x_drift) if x_drift is not None else None,
                    'p_offset_ms2': float(p_offset) if p_offset is not None else None,
                    'p_drift_ppm2': float(p_drift) if p_drift is not None else None,
                    'innov_med_ms': float(innov_med) if innov_med is not None else None,
                    'innov_p95_ms': float(innov_p95) if innov_p95 is not None else None,
                    'nis_med': float(nis_med) if nis_med is not None else None,
                    'nis_p95': float(nis_p95) if nis_p95 is not None else None,
                    'r_eff_ms2': float(r_eff) if r_eff is not None else None
                })
        
        conn.close()
    except Exception as e:
        print(f"[kalman_diagnostics] Error: {e}")
    
    return node_data


@app.route('/api/kalman/state')
def api_kalman_state():
    """Kalman state: x_offset, x_drift."""
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            'timestamps': [d['t'] for d in data],
            'x_offset_ms': [d['x_offset_ms'] for d in data],
            'x_drift_ppm': [d['x_drift_ppm'] for d in data]
        }
    
    return jsonify({'data': result, 'colors': NODE_COLORS})


@app.route('/api/kalman/covariance')
def api_kalman_covariance():
    """Kalman covariance: P_offset, P_drift."""
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            'timestamps': [d['t'] for d in data],
            'p_offset_ms2': [d['p_offset_ms2'] for d in data],
            'p_drift_ppm2': [d['p_drift_ppm2'] for d in data]
        }
    
    return jsonify({'data': result, 'colors': NODE_COLORS})


@app.route('/api/kalman/innovation')
def api_kalman_innovation():
    """Innovation: median and p95."""
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            'timestamps': [d['t'] for d in data],
            'innov_med_ms': [d['innov_med_ms'] for d in data],
            'innov_p95_ms': [d['innov_p95_ms'] for d in data]
        }
    
    return jsonify({'data': result, 'colors': NODE_COLORS})


@app.route('/api/kalman/nis')
def api_kalman_nis():
    """NIS (Normalized Innovation Squared): median and p95."""
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            'timestamps': [d['t'] for d in data],
            'nis_med': [d['nis_med'] for d in data],
            'nis_p95': [d['nis_p95'] for d in data]
        }
    
    return jsonify({'data': result, 'colors': NODE_COLORS})


@app.route('/api/kalman/r_eff')
def api_kalman_r_eff():
    """Effective measurement noise R."""
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            'timestamps': [d['t'] for d in data],
            'r_eff_ms2': [d['r_eff_ms2'] for d in data]
        }
    
    return jsonify({'data': result, 'colors': NODE_COLORS})


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    return render_template('convergence_full.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
