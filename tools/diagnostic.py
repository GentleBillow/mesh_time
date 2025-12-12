#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeshTime Diagnostic Script
===========================
Prüft ob Logging und Visualisierungen funktionieren.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta

def check_database(db_path: str = "mesh_data.sqlite"):
    """Prüft den Inhalt der Datenbank."""
    
    if not Path(db_path).exists():
        print(f"❌ Database nicht gefunden: {db_path}")
        return False
        
    print(f"✓ Database gefunden: {db_path}\n")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # 1. Tabellen prüfen
    print("=== TABELLEN ===")
    tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in tables:
        print(f"  • {t['name']}")
    print()
    
    # 2. ntp_reference Spalten prüfen
    print("=== NTP_REFERENCE SPALTEN ===")
    cols = cur.execute("PRAGMA table_info(ntp_reference)").fetchall()
    col_names = [c['name'] for c in cols]
    for c in cols:
        print(f"  • {c['name']:<20} {c['type']:<10}")
    print()
    
    # 3. Benötigte Spalten für Link-Metriken
    needed = ['peer_id', 'theta_ms', 'rtt_ms', 'sigma_ms']
    missing = [n for n in needed if n not in col_names]
    
    if missing:
        print(f"❌ FEHLER: Spalten fehlen: {', '.join(missing)}")
        print("   → Storage Schema muss migriert werden!")
        return False
    else:
        print("✓ Alle Link-Metrik Spalten vorhanden\n")
    
    # 4. Daten prüfen
    print("=== DATENBANK INHALT ===")
    
    # Gesamt-Einträge
    total = cur.execute("SELECT COUNT(*) as cnt FROM ntp_reference").fetchone()
    print(f"Gesamt Einträge: {total['cnt']}")
    
    # Einträge mit peer_id
    with_peer = cur.execute("SELECT COUNT(*) as cnt FROM ntp_reference WHERE peer_id IS NOT NULL").fetchone()
    print(f"Mit peer_id:     {with_peer['cnt']}")
    
    # Einträge mit Link-Metriken
    with_theta = cur.execute("SELECT COUNT(*) as cnt FROM ntp_reference WHERE theta_ms IS NOT NULL").fetchone()
    with_rtt = cur.execute("SELECT COUNT(*) as cnt FROM ntp_reference WHERE rtt_ms IS NOT NULL").fetchone()
    with_sigma = cur.execute("SELECT COUNT(*) as cnt FROM ntp_reference WHERE sigma_ms IS NOT NULL").fetchone()
    
    print(f"Mit theta_ms:    {with_theta['cnt']}")
    print(f"Mit rtt_ms:      {with_rtt['cnt']}")
    print(f"Mit sigma_ms:    {with_sigma['cnt']}")
    print()
    
    # 5. Pro Node/Peer Statistik
    print("=== LINK METRIKEN PRO VERBINDUNG ===")
    stats = cur.execute("""
        SELECT 
            node_id,
            peer_id,
            COUNT(*) as cnt,
            AVG(theta_ms) as avg_theta,
            AVG(rtt_ms) as avg_rtt,
            AVG(sigma_ms) as avg_sigma,
            MAX(created_at) as last_seen
        FROM ntp_reference
        WHERE peer_id IS NOT NULL
        GROUP BY node_id, peer_id
        ORDER BY node_id, peer_id
    """).fetchall()
    
    if not stats:
        print("❌ KEINE LINK-METRIKEN GEFUNDEN!")
        print("\nMögliche Ursachen:")
        print("  1. Nodes laufen nicht")
        print("  2. sync.py ist nicht gepatcht (fehlt _log_link_metrics)")
        print("  3. Warmup-Phase läuft noch (< min_samples_before_log)")
        print("  4. CoAP Beacons schlagen fehl\n")
        return False
    
    for s in stats:
        node = s['node_id']
        peer = s['peer_id']
        cnt = s['cnt']
        theta = s['avg_theta'] or 0
        rtt = s['avg_rtt'] or 0
        sigma = s['avg_sigma'] or 0
        last = datetime.fromtimestamp(s['last_seen']).strftime("%H:%M:%S")
        
        print(f"  {node} → {peer}:")
        print(f"    Samples: {cnt}")
        print(f"    θ_avg:   {theta:>7.2f} ms")
        print(f"    RTT_avg: {rtt:>7.2f} ms")
        print(f"    σ_avg:   {sigma:>7.2f} ms")
        print(f"    Zuletzt: {last}")
    print()
    
    # 6. Zeitliche Verteilung (letzte 10 Minuten)
    print("=== ZEITLICHE VERTEILUNG (letzte 10 Min) ===")
    cutoff = datetime.now() - timedelta(minutes=10)
    cutoff_ts = cutoff.timestamp()
    
    recent = cur.execute("""
        SELECT 
            strftime('%Y-%m-%d %H:%M', datetime(created_at, 'unixepoch')) as minute,
            COUNT(*) as cnt
        FROM ntp_reference
        WHERE created_at >= ? AND peer_id IS NOT NULL
        GROUP BY minute
        ORDER BY minute DESC
        LIMIT 10
    """, (cutoff_ts,)).fetchall()
    
    if recent:
        for r in recent:
            bar = '█' * min(50, r['cnt'])
            print(f"  {r['minute']}: {bar} ({r['cnt']})")
    else:
        print("  ❌ Keine Daten in den letzten 10 Minuten!")
    print()
    
    # 7. Sensor Readings
    print("=== SENSOR READINGS ===")
    sensor_cnt = cur.execute("SELECT COUNT(*) as cnt FROM sensor_readings").fetchone()
    print(f"Gesamt: {sensor_cnt['cnt']}")
    
    sensor_by_node = cur.execute("""
        SELECT node_id, COUNT(*) as cnt
        FROM sensor_readings
        GROUP BY node_id
    """).fetchall()
    
    for s in sensor_by_node:
        print(f"  {s['node_id']}: {s['cnt']}")
    print()
    
    conn.close()
    
    # Erfolgs-Check
    if with_peer['cnt'] > 0 and with_theta['cnt'] > 0:
        print("✅ DATABASE STATUS: OK")
        print("   Link-Metriken werden geloggt!")
        return True
    else:
        print("⚠️  DATABASE STATUS: Unvollständig")
        print("   Einige Metriken fehlen noch")
        return False


def check_config(config_path: str = "config/nodes.json"):
    """Prüft die Konfiguration."""
    import json
    
    if not Path(config_path).exists():
        print(f"❌ Config nicht gefunden: {config_path}")
        return False
    
    print(f"\n=== CONFIG CHECK: {config_path} ===")
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    # Nodes identifizieren
    nodes = {k: v for k, v in cfg.items() if k != "sync" and isinstance(v, dict)}
    print(f"Nodes gefunden: {', '.join(nodes.keys())}\n")
    
    # Sync-Settings
    sync = cfg.get("sync", {})
    print("Sync Settings:")
    print(f"  controller:              {sync.get('controller', 'N/A')}")
    print(f"  min_beacons_for_warmup:  {sync.get('min_beacons_for_warmup', 'N/A')}")
    print(f"  min_samples_before_log:  {sync.get('min_samples_before_log', 'N/A')}")
    print(f"  beacon_period_s:         {sync.get('beacon_period_s', 'N/A')}")
    print()
    
    # Node Details
    for nid, ncfg in nodes.items():
        print(f"Node {nid}:")
        print(f"  IP:        {ncfg.get('ip', 'N/A')}")
        print(f"  Neighbors: {', '.join(ncfg.get('neighbors', []))}")
        print(f"  Parent:    {ncfg.get('parent', 'N/A')}")
        print(f"  DB Path:   {ncfg.get('db_path', 'None (keine DB)')}")
        
        node_sync = ncfg.get("sync", {})
        if node_sync:
            print(f"  is_root:   {node_sync.get('is_root', False)}")
        print()
    
    return True


def main():
    print("=" * 60)
    print("MESHTIME DIAGNOSTIC TOOL")
    print("=" * 60)
    print()
    
    # Config Check
    config_ok = check_config()
    
    # Database Check
    db_ok = check_database()
    
    print("\n" + "=" * 60)
    if config_ok and db_ok:
        print("✅ ALLE CHECKS BESTANDEN")
        print("\nNächste Schritte:")
        print("  1. Web UI öffnen: http://192.168.178.49:5000/")
        print("  2. Charts sollten jetzt Daten zeigen")
        print("  3. Debug Monitor testen: python tools/debug_monitor.py")
    else:
        print("⚠️  EINIGE CHECKS FEHLGESCHLAGEN")
        print("\nTroubleshooting:")
        print("  1. Sind die Nodes gestartet?")
        print("  2. Wurde sync.py gepatcht? (siehe sync_fixed.py)")
        print("  3. Warmup-Phase abgeschlossen? (warte ~30 Sekunden)")
        print("  4. CoAP funktioniert? (check mit debug_monitor.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
