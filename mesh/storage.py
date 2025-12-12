# mesh/storage.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Optional


class Storage:
    """
    Zentrale SQLite-Datenbank für das Mesh.

    Tabellen:
    ----------
    1) sensor_readings
        id INTEGER PRIMARY KEY
        node_id TEXT
        t_mesh REAL
        monotonic REAL
        value REAL
        raw_json TEXT
        created_at REAL

    2) ntp_reference
        id INTEGER PRIMARY KEY
        node_id TEXT
        t_wall REAL
        t_mono REAL
        t_mesh REAL
        offset REAL
        err_mesh_vs_wall REAL
        created_at REAL

        Optional (neuere Versionen):
        peer_id TEXT
        theta_ms REAL
        rtt_ms REAL
        sigma_ms REAL
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # INTERNAL: Connection Helper
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """
        Jede Operation bekommt eine neue Connection (SQLite kann das gut),
        aber wir schützen global mit einem Lock.
        """
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=2000;")
        return conn

    # ------------------------------------------------------------------
    # SCHEMA + MIGRATION
    # ------------------------------------------------------------------

    @staticmethod
    def _table_cols(cur: sqlite3.Cursor, table: str) -> set[str]:
        return {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}

    def _ensure_schema(self) -> None:
        """
        Initialisiert Tabellen und migriert optionale Spalten (für alte DBs).
        """
        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()

            # --- sensor_readings ---
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    t_mesh REAL NOT NULL,
                    monotonic REAL,
                    value REAL,
                    raw_json TEXT,
                    created_at REAL NOT NULL
                )
                """
            )

            # --- ntp_reference (Basis) ---
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ntp_reference (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    t_wall REAL NOT NULL,
                    t_mono REAL NOT NULL,
                    t_mesh REAL NOT NULL,
                    offset REAL NOT NULL,
                    err_mesh_vs_wall REAL NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

            # --- Migration: optionale Spalten hinzufügen ---
            cols = self._table_cols(cur, "ntp_reference")

            # peer_id / theta_ms / rtt_ms / sigma_ms
            if "peer_id" not in cols:
                cur.execute("ALTER TABLE ntp_reference ADD COLUMN peer_id TEXT;")
            if "theta_ms" not in cols:
                cur.execute("ALTER TABLE ntp_reference ADD COLUMN theta_ms REAL;")
            if "rtt_ms" not in cols:
                cur.execute("ALTER TABLE ntp_reference ADD COLUMN rtt_ms REAL;")
            if "sigma_ms" not in cols:
                cur.execute("ALTER TABLE ntp_reference ADD COLUMN sigma_ms REAL;")

            # --- Indizes (UI Performance) ---
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sensor_readings_created_at ON sensor_readings(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sensor_readings_node_created ON sensor_readings(node_id, created_at);")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_ntp_reference_created_at ON ntp_reference(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ntp_reference_node_created ON ntp_reference(node_id, created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ntp_reference_peer_created ON ntp_reference(peer_id, created_at);")

            conn.commit()

    # ------------------------------------------------------------------
    # SENSOR READING API
    # ------------------------------------------------------------------

    def insert_reading(
        self,
        node_id: str,
        t_mesh: float,
        value: float,
        monotonic: Optional[float] = None,
        raw_json: Optional[str] = None,
    ) -> None:
        """
        Fügt einen Sensorwert ein.
        """
        ts = time.time()

        if isinstance(raw_json, dict):
            raw_json = json.dumps(raw_json)

        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO sensor_readings
                    (node_id, t_mesh, monotonic, value, raw_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (node_id, float(t_mesh), monotonic, float(value), raw_json, ts),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # NTP REFERENCE LOGGING API
    # ------------------------------------------------------------------

    def insert_ntp_reference(
        self,
        node_id: str,
        t_wall: float,
        t_mono: float,
        t_mesh: float,
        offset: float,
        err_mesh_vs_wall: float,
        peer_id: Optional[str] = None,
        theta_ms: Optional[float] = None,
        rtt_ms: Optional[float] = None,
        sigma_ms: Optional[float] = None,
    ) -> None:
        """
        Loggt eine Zeit-Referenz / Sync-Observation.
        created_at ist immer Sink-Clock (time.time() beim Insert).
        """
        ts = time.time()

        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()

            cols = self._table_cols(cur, "ntp_reference")

            base_cols = ["node_id", "t_wall", "t_mono", "t_mesh", "offset", "err_mesh_vs_wall", "created_at"]
            base_vals = [node_id, float(t_wall), float(t_mono), float(t_mesh), float(offset), float(err_mesh_vs_wall), ts]

            extra_cols = []
            extra_vals = []

            if "peer_id" in cols:
                extra_cols.append("peer_id")
                extra_vals.append(peer_id)

            if "theta_ms" in cols:
                extra_cols.append("theta_ms")
                extra_vals.append(theta_ms)

            if "rtt_ms" in cols:
                extra_cols.append("rtt_ms")
                extra_vals.append(rtt_ms)

            if "sigma_ms" in cols:
                extra_cols.append("sigma_ms")
                extra_vals.append(sigma_ms)

            all_cols = base_cols + extra_cols
            placeholders = ", ".join(["?"] * len(all_cols))

            cur.execute(
                f"""
                INSERT INTO ntp_reference ({", ".join(all_cols)})
                VALUES ({placeholders})
                """,
                tuple(base_vals + extra_vals),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # OPTIONAL: Query Helpers (für Debugging)
    # ------------------------------------------------------------------

    def fetch_latest_ntp_error(self, node_id: str) -> Optional[float]:
        """
        Gibt den letzten mesh_time - wallclock Fehler zurück.
        """
        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            res = cur.execute(
                """
                SELECT err_mesh_vs_wall
                FROM ntp_reference
                WHERE node_id=?
                ORDER BY id DESC
                LIMIT 1
                """,
                (node_id,),
            ).fetchone()
            return float(res[0]) if res else None
