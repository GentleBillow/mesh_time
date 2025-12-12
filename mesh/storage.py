# mesh/storage.py
# -*- coding: utf-8 -*-

import sqlite3
import threading
from typing import Optional
import time
import json


class Storage:
    """
    Zentrale SQLite-Datenbank für das Mesh.

    Tabellen:
    ----------
    1) sensor_readings
        id INTEGER PRIMARY KEY
        node_id TEXT
        t_mesh REAL               -- Mesh-Zeitstempel der Messung
        monotonic REAL            -- time.monotonic() beim Empfang
        value REAL                -- Sensorwert
        raw_json TEXT             -- Original-JSON (optional)
        created_at REAL           -- time.time()

    2) ntp_reference
        id INTEGER PRIMARY KEY
        node_id TEXT
        t_wall REAL               -- time.time() (NTP-stabil)
        t_mono REAL               -- time.monotonic()
        t_mesh REAL               -- mesh_time()
        offset REAL               -- aktueller Offset
        err_mesh_vs_wall REAL     -- t_mesh - t_wall
        created_at REAL           -- time.time()
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    #  INTERNAL: Connection Helper
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """
        Jede Operation bekommt eine neue Connection (SQLite kann das gut),
        aber wir schützen sie global mit einem Lock.
        """
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    # ------------------------------------------------------------------
    #  SCHEMA
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """
        Initialisiert alle Tabellen.
        """
        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()

            # Sensor-Readings
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

            # NTP/Wallclock-Referenz
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

            conn.commit()

    # ------------------------------------------------------------------
    #  SENSOR READING API
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
                (node_id, t_mesh, monotonic, value, raw_json, ts),
            )
            conn.commit()

    # ------------------------------------------------------------------
    #  NTP REFERENCE LOGGING API
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
        ts = time.time()

        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()

            # Welche Spalten existieren wirklich? (für alte DBs)
            cols = {row[1] for row in cur.execute("PRAGMA table_info(ntp_reference)").fetchall()}

            base_cols = ["node_id", "t_wall", "t_mono", "t_mesh", "offset", "err_mesh_vs_wall", "created_at"]
            base_vals = [node_id, t_wall, t_mono, t_mesh, offset, err_mesh_vs_wall, ts]

            extra = []
            extra_vals = []

            if "peer_id" in cols:
                extra.append("peer_id")
                extra_vals.append(peer_id)

            if "theta_ms" in cols:
                extra.append("theta_ms")
                extra_vals.append(theta_ms)

            if "rtt_ms" in cols:
                extra.append("rtt_ms")
                extra_vals.append(rtt_ms)

            if "sigma_ms" in cols:
                extra.append("sigma_ms")
                extra_vals.append(sigma_ms)

            all_cols = base_cols + extra
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
    #  OPTIONAL: Query Helpers (für Debugging)
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
            return res[0] if res else None
