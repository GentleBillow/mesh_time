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
        self._cols_cache = {}

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

    def _table_cols_cached(self, cur: sqlite3.Cursor, table: str) -> set[str]:
        if table not in self._cols_cache:
            self._cols_cache[table] = {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}
        return self._cols_cache[table]

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

            # --- mesh_clock (per tick per node; clean source for relative plots) ---
            cur.execute("""
            CREATE TABLE IF NOT EXISTS mesh_clock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_s REAL NOT NULL,   -- time.time() at insert (sink clock)
                node_id TEXT NOT NULL,

                t_wall_s REAL,                -- time.time()
                t_mono_s REAL,                -- time.monotonic()
                t_mesh_s REAL,                -- mesh_time()

                offset_s REAL,                -- controller offset state
                err_mesh_vs_wall_s REAL       -- t_mesh_s - t_wall_s (diagnostic)
            )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mesh_clock_time ON mesh_clock(created_at_s);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mesh_clock_node_time ON mesh_clock(node_id, created_at_s);")


            # --- obs_link (raw per peer measurement) ---
            cur.execute("""
            CREATE TABLE IF NOT EXISTS obs_link (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_s REAL NOT NULL,
                node_id TEXT NOT NULL,
                peer_id TEXT NOT NULL,

                theta_ms REAL,
                rtt_ms REAL,
                sigma_ms REAL,

                t1_s REAL, t2_s REAL, t3_s REAL, t4_s REAL,

                accepted INTEGER,
                weight REAL,
                reject_reason TEXT
            )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_link_time ON obs_link(created_at_s);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_link_pair ON obs_link(node_id, peer_id, created_at_s);")

            # --- diag_controller (per tick per node) ---
            cur.execute("""
            CREATE TABLE IF NOT EXISTS diag_controller (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_s REAL NOT NULL,
                node_id TEXT NOT NULL,

                dt_s REAL,
                delta_desired_ms REAL,
                delta_applied_ms REAL,
                slew_clipped INTEGER,

                max_slew_ms_s REAL,
                eff_eta REAL
            )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_diag_ctrl_time ON diag_controller(created_at_s);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_diag_ctrl_node ON diag_controller(node_id, created_at_s);")

            # --- diag_kalman (aggregated per tick per node) ---
            cur.execute("""
            CREATE TABLE IF NOT EXISTS diag_kalman (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_s REAL NOT NULL,
                node_id TEXT NOT NULL,

                n_meas INTEGER,
                innov_med_ms REAL,
                innov_p95_ms REAL,
                nis_med REAL,
                nis_p95 REAL,

                x_offset_ms REAL,
                x_drift_ppm REAL,

                p_offset_ms2 REAL,
                p_drift_ppm2 REAL,

                r_eff_ms2 REAL
            )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_diag_kalman_time ON diag_kalman(created_at_s);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_diag_kalman_node ON diag_kalman(node_id, created_at_s);")

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

            # --- Migration: controller debug fields (for dashboard) ---
            if "delta_desired_ms" not in cols:
                cur.execute("ALTER TABLE ntp_reference ADD COLUMN delta_desired_ms REAL;")
            if "delta_applied_ms" not in cols:
                cur.execute("ALTER TABLE ntp_reference ADD COLUMN delta_applied_ms REAL;")
            if "dt_s" not in cols:
                cur.execute("ALTER TABLE ntp_reference ADD COLUMN dt_s REAL;")
            if "slew_clipped" not in cols:
                cur.execute("ALTER TABLE ntp_reference ADD COLUMN slew_clipped INTEGER;")  # 0/1


            # --- Indizes (UI Performance) ---
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sensor_readings_created_at ON sensor_readings(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sensor_readings_node_created ON sensor_readings(node_id, created_at);")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_ntp_reference_created_at ON ntp_reference(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ntp_reference_node_created ON ntp_reference(node_id, created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ntp_reference_peer_created ON ntp_reference(peer_id, created_at);")


            conn.commit()

        self._cols_cache.clear()

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
        # controller debug (optional)
        delta_desired_ms: Optional[float] = None,
        delta_applied_ms: Optional[float] = None,
        dt_s: Optional[float] = None,
        slew_clipped: Optional[bool] = None,
    ) -> None:
        """
        Loggt eine Zeit-Referenz / Sync-Observation.
        created_at ist immer Sink-Clock (time.time() beim Insert).
        """
        ts = time.time()

        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()

            cols = self._table_cols_cached(cur, "ntp_reference")

            base_cols = [
                "node_id",
                "t_wall",
                "t_mono",
                "t_mesh",
                "offset",
                "err_mesh_vs_wall",
                "created_at",
            ]
            base_vals = [
                node_id,
                float(t_wall),
                float(t_mono),
                float(t_mesh),
                float(offset),
                float(err_mesh_vs_wall),
                ts,
            ]

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

            # ---- controller debug fields ----
            if "delta_desired_ms" in cols:
                extra_cols.append("delta_desired_ms")
                extra_vals.append(delta_desired_ms)

            if "delta_applied_ms" in cols:
                extra_cols.append("delta_applied_ms")
                extra_vals.append(delta_applied_ms)

            if "dt_s" in cols:
                extra_cols.append("dt_s")
                extra_vals.append(dt_s)

            if "slew_clipped" in cols:
                extra_cols.append("slew_clipped")
                extra_vals.append(
                    None if slew_clipped is None else (1 if bool(slew_clipped) else 0)
                )

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

    def insert_mesh_clock(
        self,
        node_id: str,
        t_wall_s: float,
        t_mono_s: float,
        t_mesh_s: float,
        offset_s: float,
        err_mesh_vs_wall_s: float,
    ) -> None:
        ts = time.time()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO mesh_clock (
                  created_at_s, node_id,
                  t_wall_s, t_mono_s, t_mesh_s,
                  offset_s, err_mesh_vs_wall_s
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    ts, node_id,
                    float(t_wall_s), float(t_mono_s), float(t_mesh_s),
                    float(offset_s), float(err_mesh_vs_wall_s),
                ),
            )
            conn.commit()


    def insert_obs_link(
            self,
            node_id: str,
            peer_id: str,
            theta_ms: Optional[float] = None,
            rtt_ms: Optional[float] = None,
            sigma_ms: Optional[float] = None,
            t1_s: Optional[float] = None,
            t2_s: Optional[float] = None,
            t3_s: Optional[float] = None,
            t4_s: Optional[float] = None,
            accepted: Optional[bool] = None,
            weight: Optional[float] = None,
            reject_reason: Optional[str] = None,
    ) -> None:
        ts = time.time()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO obs_link (
                  created_at_s,node_id,peer_id,theta_ms,rtt_ms,sigma_ms,
                  t1_s,t2_s,t3_s,t4_s,accepted,weight,reject_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts, node_id, peer_id, theta_ms, rtt_ms, sigma_ms,
                    t1_s, t2_s, t3_s, t4_s,
                    None if accepted is None else (1 if bool(accepted) else 0),
                    weight, reject_reason
                )
            )
            conn.commit()

    def insert_diag_controller(
            self,
            node_id: str,
            dt_s: Optional[float],
            delta_desired_ms: Optional[float],
            delta_applied_ms: Optional[float],
            slew_clipped: Optional[bool],
            max_slew_ms_s: Optional[float] = None,
            eff_eta: Optional[float] = None,
    ) -> None:
        ts = time.time()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO diag_controller (
                  created_at_s,node_id,dt_s,delta_desired_ms,delta_applied_ms,slew_clipped,
                  max_slew_ms_s,eff_eta
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    ts, node_id, dt_s, delta_desired_ms, delta_applied_ms,
                    None if slew_clipped is None else (1 if bool(slew_clipped) else 0),
                    max_slew_ms_s, eff_eta
                )
            )
            conn.commit()

    def insert_diag_kalman(
            self,
            node_id: str,
            n_meas: int,
            innov_med_ms: Optional[float],
            innov_p95_ms: Optional[float],
            nis_med: Optional[float],
            nis_p95: Optional[float],
            x_offset_ms: Optional[float],
            x_drift_ppm: Optional[float],
            p_offset_ms2: Optional[float],
            p_drift_ppm2: Optional[float],
            r_eff_ms2: Optional[float],
    ) -> None:
        ts = time.time()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO diag_kalman (
                  created_at_s,node_id,n_meas,
                  innov_med_ms,innov_p95_ms,nis_med,nis_p95,
                  x_offset_ms,x_drift_ppm,p_offset_ms2,p_drift_ppm2,r_eff_ms2
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts, node_id, int(n_meas),
                    innov_med_ms, innov_p95_ms, nis_med, nis_p95,
                    x_offset_ms, x_drift_ppm, p_offset_ms2, p_drift_ppm2, r_eff_ms2
                )
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
