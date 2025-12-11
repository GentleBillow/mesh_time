# mesh/storage.py
# -*- coding: utf-8 -*-

import sqlite3
import threading
from typing import Optional, Dict, Any
import time


class TimeSeriesDB:
    """
    Sehr einfache SQLite-DB für Sensordaten.

    Eine Tabelle:
      sensor_readings(
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id     TEXT,
        t_mesh      REAL,
        monotonic   REAL,
        value       REAL,
        raw_json    TEXT,
        created_at  REAL
      )

    Idee:
      - node_id: wer hat gemessen
      - t_mesh: Mesh-Zeitstempel der Messung
      - monotonic: (optional) lokale monotonic Zeit beim Empfang
      - value: Sensorwert (float)
      - raw_json: Original-Payload als JSON-String (Debug/Erweiterbarkeit)
      - created_at: time.time() beim Insert
    """

    def __init__(self, path: str):
        self.path = path
        # SQLite ist nicht super threadsafe → einfacher Lock
        self._lock = threading.Lock()
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        # isolation_level=None → autocommit
        return sqlite3.connect(self.path, isolation_level=None)

    def _ensure_schema(self) -> None:
        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
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
            conn.commit()

    def insert_reading(
        self,
        node_id: str,
        t_mesh: float,
        value: float,
        monotonic: Optional[float] = None,
        raw_json: Optional[str] = None,
    ) -> None:
        """
        Ein Messpunkt einfügen.
        """
        ts = time.time()
        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO sensor_readings (node_id, t_mesh, monotonic, value, raw_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (node_id, t_mesh, monotonic, value, raw_json, ts),
            )
            conn.commit()
