# -*- coding: utf-8 -*-
"""
Usage:
  python3 tools/db_inspect.py schema
  python3 tools/db_inspect.py cols ntp_reference
  python3 tools/db_inspect.py tail ntp_reference 20
  python3 tools/db_inspect.py latest_deltas 20
  python3 tools/db_inspect.py counts
  python3 tools/db_inspect.py timecols

"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]      # ~/mesh_time
DB_PATH  = BASE_DIR / "mesh_data.sqlite"


def utc(ts: float) -> str:
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


def q(conn, sql, args=()):
    cur = conn.cursor()
    cur.execute(sql, args)
    return cur.fetchall(), [d[0] for d in cur.description] if cur.description else []


def print_table(rows, cols):
    if not cols:
        print("(no columns)")
        return
    widths = [len(c) for c in cols]
    for r in rows:
        for i, c in enumerate(cols):
            v = r[i]
            s = "" if v is None else str(v)
            widths[i] = max(widths[i], len(s))
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*cols))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        out = []
        for i in range(len(cols)):
            v = r[i]
            out.append("" if v is None else str(v))
        print(fmt.format(*out))


def schema(conn):
    rows, cols = q(conn, "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;")
    for name, sql in rows:
        print(f"\n=== {name} ===")
        print(sql)


def cols(conn, table):
    rows, cols_ = q(conn, f"PRAGMA table_info({table});")
    print_table(rows, cols_)


def tail(conn, table, n):
    rows, cols_ = q(conn, f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?;", (int(n),))
    print_table(rows, cols_)


def latest_deltas(conn, n):
    # tries to show the new control logging columns if present
    info, _ = q(conn, "PRAGMA table_info(ntp_reference);")
    colset = {r[1] for r in info}
    want = ["created_at","node_id","peer_id","theta_ms","rtt_ms","sigma_ms","offset",
            "delta_desired_ms","delta_applied_ms","dt_s","slew_clipped"]
    have = [c for c in want if c in colset]

    if not have:
        print("No matching columns found in ntp_reference.")
        return

    sql = "SELECT " + ",".join(have) + " FROM ntp_reference ORDER BY id DESC LIMIT ?;"
    rows, cols_ = q(conn, sql, (int(n),))
    # Pretty-print created_at as UTC if present
    if "created_at" in cols_:
        i = cols_.index("created_at")
        rows = [list(r) for r in rows]
        for r in rows:
            r[i] = f"{r[i]} ({utc(r[i])})"
    print_table(rows, cols_)

def counts(conn):
    # Count rows for the tables we care about (v1 + v2)
    tables = ["ntp_reference", "sensor_readings", "obs_link", "diag_controller", "diag_kalman", "mesh_clock"]
    out = []
    for t in tables:
        try:
            rows, _ = q(conn, "SELECT COUNT(*) AS n FROM %s;" % t)
            n = rows[0][0] if rows else 0
            out.append((t, n))
        except Exception:
            out.append((t, "MISSING"))
    print_table(out, ["table", "rows"])


def timecols(conn):
    # Show which timestamp column exists in each table: created_at vs created_at_s
    tables = ["ntp_reference", "sensor_readings", "obs_link", "diag_controller", "diag_kalman", "mesh_clock"]
    out = []
    for t in tables:
        try:
            info, _ = q(conn, "PRAGMA table_info(%s);" % t)
            colset = {r[1] for r in info}
            has_created_at = "created_at" in colset
            has_created_at_s = "created_at_s" in colset
            out.append((t, int(has_created_at), int(has_created_at_s)))
        except Exception:
            out.append((t, "", ""))
    print_table(out, ["table", "created_at", "created_at_s"])



def main():
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        sys.exit(2)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "schema"

    conn = sqlite3.connect(str(DB_PATH))
    try:
        if cmd == "schema":
            schema(conn)
        elif cmd == "counts":
            counts(conn)
        elif cmd == "timecols":
            timecols(conn)
        elif cmd == "cols":
            table = sys.argv[2] if len(sys.argv) > 2 else "ntp_reference"
            cols(conn, table)
        elif cmd == "tail":
            table = sys.argv[2] if len(sys.argv) > 2 else "ntp_reference"
            n = sys.argv[3] if len(sys.argv) > 3 else 20
            tail(conn, table, n)
        elif cmd == "latest_deltas":
            n = sys.argv[2] if len(sys.argv) > 2 else 20
            latest_deltas(conn, n)
        else:
            print("Unknown command.")
            sys.exit(2)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
