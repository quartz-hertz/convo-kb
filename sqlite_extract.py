#!/usr/bin/env python3
"""
sqlite_extract.py — inspect and dump any SQLite database (e.g. an Osaurus
plaintext export) to text formats you can read, share, or feed to an importer.

Usage:
  python3 sqlite_extract.py inspect  path/to/db            # tables, columns, counts, samples
  python3 sqlite_extract.py dump     path/to/db out.json   # full dump, all tables -> one JSON file
  python3 sqlite_extract.py dump     path/to/db out.json --tables chats messages
  python3 sqlite_extract.py sqltext  path/to/db out.sql    # .dump-style SQL text (project-friendly)

Safe by design: opens the database read-only via URI, never writes to it.
Run against a *copy* of the DB anyway if the app that owns it is running.
"""

import argparse
import base64
import json
import sqlite3
import sys
from pathlib import Path

TRUNCATE_SAMPLE = 200  # chars shown per value in inspect mode


def connect_ro(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    if not p.exists():
        sys.exit(f"error: no such file: {db_path}")
    # Detect SQLCipher / non-SQLite files early with a clear message.
    with open(p, "rb") as f:
        header = f.read(16)
    if not header.startswith(b"SQLite format 3"):
        sys.exit(
            "error: this file does not have a SQLite header.\n"
            "It is probably still SQLCipher-encrypted (or not a database at all).\n"
            "Use Osaurus Settings -> Storage -> Export plaintext backup, and run "
            "this script on the exported copy."
        )
    # WAL-mode databases (read/write version 2 in the header — typical of app
    # exports like Osaurus's) can refuse a plain read-only open, because reading
    # WAL normally needs -shm/-wal sidecar files created next to the database.
    # immutable=1 tells SQLite the file is a static snapshot: no locks, no
    # sidecars. Safe here because we only ever operate on exported copies.
    last_err = None
    for params in ("mode=ro", "mode=ro&immutable=1"):
        try:
            conn = sqlite3.connect(f"file:{p.resolve()}?{params}", uri=True)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")  # force a real read
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            last_err = e
    sys.exit(f"error: could not open {db_path} read-only: {last_err}")


def jsonable(value):
    """Make any SQLite value JSON-serializable."""
    if isinstance(value, bytes):
        # Try UTF-8 first (lots of 'BLOB' columns are really text or JSON)
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"__blob_base64__": base64.b64encode(value).decode("ascii")}
    return value


def maybe_parse_json(value):
    """If a text value looks like embedded JSON, parse it so the dump is structured."""
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except (json.JSONDecodeError, RecursionError):
                return value
    return value


def list_tables(conn):
    rows = conn.execute(
        "SELECT name, type FROM sqlite_master "
        "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


def cmd_inspect(args):
    conn = connect_ro(args.db)
    tables = list_tables(conn)
    if not tables:
        print("No user tables found.")
        return
    print(f"Database: {args.db}")
    print(f"Tables ({len(tables)}): {', '.join(tables)}\n")
    for t in tables:
        cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
        try:
            count = conn.execute(f'SELECT COUNT(*) AS c FROM "{t}"').fetchone()["c"]
        except sqlite3.DatabaseError as e:
            print(f"== {t} ==  (unreadable: {e})\n")
            continue
        print(f"== {t} ==  ({count} rows)")
        for c in cols:
            pk = "  PK" if c["pk"] else ""
            print(f"   {c['name']:<28} {c['type'] or 'ANY'}{pk}")
        if count and not args.no_samples:
            sample = conn.execute(f'SELECT * FROM "{t}" LIMIT {args.samples}').fetchall()
            for i, row in enumerate(sample, 1):
                print(f"   --- sample row {i} ---")
                for k in row.keys():
                    v = jsonable(row[k])
                    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                    s = s.replace("\n", "\\n")
                    if len(s) > TRUNCATE_SAMPLE:
                        s = s[:TRUNCATE_SAMPLE] + f"... [{len(s)} chars]"
                    print(f"   {k}: {s}")
        print()


def cmd_dump(args):
    conn = connect_ro(args.db)
    tables = args.tables or list_tables(conn)
    out = {}
    for t in tables:
        try:
            rows = conn.execute(f'SELECT * FROM "{t}"').fetchall()
        except sqlite3.DatabaseError as e:
            print(f"warning: skipping table {t}: {e}", file=sys.stderr)
            continue
        out[t] = [
            {k: maybe_parse_json(jsonable(r[k])) for k in r.keys()} for r in rows
        ]
        print(f"dumped {t}: {len(out[t])} rows", file=sys.stderr)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)


def cmd_sqltext(args):
    conn = connect_ro(args.db)
    with open(args.out, "w", encoding="utf-8") as f:
        for line in conn.iterdump():
            f.write(line + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("inspect", help="print tables, columns, row counts, sample rows")
    p1.add_argument("db")
    p1.add_argument("--samples", type=int, default=2, help="sample rows per table (default 2)")
    p1.add_argument("--no-samples", action="store_true", help="schema and counts only, no row content")
    p1.set_defaults(func=cmd_inspect)

    p2 = sub.add_parser("dump", help="dump tables to a single JSON file")
    p2.add_argument("db")
    p2.add_argument("out")
    p2.add_argument("--tables", nargs="*", help="only these tables (default: all)")
    p2.set_defaults(func=cmd_dump)

    p3 = sub.add_parser("sqltext", help="write a .dump-style SQL text file")
    p3.add_argument("db")
    p3.add_argument("out")
    p3.set_defaults(func=cmd_sqltext)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
