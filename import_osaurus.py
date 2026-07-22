#!/usr/bin/env python3
"""
import_osaurus.py — normalize an Osaurus history database (sessions/turns)
into the knowledge base's common conversation schema.

Usage:
  python3 import_osaurus.py history.sqlite.plaintext out.json
  python3 import_osaurus.py history.sqlite.plaintext out.jsonl --jsonl
  python3 import_osaurus.py history.sqlite.plaintext out.json --include-thinking --include-archived

Output record (one per session):
{
  "id": "<uuid5 of 'osaurus:<session_id>' — deterministic, safe to re-import>",
  "source": "osaurus",
  "platform_id": "<original session id>",
  "title": "...",
  "created_at": "ISO8601 UTC",
  "updated_at": "ISO8601 UTC",
  "model": "...",
  "agent_id": "...",
  "archived": false,
  "message_count": N,
  "messages": [ {"role": "...", "content": "...", "timestamp": "ISO8601 or null"} ],
  "raw_path": "<path to the source sqlite file>"
}
"""

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

OSAURUS_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "kb://osaurus")


def connect_ro(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    if not p.exists():
        sys.exit(f"error: no such file: {db_path}")
    with open(p, "rb") as f:
        if not f.read(16).startswith(b"SQLite format 3"):
            sys.exit("error: not a SQLite file (still SQLCipher-encrypted?)")
    last_err = None
    for params in ("mode=ro", "mode=ro&immutable=1"):
        try:
            conn = sqlite3.connect(f"file:{p.resolve()}?{params}", uri=True)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            last_err = e
    sys.exit(f"error: could not open {db_path} read-only: {last_err}")


def iso(epoch):
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def parse_json(text, default):
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def summarize_tool_calls(tool_calls_json):
    """Render tool calls as a short readable annotation."""
    calls = parse_json(tool_calls_json, [])
    parts = []
    for call in calls if isinstance(calls, list) else []:
        fn = (call.get("function") or {}) if isinstance(call, dict) else {}
        name = fn.get("name", "unknown_tool")
        args = fn.get("arguments", "")
        if isinstance(args, str) and len(args) > 120:
            args = args[:120] + "..."
        parts.append(f"{name}({args})")
    return f"[tool call: {'; '.join(parts)}]" if parts else None


def build_messages(turns, include_thinking, include_tool_results, max_tool_result_chars):
    messages = []
    for t in turns:
        role = t["role"] or "unknown"
        content = t["content"] or ""
        extras = []

        if include_thinking and (t["thinking"] or "").strip():
            extras.append(f"[thinking] {t['thinking'].strip()}")

        annotation = summarize_tool_calls(t["tool_calls"])
        if annotation:
            extras.append(annotation)

        if include_tool_results:
            results = parse_json(t["tool_results"], {})
            if isinstance(results, dict) and results:
                for call_id, payload in results.items():
                    s = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
                    if len(s) > max_tool_result_chars:
                        s = s[:max_tool_result_chars] + "..."
                    extras.append(f"[tool result {call_id}] {s}")

        full = "\n".join(x for x in ([content] if content.strip() else []) + extras)
        if not full.strip():
            continue  # pure-plumbing turn with nothing renderable
        messages.append({
            "role": role,
            "content": full,
            "timestamp": iso(t["created_at"]),
        })
    return messages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="Osaurus history database (plaintext export)")
    ap.add_argument("out", help="output file (.json array, or .jsonl with --jsonl)")
    ap.add_argument("--jsonl", action="store_true", help="write one record per line")
    ap.add_argument("--include-archived", action="store_true", help="include archived sessions")
    ap.add_argument("--include-thinking", action="store_true", help="include model thinking traces")
    ap.add_argument("--include-tool-results", action="store_true", help="include (truncated) tool results")
    ap.add_argument("--max-tool-result-chars", type=int, default=500)
    args = ap.parse_args()

    conn = connect_ro(args.db)

    where = "" if args.include_archived else "WHERE COALESCE(archived, 0) = 0"
    sessions = conn.execute(f"SELECT * FROM sessions {where} ORDER BY created_at").fetchall()

    records, skipped_empty = [], 0
    for s in sessions:
        turns = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY seq", (s["id"],)
        ).fetchall()
        messages = build_messages(
            turns, args.include_thinking, args.include_tool_results, args.max_tool_result_chars
        )
        if not messages:
            skipped_empty += 1
            continue
        records.append({
            "id": str(uuid.uuid5(OSAURUS_NAMESPACE, f"osaurus:{s['id']}")),
            "source": "osaurus",
            "platform_id": s["id"],
            "title": s["title"],
            "created_at": iso(s["created_at"]),
            "updated_at": iso(s["updated_at"]),
            "model": s["selected_model"],
            "agent_id": s["agent_id"],
            "archived": bool(s["archived"]),
            "message_count": len(messages),
            "messages": messages,
            "raw_path": str(Path(args.db)),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        if args.jsonl:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"normalized {len(records)} conversations -> {args.out} "
          f"({skipped_empty} empty sessions skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
