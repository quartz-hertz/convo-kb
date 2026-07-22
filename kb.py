#!/usr/bin/env python3
"""
kb.py — SQLite store and query CLI for the conversation knowledge base.

Usage:
  python3 kb.py import  osaurus.json chatgpt.json claude.json   [--db kb.sqlite]
  python3 kb.py search  "actor model concurrency"               [--tag swift --min-importance 3 --source chatgpt --limit 10]
  python3 kb.py filter  --tag sysadmin --flag has_code          [--min-importance 3]
  python3 kb.py show    <id or unique prefix>                   [--full]
  python3 kb.py stats
  python3 kb.py tags

Import is idempotent: unchanged conversations are skipped, changed ones
(newer updated_at) are re-imported and marked for re-classification.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  platform_id TEXT,
  title TEXT,
  created_at TEXT,
  updated_at TEXT,
  model TEXT,
  message_count INTEGER,
  summary TEXT,
  importance INTEGER,
  classified_at TEXT,
  raw_path TEXT,
  meta TEXT
);
CREATE TABLE IF NOT EXISTS messages (
  conversation_id TEXT,
  seq INTEGER,
  role TEXT,
  content TEXT,
  timestamp TEXT,
  PRIMARY KEY (conversation_id, seq)
);
CREATE TABLE IF NOT EXISTS tags (
  conversation_id TEXT,
  tag TEXT,
  kind TEXT,              -- 'topic' | 'tag' | 'proposed' | 'language'
  UNIQUE (conversation_id, tag, kind)
);
CREATE TABLE IF NOT EXISTS flags (
  conversation_id TEXT PRIMARY KEY,
  has_attachments INTEGER, has_code INTEGER, has_images INTEGER,
  is_sysadmin INTEGER, is_research INTEGER, is_creative INTEGER,
  concluded INTEGER
);
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
  id UNINDEXED, title, summary, full_text
);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_conv_source ON conversations(source);
"""

CORE_FIELDS = {"id", "source", "platform_id", "title", "created_at", "updated_at",
               "model", "message_count", "messages", "raw_path"}


def open_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def full_text_of(messages):
    return "\n".join(f"{m.get('role','?')}: {m.get('content','')}" for m in messages)


def fts_upsert(conn, cid, title, summary, full_text):
    conn.execute("DELETE FROM conversations_fts WHERE id = ?", (cid,))
    conn.execute(
        "INSERT INTO conversations_fts (id, title, summary, full_text) VALUES (?,?,?,?)",
        (cid, title or "", summary or "", full_text or ""),
    )


def cmd_import(args):
    conn = open_db(args.db)
    added = updated = unchanged = 0
    for path in args.files:
        text = Path(path).read_text(encoding="utf-8")
        records = ([json.loads(line) for line in text.splitlines() if line.strip()]
                   if path.endswith(".jsonl") else json.loads(text))
        for r in records:
            cid = r["id"]
            existing = conn.execute(
                "SELECT updated_at FROM conversations WHERE id = ?", (cid,)
            ).fetchone()
            if existing and existing["updated_at"] == r.get("updated_at"):
                unchanged += 1
                continue

            meta = {k: v for k, v in r.items() if k not in CORE_FIELDS}
            conn.execute(
                """INSERT INTO conversations
                   (id, source, platform_id, title, created_at, updated_at, model,
                    message_count, raw_path, meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     title=excluded.title, updated_at=excluded.updated_at,
                     model=excluded.model, message_count=excluded.message_count,
                     raw_path=excluded.raw_path, meta=excluded.meta,
                     classified_at=NULL""",
                (cid, r["source"], r.get("platform_id"), r.get("title"),
                 r.get("created_at"), r.get("updated_at"), r.get("model"),
                 r.get("message_count"), r.get("raw_path"),
                 json.dumps(meta, ensure_ascii=False) if meta else None),
            )
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
            for i, m in enumerate(r.get("messages", [])):
                conn.execute(
                    "INSERT INTO messages (conversation_id, seq, role, content, timestamp) VALUES (?,?,?,?,?)",
                    (cid, i, m.get("role"), m.get("content"), m.get("timestamp")),
                )
            row = conn.execute("SELECT summary FROM conversations WHERE id=?", (cid,)).fetchone()
            fts_upsert(conn, cid, r.get("title"), row["summary"] if row else None,
                       full_text_of(r.get("messages", [])))
            if existing:
                updated += 1
            else:
                added += 1
    conn.commit()
    print(f"imported: {added} new, {updated} updated (re-classification pending), {unchanged} unchanged")


def print_results(rows):
    if not rows:
        print("no results")
        return
    for r in rows:
        imp = f" imp={r['importance']}" if r["importance"] is not None else ""
        title = (r["title"] or "(untitled)")[:70]
        summary = (r["summary"] or "").strip()
        print(f"{r['id'][:8]}  [{r['source']:>8}]{imp}  {r['created_at'] or '?':<25}  {title}")
        if summary:
            print(f"          {summary[:160]}")


def apply_filters(args, where, params):
    if getattr(args, "source", None):
        where.append("c.source = ?"); params.append(args.source)
    if getattr(args, "min_importance", None) is not None:
        where.append("c.importance >= ?"); params.append(args.min_importance)
    for tag in getattr(args, "tag", None) or []:
        where.append("c.id IN (SELECT conversation_id FROM tags WHERE tag = ?)")
        params.append(tag)
    for flag in getattr(args, "flag", None) or []:
        col = flag.replace("-", "_")
        if col not in ("has_attachments", "has_code", "has_images",
                       "is_sysadmin", "is_research", "is_creative", "concluded"):
            sys.exit(f"unknown flag: {flag}")
        where.append(f"c.id IN (SELECT conversation_id FROM flags WHERE {col} = 1)")
    return where, params


def cmd_search(args):
    conn = open_db(args.db)
    where, params = apply_filters(args, ["conversations_fts MATCH ?"], [args.query])
    rows = conn.execute(
        f"""SELECT c.* FROM conversations_fts f
            JOIN conversations c ON c.id = f.id
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(c.importance, 0) DESC, bm25(conversations_fts)
            LIMIT ?""",
        params + [args.limit],
    ).fetchall()
    print_results(rows)


def cmd_filter(args):
    conn = open_db(args.db)
    where, params = apply_filters(args, ["1=1"], [])
    rows = conn.execute(
        f"""SELECT c.* FROM conversations c WHERE {' AND '.join(where)}
            ORDER BY COALESCE(c.importance,0) DESC, c.updated_at DESC LIMIT ?""",
        params + [args.limit],
    ).fetchall()
    print_results(rows)


def cmd_show(args):
    conn = open_db(args.db)
    rows = conn.execute(
        "SELECT * FROM conversations WHERE id LIKE ?", (args.id + "%",)
    ).fetchall()
    if not rows:
        sys.exit("not found")
    if len(rows) > 1:
        sys.exit("ambiguous prefix:\n" + "\n".join(f"  {r['id']}  {r['title']}" for r in rows))
    c = rows[0]
    print(f"id:         {c['id']}")
    print(f"source:     {c['source']}   model: {c['model'] or '-'}")
    print(f"title:      {c['title']}")
    print(f"created:    {c['created_at']}")
    print(f"importance: {c['importance']}   classified: {c['classified_at'] or 'no'}")
    tags = conn.execute("SELECT tag, kind FROM tags WHERE conversation_id=? ORDER BY kind, tag", (c["id"],)).fetchall()
    if tags:
        by_kind = {}
        for t in tags:
            by_kind.setdefault(t["kind"], []).append(t["tag"])
        for kind in ("topic", "tag", "language", "proposed"):
            if kind in by_kind:
                print(f"{kind + 's:':<12}{', '.join(by_kind[kind])}")
    if c["summary"]:
        print(f"summary:    {c['summary']}")
    print("-" * 72)
    msgs = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY seq", (c["id"],)
    ).fetchall()
    for m in msgs:
        body = m["content"] if args.full else (m["content"][:400] + ("..." if len(m["content"]) > 400 else ""))
        print(f"\n[{m['role']}]\n{body}")


def cmd_reset(args):
    conn = open_db(args.db)
    if args.mock:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM conversations WHERE summary LIKE '(mock)%'")]
    elif args.all:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM conversations WHERE classified_at IS NOT NULL")]
    elif args.ids:
        ids = []
        for prefix in args.ids:
            rows = conn.execute("SELECT id FROM conversations WHERE id LIKE ?", (prefix + "%",)).fetchall()
            if len(rows) != 1:
                sys.exit(f"prefix '{prefix}' matched {len(rows)} conversations")
            ids.append(rows[0]["id"])
    else:
        sys.exit("specify --mock, --all, or one or more id prefixes")

    for cid in ids:
        conn.execute("UPDATE conversations SET summary=NULL, importance=NULL, classified_at=NULL WHERE id=?", (cid,))
        conn.execute("DELETE FROM tags WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM flags WHERE conversation_id=?", (cid,))
        row = conn.execute("SELECT title, full_text FROM conversations_fts WHERE id=?", (cid,)).fetchone()
        conn.execute("DELETE FROM conversations_fts WHERE id=?", (cid,))
        conn.execute("INSERT INTO conversations_fts (id, title, summary, full_text) VALUES (?,?,?,?)",
                     (cid, row["title"] if row else "", "", row["full_text"] if row else ""))
    conn.commit()
    print(f"reset {len(ids)} conversations (now pending classification)")


def cmd_stats(args):
    conn = open_db(args.db)
    total = conn.execute("SELECT COUNT(*) c FROM conversations").fetchone()["c"]
    print(f"conversations: {total}")
    for r in conn.execute("SELECT source, COUNT(*) c FROM conversations GROUP BY source ORDER BY c DESC"):
        print(f"  {r['source']:>10}: {r['c']}")
    classified = conn.execute("SELECT COUNT(*) c FROM conversations WHERE classified_at IS NOT NULL").fetchone()["c"]
    msgs = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
    print(f"messages: {msgs}")
    print(f"classified: {classified}/{total}")
    if classified:
        print("importance distribution:")
        for r in conn.execute("SELECT importance, COUNT(*) c FROM conversations WHERE importance IS NOT NULL GROUP BY importance ORDER BY importance DESC"):
            print(f"  {r['importance']}: {'#' * min(r['c'], 60)} {r['c']}")


def cmd_tags(args):
    conn = open_db(args.db)
    for r in conn.execute(
        "SELECT tag, kind, COUNT(*) c FROM tags GROUP BY tag, kind ORDER BY c DESC LIMIT ?",
        (args.limit,),
    ):
        print(f"{r['c']:>5}  {r['kind']:<9} {r['tag']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="kb.sqlite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("import"); p.add_argument("files", nargs="+"); p.set_defaults(func=cmd_import)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--tag", action="append"); p.add_argument("--flag", action="append")
    p.add_argument("--source"); p.add_argument("--min-importance", type=int)
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("filter")
    p.add_argument("--tag", action="append"); p.add_argument("--flag", action="append")
    p.add_argument("--source"); p.add_argument("--min-importance", type=int)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("show"); p.add_argument("id"); p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("reset", help="clear classification so conversations get re-classified")
    p.add_argument("ids", nargs="*", help="conversation id prefixes")
    p.add_argument("--mock", action="store_true", help="reset all mock-classified conversations")
    p.add_argument("--all", action="store_true", help="reset every classification")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("stats"); p.set_defaults(func=cmd_stats)
    p = sub.add_parser("tags"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_tags)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)  # output piped to head/less and closed early — not an error
