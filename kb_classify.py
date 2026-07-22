#!/usr/bin/env python3
"""
kb_classify.py — classify conversations in the KB with an LLM.

Backends:
  anthropic  Anthropic API (reads ANTHROPIC_API_KEY env var)
  openai     any OpenAI-compatible server, e.g. Osaurus or LM Studio
  mock       deterministic heuristics, no network — for testing the pipeline

Usage:
  # fast pass with a local model via Osaurus:
  python3 kb_classify.py --backend openai --base-url http://127.0.0.1:1337/v1 --model your-model

  # fast pass with Anthropic Haiku:
  python3 kb_classify.py --backend anthropic --model claude-haiku-4-5

  # slow pass: re-do only the important stuff with a stronger model:
  python3 kb_classify.py --backend anthropic --model claude-sonnet-4-6 --reclassify-min-importance 4

  # dry-run the pipeline end to end:
  python3 kb_classify.py --backend mock --limit 5

Only unclassified conversations are processed (resume-safe). Every raw model
response is appended to the JSONL log before the DB is touched, so results
can be re-ingested without re-inference via --from-log.
"""

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

SEED_TOPICS = ["swift", "python", "sysadmin", "networking", "architecture", "security",
               "devops", "ai-ml", "writing", "research", "productivity"]
SEED_TAGS = ["has_code", "debugging", "architecture_decision", "how-to", "brainstorming",
             "reference", "tutorial", "review"]
FLAG_COLS = ["has_attachments", "has_code", "has_images", "is_sysadmin",
             "is_research", "is_creative", "concluded"]

SYSTEM_PROMPT = f"""You are a precise conversation classifier for a personal knowledge base.
You will be given one conversation between a user and an AI assistant, possibly truncated.
Respond with ONLY a JSON object — no markdown fences, no preamble — with exactly these keys:

{{
  "summary": "1-2 sentences: what the conversation accomplished or covered",
  "importance": 1,
  "topics": ["..."],
  "tags": ["..."],
  "flags": {{"has_attachments": false, "has_code": false, "has_images": false,
            "is_sysadmin": false, "is_research": false, "is_creative": false,
            "concluded": true}},
  "languages": ["..."],
  "proposed_tags": ["..."]
}}

Rules:
- importance: 1 (trivial/small-talk) to 5 (high-value durable knowledge, decisions, working solutions).
- topics: subject matter. Prefer this seed list, extend only when necessary: {", ".join(SEED_TOPICS)}
- tags: structural labels from this seed list: {", ".join(SEED_TAGS)}
- languages: programming languages present, lowercase, empty list if none.
- proposed_tags: new labels NOT in the seed lists that would be useful; empty list if none.
- concluded: true if the conversation reached a resolution rather than trailing off.
"""

HEAD_CHARS = 8000   # ~2000 tokens
TAIL_CHARS = 2000   # ~500 tokens


def truncate_convo(text):
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return (text[:HEAD_CHARS] + "\n\n[... middle truncated ...]\n\n" + text[-TAIL_CHARS:])


def http_json(url, payload, headers, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_anthropic(model, prompt, api_key):
    data = http_json(
        "https://api.anthropic.com/v1/messages",
        {"model": model, "max_tokens": 1000, "system": SYSTEM_PROMPT,
         "messages": [{"role": "user", "content": prompt}]},
        {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
    )
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def call_openai(base_url, model, prompt, api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = http_json(
        base_url.rstrip("/") + "/chat/completions",
        {"model": model, "max_tokens": 1000, "temperature": 0,
         "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}]},
        headers,
    )
    return data["choices"][0]["message"]["content"]


def call_mock(convo_text, title):
    """No-network stand-in: crude heuristics, valid schema."""
    has_code = "```" in convo_text
    words = len(convo_text.split())
    return json.dumps({
        "summary": f"(mock) Conversation titled '{title or 'untitled'}' with ~{words} words.",
        "importance": 3 if has_code else (2 if words > 300 else 1),
        "topics": ["ai-ml"] if "model" in convo_text.lower() else [],
        "tags": (["has_code"] if has_code else []) + (["how-to"] if "how" in convo_text.lower() else []),
        "flags": {"has_attachments": "[attachment:" in convo_text, "has_code": has_code,
                  "has_images": "[image" in convo_text, "is_sysadmin": False,
                  "is_research": False, "is_creative": False, "concluded": True},
        "languages": ["swift"] if "swift" in convo_text.lower() else [],
        "proposed_tags": [],
    })


def parse_classification(raw):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    obj = json.loads(s[start:end + 1])
    obj["summary"] = str(obj.get("summary", ""))[:1000]
    obj["importance"] = max(1, min(5, int(obj.get("importance", 1))))
    for key in ("topics", "tags", "languages", "proposed_tags"):
        obj[key] = [str(t).strip().lower() for t in obj.get(key) or [] if str(t).strip()]
    flags = obj.get("flags") or {}
    obj["flags"] = {c: bool(flags.get(c, False)) for c in FLAG_COLS}
    return obj


def write_classification(conn, cid, cls):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE conversations SET summary=?, importance=?, classified_at=? WHERE id=?",
        (cls["summary"], cls["importance"], now, cid),
    )
    conn.execute("DELETE FROM tags WHERE conversation_id=?", (cid,))
    rows = ([(cid, t, "topic") for t in cls["topics"]]
            + [(cid, t, "tag") for t in cls["tags"]]
            + [(cid, t, "language") for t in cls["languages"]]
            + [(cid, t, "proposed") for t in cls["proposed_tags"]])
    conn.executemany("INSERT OR IGNORE INTO tags (conversation_id, tag, kind) VALUES (?,?,?)", rows)
    f = cls["flags"]
    conn.execute(
        f"""INSERT INTO flags (conversation_id, {', '.join(FLAG_COLS)})
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(conversation_id) DO UPDATE SET
            {', '.join(f'{c}=excluded.{c}' for c in FLAG_COLS)}""",
        (cid, *[int(f[c]) for c in FLAG_COLS]),
    )
    # refresh summary in the FTS row
    row = conn.execute(
        "SELECT title, full_text FROM conversations_fts WHERE id=?", (cid,)
    ).fetchone()
    conn.execute("DELETE FROM conversations_fts WHERE id=?", (cid,))
    conn.execute(
        "INSERT INTO conversations_fts (id, title, summary, full_text) VALUES (?,?,?,?)",
        (cid, row["title"] if row else "", cls["summary"], row["full_text"] if row else ""),
    )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="kb.sqlite")
    ap.add_argument("--backend", choices=["anthropic", "openai", "mock"], default="mock")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1",
                    help="openai backend only (Osaurus/LM Studio endpoint)")
    ap.add_argument("--api-key", default=None,
                    help="explicit API key (prefer env vars — CLI args leak into shell history)")
    ap.add_argument("--api-key-env", default=None,
                    help="name of env var holding the key (defaults: ANTHROPIC_API_KEY / OPENAI_API_KEY)")
    ap.add_argument("--limit", type=int, default=0, help="max conversations this run (0 = all pending)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--log", default="classify_log.jsonl")
    ap.add_argument("--reclassify-min-importance", type=int, default=None,
                    help="slow pass: re-classify already-classified conversations with importance >= N")
    ap.add_argument("--from-log", default=None,
                    help="skip inference; re-ingest classifications from a previous JSONL log")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    if args.from_log:
        n = 0
        for line in open(args.from_log, encoding="utf-8"):
            entry = json.loads(line)
            if entry.get("ok"):
                write_classification(conn, entry["id"], parse_classification(entry["raw"]))
                n += 1
        print(f"re-ingested {n} classifications from {args.from_log}")
        return

    import os
    default_env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(args.backend)
    env_name = args.api_key_env or default_env
    api_key = args.api_key or (os.environ.get(env_name) if env_name else None)
    if args.backend == "anthropic" and not api_key:
        sys.exit(f"anthropic backend needs a key: set {env_name} or pass --api-key")
    # openai backend: key is optional — local servers (Osaurus, LM Studio)
    # usually accept anonymous requests; a Bearer header is sent only if a key
    # was found, which also covers hosted OpenAI-compatible providers.

    if args.reclassify_min_importance is not None:
        where = "classified_at IS NOT NULL AND importance >= ?"
        params = (args.reclassify_min_importance,)
    else:
        where = "classified_at IS NULL"
        params = ()
    q = f"SELECT id, title FROM conversations WHERE {where} ORDER BY created_at"
    pending = conn.execute(q, params).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(pending)} conversations to classify", file=sys.stderr)

    ok = fail = 0
    with open(args.log, "a", encoding="utf-8") as log:
        for i, row in enumerate(pending, 1):
            cid, title = row["id"], row["title"]
            msgs = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY seq", (cid,)
            ).fetchall()
            convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            prompt = f"Title: {title or '(untitled)'}\n\n{truncate_convo(convo_text)}"

            try:
                if args.backend == "anthropic":
                    raw = call_anthropic(args.model, prompt, api_key)
                elif args.backend == "openai":
                    raw = call_openai(args.base_url, args.model, prompt, api_key)
                else:
                    raw = call_mock(convo_text, title)
                cls = parse_classification(raw)
                log.write(json.dumps({"id": cid, "ok": True, "backend": args.backend,
                                      "model": args.model, "raw": raw},
                                     ensure_ascii=False) + "\n")
                log.flush()
                write_classification(conn, cid, cls)
                ok += 1
                print(f"[{i}/{len(pending)}] {cid[:8]} imp={cls['importance']} {title or ''}"[:100],
                      file=sys.stderr)
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
                    KeyError, json.JSONDecodeError) as e:
                fail += 1
                log.write(json.dumps({"id": cid, "ok": False, "error": str(e)}) + "\n")
                log.flush()
                print(f"[{i}/{len(pending)}] {cid[:8]} FAILED: {e}", file=sys.stderr)
            if args.sleep:
                time.sleep(args.sleep)

    print(f"done: {ok} classified, {fail} failed (failures remain pending; re-run to retry)")


if __name__ == "__main__":
    main()
