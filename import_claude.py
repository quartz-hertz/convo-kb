#!/usr/bin/env python3
"""
import_claude.py — normalize a Claude export (conversations.json) into the
knowledge base's common conversation schema.

Usage:
  python3 import_claude.py conversations.json out.json
  python3 import_claude.py conversations.json out.jsonl --jsonl
  python3 import_claude.py conversations.json out.json --include-attachments --include-tool

Notes:
  - Claude timestamps are already ISO 8601 strings; passed through as-is.
  - The export has no per-conversation model field; "model" is null.
  - Message text is taken from the structured `content` block array when
    present (text blocks joined, tool activity annotated), falling back to
    the flat `text` field otherwise.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

CLAUDE_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "kb://claude")

ROLE_MAP = {"human": "user", "assistant": "assistant", "system": "system"}


def render_blocks(blocks, include_tool, max_tool_chars):
    """Flatten a Claude `content` block array to text."""
    out = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type", "")
        if btype == "text":
            t = b.get("text", "")
            if t:
                out.append(t)
        elif btype == "tool_use":
            name = b.get("name", "tool")
            if include_tool:
                args = json.dumps(b.get("input", {}), ensure_ascii=False)
                if len(args) > max_tool_chars:
                    args = args[:max_tool_chars] + "..."
                out.append(f"[tool call: {name}({args})]")
            else:
                out.append(f"[tool call: {name}]")
        elif btype == "tool_result":
            if include_tool:
                content = b.get("content")
                if isinstance(content, list):
                    s = " ".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                else:
                    s = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                s = (s or "").strip()
                if s:
                    if len(s) > max_tool_chars:
                        s = s[:max_tool_chars] + "..."
                    out.append(f"[tool result] {s}")
        elif btype == "thinking":
            continue  # model scratchpad; skip for KB purposes
        else:
            t = b.get("text", "")
            if t:
                out.append(f"[{btype}] {t}")
    return "\n".join(out)


def render_message(msg, include_attachments, include_tool, max_attach_chars, max_tool_chars):
    blocks = msg.get("content")
    if isinstance(blocks, list) and blocks:
        text = render_blocks(blocks, include_tool, max_tool_chars)
    else:
        text = msg.get("text", "") or ""

    extras = []
    for att in msg.get("attachments") or []:
        if not isinstance(att, dict):
            continue
        name = att.get("file_name", "attachment")
        extracted = (att.get("extracted_content") or "").strip()
        if include_attachments and extracted:
            if len(extracted) > max_attach_chars:
                extracted = extracted[:max_attach_chars] + "..."
            extras.append(f"[attachment: {name}]\n{extracted}")
        else:
            extras.append(f"[attachment: {name}]")
    for f in msg.get("files") or []:
        if isinstance(f, dict) and f.get("file_name"):
            extras.append(f"[file: {f['file_name']}]")

    return "\n".join(x for x in [text.strip()] + extras if x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("conversations_json", nargs="+",
                    help="conversations.json from the Claude export ZIP (one or more)")
    ap.add_argument("out")
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--include-attachments", action="store_true",
                    help="include (truncated) extracted text of pasted attachments")
    ap.add_argument("--include-tool", action="store_true",
                    help="include tool call arguments and tool results")
    ap.add_argument("--max-attachment-chars", type=int, default=2000)
    ap.add_argument("--max-tool-chars", type=int, default=500)
    args = ap.parse_args()

    convs = []
    for path in args.conversations_json:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("conversations", [raw])
        convs.extend((c, path) for c in raw)

    records, skipped, dupes = [], 0, 0
    seen = set()
    for conv, src_path in convs:
        pid = conv.get("uuid") or conv.get("id") or ""
        if pid and pid in seen:
            dupes += 1
            continue
        if pid:
            seen.add(pid)

        messages = []
        for msg in conv.get("chat_messages") or []:
            role = ROLE_MAP.get(msg.get("sender", ""), msg.get("sender") or "unknown")
            text = render_message(
                msg, args.include_attachments, args.include_tool,
                args.max_attachment_chars, args.max_tool_chars,
            )
            if not text.strip():
                continue
            messages.append({
                "role": role,
                "content": text,
                "timestamp": msg.get("created_at"),
            })
        if not messages:
            skipped += 1
            continue

        records.append({
            "id": str(uuid.uuid5(CLAUDE_NAMESPACE, f"claude:{pid}")),
            "source": "claude",
            "platform_id": pid,
            "title": conv.get("name") or None,
            "created_at": conv.get("created_at"),
            "updated_at": conv.get("updated_at"),
            "model": conv.get("model"),  # not present in exports today; kept for future-proofing
            "message_count": len(messages),
            "messages": messages,
            "raw_path": str(Path(src_path)),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        if args.jsonl:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"normalized {len(records)} conversations from {len(args.conversations_json)} file(s) -> {args.out} "
          f"({skipped} empty skipped, {dupes} duplicates dropped)", file=sys.stderr)


if __name__ == "__main__":
    main()
