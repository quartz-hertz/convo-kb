#!/usr/bin/env python3
"""
import_chatgpt.py — normalize a ChatGPT export (conversations.json) into the
knowledge base's common conversation schema.

Usage:
  python3 import_chatgpt.py conversations.json out.json
  python3 import_chatgpt.py conversations.json out.jsonl --jsonl
  python3 import_chatgpt.py conversations.json out.json --include-hidden --include-tool

Branch selection: walks parent links backward from `current_node` (the branch
the user actually kept after edits/regenerations). If `current_node` is missing
or broken, falls back to the deepest leaf path. Total branch count is recorded
in `branch_count` so regeneration-heavy conversations are identifiable.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

CHATGPT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "kb://chatgpt")


def iso(epoch):
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def render_content(content):
    """Flatten the various content_type payloads into plain text."""
    if not isinstance(content, dict):
        return ""
    ctype = content.get("content_type", "text")
    parts = content.get("parts") or []

    if ctype in ("text", "multimodal_text"):
        out = []
        for p in parts:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict):
                pt = p.get("content_type", "")
                if pt == "image_asset_pointer":
                    out.append(f"[image: {p.get('asset_pointer', 'attached')}]")
                elif pt == "audio_transcription":
                    out.append(p.get("text", ""))
                elif "text" in p:
                    out.append(str(p["text"]))
        return "\n".join(x for x in out if x)

    if ctype == "code":
        lang = content.get("language") or ""
        text = content.get("text", "")
        return f"```{lang}\n{text}\n```" if text else ""

    if ctype == "execution_output":
        text = content.get("text", "")
        return f"[execution output]\n{text}" if text else ""

    if ctype == "tether_quote":
        return f"[quoted: {content.get('title', '')}] {content.get('text', '')}".strip()

    if ctype == "tether_browsing_display":
        return content.get("result", "") or ""

    if ctype == "user_editable_context":
        return ""  # custom-instructions blob; system-level, skip content

    # Unknown type: salvage any string parts or text field
    text = content.get("text", "")
    if text:
        return f"[{ctype}] {text}"
    return "\n".join(p for p in parts if isinstance(p, str))


def current_path(mapping, current_node):
    """Walk parent links from current_node to root; return node list root->leaf."""
    path, seen = [], set()
    node_id = current_node
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        path.append(mapping[node_id])
        node_id = mapping[node_id].get("parent")
    return list(reversed(path))


def deepest_leaf_path(mapping):
    """Fallback: longest root->leaf path by walking children."""
    children_of = {nid: (n.get("children") or []) for nid, n in mapping.items()}
    parents = {c for kids in children_of.values() for c in kids}
    roots = [nid for nid in mapping if nid not in parents]
    best = []

    def dfs(nid, acc):
        nonlocal best
        acc = acc + [mapping[nid]]
        kids = [k for k in children_of.get(nid, []) if k in mapping]
        if not kids and len(acc) > len(best):
            best = acc
        for k in kids:
            dfs(k, acc)

    for r in roots:
        dfs(r, [])
    return best


def count_branches(mapping):
    return sum(1 for n in mapping.values() if len(n.get("children") or []) > 1)


def extract_messages(path_nodes, include_hidden, include_tool):
    messages = []
    for node in path_nodes:
        msg = node.get("message")
        if not msg:
            continue
        meta = msg.get("metadata") or {}
        author = msg.get("author") or {}
        role = author.get("role", "unknown")

        if meta.get("is_visually_hidden_from_conversation") and not include_hidden:
            continue
        if role == "system" and not include_hidden:
            continue
        if role == "tool" and not include_tool:
            continue

        text = render_content(msg.get("content") or {}).strip()
        if not text:
            continue

        if role == "tool":
            tool_name = author.get("name") or "tool"
            text = f"[{tool_name}] {text}"

        messages.append({
            "role": role,
            "content": text,
            "timestamp": iso(msg.get("create_time")),
        })
    return messages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("conversations_json", nargs="+",
                    help="one or more export files, e.g. conversations.json or conversations-*.json")
    ap.add_argument("out")
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--include-hidden", action="store_true", help="keep system/hidden messages")
    ap.add_argument("--include-tool", action="store_true", help="keep tool-role messages (browsing, code interpreter)")
    args = ap.parse_args()

    convs = []  # (conversation, source_path)
    for path in args.conversations_json:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):  # some exports wrap the array
            raw = raw.get("conversations", [raw])
        convs.extend((c, path) for c in raw)

    records, skipped, fallbacks, dupes = [], 0, 0, 0
    seen_ids = set()
    for conv, src_path in convs:
        pid = conv.get("conversation_id") or conv.get("id")
        if pid and pid in seen_ids:
            dupes += 1
            continue
        if pid:
            seen_ids.add(pid)
        mapping = conv.get("mapping") or {}
        path = current_path(mapping, conv.get("current_node"))
        if not path and mapping:
            path = deepest_leaf_path(mapping)
            fallbacks += 1
        messages = extract_messages(path, args.include_hidden, args.include_tool)
        if not messages:
            skipped += 1
            continue

        platform_id = pid or ""
        records.append({
            "id": str(uuid.uuid5(CHATGPT_NAMESPACE, f"chatgpt:{platform_id}")),
            "source": "chatgpt",
            "platform_id": platform_id,
            "title": conv.get("title"),
            "created_at": iso(conv.get("create_time")),
            "updated_at": iso(conv.get("update_time")),
            "model": conv.get("default_model_slug"),
            "branch_count": count_branches(mapping),
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
          f"({skipped} empty skipped, {fallbacks} current_node fallbacks, {dupes} duplicates dropped)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
