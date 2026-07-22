#!/usr/bin/env python3
"""
kb_web.py — local web UI for the conversation knowledge base.

Pure stdlib, single file, read-only. Serves search / browse / read over
kb.sqlite using the same query patterns as kb.py.

Usage:
  python3 kb_web.py                     # http://127.0.0.1:8765, ./kb.sqlite
  python3 kb_web.py --db kb.sqlite --port 8765 --host 127.0.0.1

Routes:
  /            browse (timeline) + search + filters
  /c/<prefix>  conversation view (id or unique prefix, like `kb.py show`)
"""

import argparse
import html
import re
import sqlite3
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlencode, urlparse

PAGE_SIZE = 50
SNIPPET_TOKENS = 24

FLAG_COLS = ("has_attachments", "has_code", "has_images",
             "is_sysadmin", "is_research", "is_creative", "concluded")
KIND_ORDER = ("topic", "tag", "language", "proposed")
KIND_LABELS = {"topic": "Topics", "tag": "Tags",
               "language": "Languages", "proposed": "Proposed"}
SIDEBAR_TAG_CAP = 12          # per kind; the rest fold behind "show all"

SORTS = {
    "newest":     "COALESCE(c.created_at,'') DESC",
    "oldest":     "COALESCE(c.created_at,'') ASC",
    "importance": "COALESCE(c.importance,0) DESC, COALESCE(c.updated_at,'') DESC",
}

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")

DB_PATH = "kb.sqlite"          # set from --db in main()


# ---------------------------------------------------------------- database

def open_db():
    """Read-only connection per request (handler threads must not share one)."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def fts_where_and_params(q):
    """FTS5 MATCH clause; caller falls back to quoted form on syntax errors."""
    return "conversations_fts MATCH ?", q


def quoted_fts_query(q):
    """Literal-phrase fallback for queries FTS5 rejects (stray quotes, AND/OR typos)."""
    return '"' + q.replace('"', '""') + '"'


def apply_filters(params, where, sql_params):
    """Mirror of kb.py's apply_filters, driven by parsed query-string params."""
    if params.get("source"):
        where.append("c.source = ?")
        sql_params.append(params["source"])
    if params.get("min_imp"):
        where.append("c.importance >= ?")
        sql_params.append(params["min_imp"])
    for tag in params.get("tags", []):
        where.append("c.id IN (SELECT conversation_id FROM tags WHERE tag = ?)")
        sql_params.append(tag)
    for flag in params.get("flags", []):
        if flag in FLAG_COLS:   # whitelist — never interpolate user input as SQL
            where.append(
                f"c.id IN (SELECT conversation_id FROM flags WHERE {flag} = 1)")
    return where, sql_params


def run_search(conn, params, offset):
    """Search: importance DESC then BM25, exactly like `kb.py search` — plus a
    highlighted snippet. Falls back to a quoted phrase if FTS5 rejects the query."""
    q = params["q"]
    for attempt in (q, quoted_fts_query(q)):
        where, sqlp = apply_filters(params, ["conversations_fts MATCH ?"], [attempt])
        sql = f"""SELECT c.*, snippet(conversations_fts, 3, char(2), char(3), ' … ', {SNIPPET_TOKENS}) AS snip
                  FROM conversations_fts
                  JOIN conversations c ON c.id = conversations_fts.id
                  WHERE {' AND '.join(where)}
                  ORDER BY COALESCE(c.importance, 0) DESC, bm25(conversations_fts)
                  LIMIT ? OFFSET ?"""
        count_sql = f"""SELECT COUNT(*) n FROM conversations_fts
                        JOIN conversations c ON c.id = conversations_fts.id
                        WHERE {' AND '.join(where)}"""
        try:
            total = conn.execute(count_sql, sqlp).fetchone()["n"]
            rows = conn.execute(sql, sqlp + [PAGE_SIZE, offset]).fetchall()
            return rows, total
        except sqlite3.OperationalError:
            continue
    return [], 0


def run_browse(conn, params, offset):
    """Browse/timeline: `kb.py filter` semantics with selectable sort."""
    where, sqlp = apply_filters(params, ["1=1"], [])
    order = SORTS.get(params["sort"], SORTS["newest"])
    total = conn.execute(
        f"SELECT COUNT(*) n FROM conversations c WHERE {' AND '.join(where)}",
        sqlp).fetchone()["n"]
    rows = conn.execute(
        f"""SELECT c.*, NULL AS snip FROM conversations c
            WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ? OFFSET ?""",
        sqlp + [PAGE_SIZE, offset]).fetchall()
    return rows, total


def sidebar_data(conn):
    sources = conn.execute(
        "SELECT source, COUNT(*) c FROM conversations GROUP BY source ORDER BY c DESC"
    ).fetchall()
    tags_by_kind = {k: [] for k in KIND_ORDER}
    for r in conn.execute(
            "SELECT tag, kind, COUNT(*) c FROM tags GROUP BY tag, kind ORDER BY c DESC, tag"):
        if r["kind"] in tags_by_kind:
            tags_by_kind[r["kind"]].append((r["tag"], r["c"]))
    flag_counts = {}
    row = conn.execute(
        "SELECT " + ", ".join(f"SUM({f}) AS {f}" for f in FLAG_COLS) + " FROM flags"
    ).fetchone()
    if row is not None:
        flag_counts = {f: (row[f] or 0) for f in FLAG_COLS}
    stats = conn.execute(
        """SELECT COUNT(*) total,
                  SUM(CASE WHEN classified_at IS NOT NULL THEN 1 ELSE 0 END) classified
           FROM conversations""").fetchone()
    return sources, tags_by_kind, flag_counts, stats


def fetch_conversation(conn, prefix):
    """Returns (row, msgs, tags) | ('ambiguous', rows) | (None, ...)."""
    rows = conn.execute(
        "SELECT * FROM conversations WHERE id LIKE ? LIMIT 25", (prefix + "%",)
    ).fetchall()
    if not rows:
        return None, None, None
    if len(rows) > 1:
        return "ambiguous", rows, None
    c = rows[0]
    msgs = conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE conversation_id=? ORDER BY seq",
        (c["id"],)).fetchall()
    tags = conn.execute(
        "SELECT tag, kind FROM tags WHERE conversation_id=? ORDER BY kind, tag",
        (c["id"],)).fetchall()
    return c, msgs, tags


# ------------------------------------------------------- markdown-lite HTML

RE_FENCE = re.compile(r"```[ \t]*(\w[\w+#.-]*)?[ \t]*\n(.*?)(?:\n```[ \t]*$|\Z)",
                      re.S | re.M)
RE_CODE = re.compile(r"`([^`\n]+)`")
RE_BOLD = re.compile(r"\*\*([^*\n]+?)\*\*|__([^_\n]+?)__")
RE_ITAL = re.compile(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])")
RE_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
RE_ANNOT = re.compile(r"^\[(tool call|tool result|attachment|image|thinking)\b[^\]]*\]$")
RE_OL = re.compile(r"^\s{0,3}\d{1,3}[.)]\s+(.*)$")
RE_UL = re.compile(r"^\s{0,3}[-*+]\s+(.*)$")
RE_H = re.compile(r"^(#{1,4})\s+(.*)$")

CODE_PH = "\x00CODE%d\x00"


def render_inline(escaped):
    """Inline markdown on already-HTML-escaped text."""
    stash = []

    def keep(m):
        stash.append(m.group(1))
        return CODE_PH % (len(stash) - 1)

    s = RE_CODE.sub(keep, escaped)
    s = RE_LINK.sub(
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', s)
    s = RE_BOLD.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", s)
    s = RE_ITAL.sub(r"<em>\1</em>", s)
    for i, code in enumerate(stash):
        s = s.replace(CODE_PH % i, f"<code>{code}</code>")
    return s


def render_prose(text):
    """Escaped line-oriented rendering: headings, lists, quotes, paragraphs."""
    out, para, lst = [], [], None   # lst: (tag, [items]) while inside a list

    def flush_para():
        if para:
            out.append("<p>" + render_inline("<br>".join(para)) + "</p>")
            para.clear()

    def flush_list():
        nonlocal lst
        if lst:
            tag, items = lst
            out.append(f"<{tag}>" +
                       "".join(f"<li>{render_inline(i)}</li>" for i in items) +
                       f"</{tag}>")
            lst = None

    for raw in text.split("\n"):
        line = html.escape(raw.rstrip())
        stripped = line.strip()
        if not stripped:
            flush_para(); flush_list(); continue
        if RE_ANNOT.match(raw.strip()):
            flush_para(); flush_list()
            out.append(f'<span class="annot">{html.escape(raw.strip())}</span>')
            continue
        m = RE_H.match(line)
        if m:
            flush_para(); flush_list()
            n = min(len(m.group(1)) + 2, 5)     # msg headings start at h3
            out.append(f"<h{n}>{render_inline(m.group(2))}</h{n}>")
            continue
        if stripped in ("---", "***", "___"):
            flush_para(); flush_list(); out.append("<hr>"); continue
        m = RE_UL.match(line) or RE_OL.match(line)
        if m:
            flush_para()
            tag = "ul" if RE_UL.match(line) else "ol"
            if not lst or lst[0] != tag:
                flush_list(); lst = (tag, [])
            lst[1].append(m.group(1))
            continue
        if stripped.startswith("&gt;"):
            flush_para(); flush_list()
            out.append("<blockquote>" +
                       render_inline(stripped[4:].lstrip()) + "</blockquote>")
            continue
        flush_list()
        para.append(line)
    flush_para(); flush_list()
    return "".join(out)


def render_message_html(content):
    """Full message body: fenced code blocks + prose between them."""
    parts, pos = [], 0
    for m in RE_FENCE.finditer(content):
        if m.start() > pos:
            parts.append(render_prose(content[pos:m.start()]))
        lang = m.group(1) or ""
        label = f'<span class="lang">{html.escape(lang)}</span>' if lang else ""
        parts.append(f'<div class="codeblock">{label}<pre><code>'
                     f"{html.escape(m.group(2))}</code></pre></div>")
        pos = m.end()
    if pos < len(content):
        parts.append(render_prose(content[pos:]))
    return "".join(parts)


# ------------------------------------------------------------ URL helpers

def parse_params(qs):
    d = parse_qs(qs, keep_blank_values=False)
    p = {
        "q": d.get("q", [""])[0].strip(),
        "tags": [t for t in d.get("tag", []) if t],
        "flags": [f for f in d.get("flag", []) if f in FLAG_COLS],
        "source": d.get("source", [""])[0],
        "sort": d.get("sort", ["newest"])[0],
        "min_imp": 0,
        "page": 1,
    }
    try:
        p["min_imp"] = max(0, min(5, int(d.get("min_imp", ["0"])[0])))
    except ValueError:
        pass
    try:
        p["page"] = max(1, int(d.get("page", ["1"])[0]))
    except ValueError:
        pass
    if p["sort"] not in SORTS:
        p["sort"] = "newest"
    return p


def make_url(p, **overrides):
    """Rebuild / with current filters, applying overrides. page resets unless kept."""
    merged = dict(p)
    merged.update(overrides)
    pairs = []
    if merged.get("q"):
        pairs.append(("q", merged["q"]))
    for t in merged.get("tags", []):
        pairs.append(("tag", t))
    for f in merged.get("flags", []):
        pairs.append(("flag", f))
    if merged.get("source"):
        pairs.append(("source", merged["source"]))
    if merged.get("min_imp"):
        pairs.append(("min_imp", str(merged["min_imp"])))
    if merged.get("sort") and merged["sort"] != "newest":
        pairs.append(("sort", merged["sort"]))
    if merged.get("page", 1) > 1:
        pairs.append(("page", str(merged["page"])))
    return "/?" + urlencode(pairs) if pairs else "/"


def toggle(seq, item):
    return [x for x in seq if x != item] if item in seq else list(seq) + [item]


# ------------------------------------------------------------- formatting

def fmt_date(iso, with_time=False):
    if not iso:
        return "undated"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M") if with_time else dt.strftime("%Y-%m-%d")
    except ValueError:
        return iso[:16 if with_time else 10]


def month_label(iso):
    if not iso or len(iso) < 7:
        return "Undated"
    try:
        y, m = int(iso[:4]), int(iso[5:7])
        return f"{MONTHS[m - 1]} {y}"
    except (ValueError, IndexError):
        return "Undated"


def importance_dots(imp):
    if imp is None:
        return '<span class="imp unclassified" title="not yet classified">—</span>'
    filled, empty = "●" * imp, "○" * (5 - imp)
    return (f'<span class="imp" title="importance {imp}/5">'
            f'<b>{filled}</b>{empty}</span>')


def esc(s):
    return html.escape(s or "")


# ------------------------------------------------------------------- CSS

STYLE = """
:root{
  --bg:#f5f6f7; --panel:#fdfdfc; --ink:#1b1f24; --muted:#5c6670;
  --line:#e2e5e8; --accent:#0e7490; --accent-ink:#0b5c73;
  --accent-soft:#e3f0f3; --amber:#b45309; --code-bg:#eef0f2;
  --user-rail:#0e7490; --assist-rail:#c4cbd1; --tool-rail:#b45309;
  --mark:#fde68a; --radius:6px;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#15181c; --panel:#1c2025; --ink:#e6e9ec; --muted:#9aa4ad;
    --line:#2a2f36; --accent:#4cb8d4; --accent-ink:#7fd0e6;
    --accent-soft:#123742; --amber:#e8a75a; --code-bg:#14171b;
    --user-rail:#4cb8d4; --assist-rail:#3a424b; --tool-rail:#e8a75a;
    --mark:#6b5d1f;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent-ink);text-decoration:none}
a:hover{text-decoration:underline}
a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible{
  outline:2px solid var(--accent);outline-offset:2px}
mark{background:var(--mark);color:inherit;border-radius:2px;padding:0 1px}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}

.layout{display:grid;grid-template-columns:250px minmax(0,1fr);min-height:100vh}
@media (max-width:820px){.layout{grid-template-columns:1fr}
  .sidebar{position:static;height:auto;border-right:none;border-bottom:1px solid var(--line)}}

/* -------- sidebar -------- */
.sidebar{border-right:1px solid var(--line);padding:18px 16px 32px;
  position:sticky;top:0;height:100vh;overflow-y:auto;background:var(--panel)}
.wordmark{font-family:ui-monospace,Menlo,monospace;font-size:20px;font-weight:700;
  letter-spacing:-.5px;margin:0 0 2px}
.wordmark a{color:var(--ink)}
.tagline{color:var(--muted);font-size:12px;margin:0 0 18px}
.side-h{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);margin:18px 0 6px}
.side-list{list-style:none;margin:0;padding:0;font-size:13.5px}
.side-list li{margin:1px 0}
.side-list a{display:flex;justify-content:space-between;gap:8px;padding:2px 6px;
  border-radius:4px;color:var(--ink)}
.side-list a:hover{background:var(--accent-soft);text-decoration:none}
.side-list a.on{background:var(--accent-soft);color:var(--accent-ink);font-weight:600}
.side-list .n{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
details.more{margin-top:2px}
details.more summary{cursor:pointer;color:var(--muted);font-size:12.5px;padding:2px 6px}
.side-stats{margin-top:26px;padding-top:12px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12px}

/* -------- main / toolbar -------- */
.main{padding:0 clamp(16px,4vw,44px) 60px;max-width:980px}
.toolbar{position:sticky;top:0;z-index:5;background:var(--bg);
  padding:16px 0 10px;border-bottom:1px solid var(--line);margin-bottom:6px}
.searchrow{display:flex;gap:8px}
.searchrow input[type=search]{flex:1;padding:8px 12px;font-size:15px;color:var(--ink);
  background:var(--panel);border:1px solid var(--line);border-radius:var(--radius)}
.searchrow button{padding:8px 16px;border:1px solid var(--accent);border-radius:var(--radius);
  background:var(--accent);color:#fff;font-size:14px;cursor:pointer}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:10px;
  font-size:13px;color:var(--muted)}
.controls select{padding:3px 6px;font-size:13px;background:var(--panel);
  color:var(--ink);border:1px solid var(--line);border-radius:4px}
.chip{display:inline-flex;align-items:center;gap:6px;background:var(--accent-soft);
  color:var(--accent-ink);border-radius:999px;padding:2px 10px;font-size:12.5px}
.chip a{color:inherit;font-weight:700;text-decoration:none}
.result-count{margin-left:auto;font-variant-numeric:tabular-nums}

/* -------- timeline list -------- */
.month{display:flex;align-items:center;gap:12px;margin:26px 0 4px}
.month h2{font-family:ui-monospace,Menlo,monospace;font-size:13px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0;white-space:nowrap}
.month::after{content:"";height:1px;background:var(--line);flex:1}
.card{display:block;position:relative;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--radius);padding:10px 14px 10px 18px;margin:8px 0;color:var(--ink)}
.card:hover{border-color:var(--accent);text-decoration:none}
.card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;
  border-radius:var(--radius) 0 0 var(--radius);background:var(--assist-rail)}
.card.src-chatgpt::before{background:#6b8f71}
.card.src-claude::before{background:#c0764a}
.card.src-osaurus::before{background:#7d6ba0}
.card .top{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px}
.card .title{font-weight:600;font-size:15px}
.card .meta{color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace;
  display:flex;gap:10px;flex-wrap:wrap;margin-left:auto}
.imp{font-size:11px;letter-spacing:1px;color:var(--muted);white-space:nowrap}
.imp b{color:var(--amber);font-weight:400}
.card .summary{color:var(--muted);font-size:13px;margin-top:3px;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.card .snippet{font-size:13px;margin-top:4px;color:var(--ink);
  border-left:2px solid var(--line);padding-left:8px}
.pager{display:flex;justify-content:space-between;margin-top:24px;font-size:14px}
.empty{color:var(--muted);margin:48px 0;text-align:center}
.empty b{display:block;font-size:16px;color:var(--ink);margin-bottom:6px}

/* -------- conversation view -------- */
.conv-head{padding:22px 0 14px;border-bottom:1px solid var(--line)}
.conv-head h1{font-size:21px;line-height:1.3;margin:2px 0 8px}
.crumb{font-size:13px}
.conv-meta{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:12.5px;
  font-family:ui-monospace,Menlo,monospace;margin-bottom:8px}
.conv-summary{font-size:14px;color:var(--muted);max-width:72ch;margin:6px 0 10px}
.tagrow{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.tagrow .kind{font-size:11px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.06em;align-self:center;margin-right:2px}
.tagrow a{font-size:12px;background:var(--accent-soft);color:var(--accent-ink);
  padding:1px 9px;border-radius:999px}
.tagrow a.proposed{background:transparent;border:1px dashed var(--line);color:var(--muted)}

.thread{max-width:76ch;margin-top:10px}
.msg{position:relative;padding:14px 0 14px 20px;border-left:3px solid var(--assist-rail)}
.msg.role-user{border-left-color:var(--user-rail)}
.msg.role-tool,.msg.role-system{border-left-color:var(--tool-rail)}
.msg + .msg{margin-top:2px}
.msg .who{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;
  display:flex;gap:10px;align-items:baseline}
.msg.role-user .who{color:var(--accent-ink)}
.msg .when{font-weight:400;letter-spacing:0;text-transform:none}
.msg-body{font-size:15px}
.msg-body p{margin:.55em 0;max-width:72ch}
.msg-body h3,.msg-body h4,.msg-body h5{margin:1em 0 .4em;line-height:1.3}
.msg-body ul,.msg-body ol{margin:.5em 0;padding-left:1.4em;max-width:70ch}
.msg-body li{margin:.2em 0}
.msg-body blockquote{margin:.6em 0;padding:2px 12px;border-left:3px solid var(--line);
  color:var(--muted)}
.msg-body hr{border:none;border-top:1px solid var(--line);margin:1em 0}
.msg-body code{background:var(--code-bg);border:1px solid var(--line);
  border-radius:4px;padding:.08em .35em;font-size:.88em}
.codeblock{position:relative;margin:.7em 0}
.codeblock .lang{position:absolute;top:6px;right:10px;font-size:10.5px;
  font-family:ui-monospace,Menlo,monospace;color:var(--muted);text-transform:lowercase}
.codeblock pre{background:var(--code-bg);border:1px solid var(--line);
  border-radius:var(--radius);padding:12px 14px;overflow-x:auto;margin:0;
  font-size:13px;line-height:1.5}
.codeblock pre code{background:none;border:none;padding:0;font-size:inherit}
.annot{display:inline-block;font-family:ui-monospace,Menlo,monospace;font-size:12px;
  color:var(--tool-rail);border:1px dashed var(--tool-rail);border-radius:4px;
  padding:1px 8px;margin:4px 0;opacity:.85}

.msg-body.clamp{max-height:34em;overflow:hidden;position:relative}
.msg-body.clamp::after{content:"";position:absolute;left:0;right:0;bottom:0;height:5em;
  background:linear-gradient(transparent,var(--bg))}
.expand{background:none;border:none;color:var(--accent-ink);cursor:pointer;
  font-size:13px;padding:4px 0;font-family:inherit}
.backtop{display:block;margin:36px 0 0;font-size:13px}
@media (prefers-reduced-motion: no-preference){
  .card{transition:border-color .12s ease}
}
"""

SCRIPT = """
document.addEventListener('click', function (e) {
  var b = e.target.closest('.expand'); if (!b) return;
  var body = b.previousElementSibling;
  var open = body.classList.toggle('clamp');
  b.textContent = open ? 'Show full message' : 'Collapse';
});
"""


# ---------------------------------------------------------------- pages

def page_shell(title, sidebar_html, main_html):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{STYLE}</style></head>
<body><div class="layout">
<nav class="sidebar">{sidebar_html}</nav>
<main class="main" id="top">{main_html}</main>
</div><script>{SCRIPT}</script></body></html>"""


def render_sidebar(conn, p):
    sources, tags_by_kind, flag_counts, stats = sidebar_data(conn)
    out = ['<div class="wordmark"><a href="/">kb</a></div>',
           '<p class="tagline">conversation knowledge base</p>']

    out.append('<div class="side-h">Sources</div><ul class="side-list">')
    all_on = "" if p["source"] else ' class="on"'
    out.append(f'<li><a{all_on} href="{esc(make_url(p, source="", page=1))}">'
               f'<span>all</span><span class="n">{stats["total"]}</span></a></li>')
    for s in sources:
        on = ' class="on"' if p["source"] == s["source"] else ""
        url = make_url(p, source="" if p["source"] == s["source"] else s["source"], page=1)
        out.append(f'<li><a{on} href="{esc(url)}"><span>{esc(s["source"])}</span>'
                   f'<span class="n">{s["c"]}</span></a></li>')
    out.append("</ul>")

    out.append('<div class="side-h">Importance</div><ul class="side-list">')
    for lvl in (0, 3, 4, 5):
        label = "any" if lvl == 0 else f"≥ {lvl}"
        on = ' class="on"' if p["min_imp"] == lvl else ""
        out.append(f'<li><a{on} href="{esc(make_url(p, min_imp=lvl, page=1))}">'
                   f'<span>{label}</span></a></li>')
    out.append("</ul>")

    for kind in KIND_ORDER:
        items = tags_by_kind.get(kind) or []
        if not items:
            continue
        out.append(f'<div class="side-h">{KIND_LABELS[kind]}</div><ul class="side-list">')

        def li(tag, n):
            on = ' class="on"' if tag in p["tags"] else ""
            url = make_url(p, tags=toggle(p["tags"], tag), page=1)
            return (f'<li><a{on} href="{esc(url)}"><span>{esc(tag)}</span>'
                    f'<span class="n">{n}</span></a></li>')

        head, rest = items[:SIDEBAR_TAG_CAP], items[SIDEBAR_TAG_CAP:]
        out.extend(li(t, n) for t, n in head)
        if rest:
            out.append(f'</ul><details class="more"><summary>show {len(rest)} more…'
                       f'</summary><ul class="side-list">')
            out.extend(li(t, n) for t, n in rest)
            out.append("</ul></details>")
        else:
            out.append("</ul>")

    if flag_counts:
        out.append('<div class="side-h">Flags</div><ul class="side-list">')
        for f in FLAG_COLS:
            n = flag_counts.get(f, 0)
            if not n:
                continue
            on = ' class="on"' if f in p["flags"] else ""
            url = make_url(p, flags=toggle(p["flags"], f), page=1)
            out.append(f'<li><a{on} href="{esc(url)}">'
                       f'<span>{esc(f.replace("_", " "))}</span><span class="n">{n}</span></a></li>')
        out.append("</ul>")

    out.append(f'<div class="side-stats">{stats["total"]} conversations<br>'
               f'{stats["classified"] or 0} classified</div>')
    return "".join(out)


def render_card(r, p):
    imp = importance_dots(r["importance"])
    title = esc(r["title"] or "(untitled)")
    date = fmt_date(r["created_at"])
    body = ""
    if r["snip"]:
        snip = esc(r["snip"]).replace("\x02", "<mark>").replace("\x03", "</mark>")
        body = f'<div class="snippet">{snip}</div>'
    elif r["summary"]:
        body = f'<div class="summary">{esc(r["summary"])}</div>'
    return f"""<a class="card src-{esc(r["source"])}" href="/c/{quote(r["id"])}">
<div class="top"><span class="title">{title}</span>
<span class="meta"><span>{esc(r["source"])}</span>{imp}
<span>{esc(date)}</span><span>{r["message_count"] or 0} msgs</span></span></div>
{body}</a>"""


def render_index(conn, p):
    offset = (p["page"] - 1) * PAGE_SIZE
    if p["q"]:
        rows, total = run_search(conn, p, offset)
    else:
        rows, total = run_browse(conn, p, offset)

    parts = ['<div class="toolbar"><form class="searchrow" action="/" method="get">']
    parts.append(f'<input type="search" name="q" value="{esc(p["q"])}" '
                 'placeholder="Search titles, summaries, full text…" autofocus>')
    for t in p["tags"]:
        parts.append(f'<input type="hidden" name="tag" value="{esc(t)}">')
    for f in p["flags"]:
        parts.append(f'<input type="hidden" name="flag" value="{esc(f)}">')
    if p["source"]:
        parts.append(f'<input type="hidden" name="source" value="{esc(p["source"])}">')
    if p["min_imp"]:
        parts.append(f'<input type="hidden" name="min_imp" value="{p["min_imp"]}">')
    parts.append("<button>Search</button></form>")

    parts.append('<div class="controls">')
    if p["q"]:
        parts.append(f'<span class="chip">“{esc(p["q"])}”'
                     f'<a href="{esc(make_url(p, q="", page=1))}" title="clear search">×</a></span>')
        parts.append("<span>ranked by importance, then relevance</span>")
    else:
        parts.append('<label>sort <select onchange="location=this.value">')
        for key, label in (("newest", "newest first"), ("oldest", "oldest first"),
                           ("importance", "importance")):
            sel = " selected" if p["sort"] == key else ""
            parts.append(f'<option value="{esc(make_url(p, sort=key, page=1))}"{sel}>'
                         f"{label}</option>")
        parts.append("</select></label>")
    for t in p["tags"]:
        parts.append(f'<span class="chip">{esc(t)}'
                     f'<a href="{esc(make_url(p, tags=toggle(p["tags"], t), page=1))}">×</a></span>')
    for f in p["flags"]:
        parts.append(f'<span class="chip">{esc(f.replace("_", " "))}'
                     f'<a href="{esc(make_url(p, flags=toggle(p["flags"], f), page=1))}">×</a></span>')
    n0, n1 = (offset + 1 if rows else 0), offset + len(rows)
    parts.append(f'<span class="result-count">{n0}–{n1} of {total}</span>')
    parts.append("</div></div>")

    if not rows:
        parts.append('<div class="empty"><b>No conversations match.</b>'
                     "Clear a filter, or try fewer search terms.</div>")

    timeline = not p["q"] and p["sort"] in ("newest", "oldest")
    last_month = None
    for r in rows:
        if timeline:
            m = month_label(r["created_at"])
            if m != last_month:
                parts.append(f'<div class="month"><h2>{esc(m)}</h2></div>')
                last_month = m
        parts.append(render_card(r, p))

    if total > PAGE_SIZE:
        prev_url = make_url(p, page=p["page"] - 1) if p["page"] > 1 else None
        next_url = make_url(p, page=p["page"] + 1) if offset + PAGE_SIZE < total else None
        parts.append('<div class="pager">')
        parts.append(f'<a href="{esc(prev_url)}">← Newer / previous</a>' if prev_url else "<span></span>")
        parts.append(f'<a href="{esc(next_url)}">Older / next →</a>' if next_url else "<span></span>")
        parts.append("</div>")

    return page_shell("kb — conversations", render_sidebar(conn, p), "".join(parts))


CLAMP_CHARS = 3000   # messages longer than this start collapsed


def render_conversation(conn, c, msgs, tags):
    p = parse_params("")
    head = ['<div class="conv-head">',
            '<div class="crumb"><a href="/">← All conversations</a></div>',
            f"<h1>{esc(c['title'] or '(untitled)')}</h1>",
            '<div class="conv-meta">',
            f"<span>{esc(c['source'])}</span>",
            f"<span>{esc(fmt_date(c['created_at'], True))}</span>",
            f"<span>{c['message_count'] or len(msgs)} messages</span>"]
    if c["model"]:
        head.append(f"<span>{esc(c['model'])}</span>")
    head.append(importance_dots(c["importance"]))
    head.append(f'<span title="{esc(c["id"])}">{esc(c["id"][:8])}</span>')
    head.append("</div>")
    if c["summary"]:
        head.append(f'<p class="conv-summary">{esc(c["summary"])}</p>')
    if tags:
        by_kind = {}
        for t in tags:
            by_kind.setdefault(t["kind"], []).append(t["tag"])
        head.append('<div class="tagrow">')
        for kind in KIND_ORDER:
            for tag in by_kind.get(kind, []):
                cls = ' class="proposed"' if kind == "proposed" else ""
                head.append(f'<a{cls} href="/?tag={quote(tag)}" '
                            f'title="{kind}">{esc(tag)}</a>')
        head.append("</div>")
    head.append("</div>")

    body = ['<div class="thread">']
    for m in msgs:
        role = (m["role"] or "?").lower()
        when = (f'<span class="when">{esc(fmt_date(m["timestamp"], True))}</span>'
                if m["timestamp"] else "")
        content = m["content"] or ""
        clamp = " clamp" if len(content) > CLAMP_CHARS else ""
        expand = ('<button class="expand">Show full message</button>' if clamp else "")
        body.append(
            f'<div class="msg role-{esc(role)}">'
            f'<div class="who">{esc(role)}{when}</div>'
            f'<div class="msg-body{clamp}">{render_message_html(content)}</div>'
            f"{expand}</div>")
    body.append('</div><a class="backtop" href="#top">↑ Back to top</a>')

    return page_shell(f"{c['title'] or c['id'][:8]} — kb",
                      render_sidebar(conn, p), "".join(head + body))


def render_ambiguous(conn, prefix, rows):
    p = parse_params("")
    items = "".join(
        f'<li><a href="/c/{quote(r["id"])}" class="mono">{esc(r["id"][:12])}</a> '
        f"— {esc(r['title'] or '(untitled)')}</li>" for r in rows)
    main = (f"<h1>Ambiguous prefix “{esc(prefix)}”</h1>"
            f"<p>It matches {len(rows)} conversations:</p><ul>{items}</ul>")
    return page_shell("ambiguous — kb", render_sidebar(conn, p), main)


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    server_version = "kbweb/1.0"

    def _send(self, body, status=200, ctype="text/html; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            conn = open_db()
        except sqlite3.OperationalError as e:
            self._send(f"<h1>Cannot open database</h1><p>{esc(DB_PATH)}: {esc(str(e))}</p>"
                       "<p>Point kb_web.py at your store with <code>--db path/to/kb.sqlite</code>.</p>",
                       status=500)
            return
        try:
            if url.path == "/":
                self._send(render_index(conn, parse_params(url.query)))
            elif url.path.startswith("/c/"):
                prefix = url.path[3:].strip("/")
                c, msgs, tags = fetch_conversation(conn, prefix)
                if c is None:
                    self._send("<h1>Not found</h1><p><a href='/'>← back</a></p>", status=404)
                elif c == "ambiguous":
                    self._send(render_ambiguous(conn, prefix, msgs))
                else:
                    self._send(render_conversation(conn, c, msgs, tags))
            elif url.path == "/favicon.ico":
                self._send("", status=404, ctype="text/plain")
            else:
                self._send("<h1>Not found</h1><p><a href='/'>← back</a></p>", status=404)
        except Exception as e:  # never take the server down for one bad page
            self._send(f"<h1>Error</h1><pre>{esc(type(e).__name__)}: {esc(str(e))}</pre>",
                       status=500)
        finally:
            conn.close()

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()}  {fmt % args}\n")


def main():
    global DB_PATH
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="kb.sqlite")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1 — local only)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    DB_PATH = args.db

    try:  # fail fast with a clear message instead of on first request
        open_db().close()
    except sqlite3.OperationalError as e:
        sys.exit(f"cannot open {DB_PATH}: {e}")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"kb_web: serving {DB_PATH} at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
