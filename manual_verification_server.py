from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from html import escape
from pathlib import Path

from flask import Flask, request, render_template_string, url_for


ROOT = Path("/Users/zhaipengxiang/Desktop/benchmark")
MATCHED_PATH = ROOT / "high_difficulty_samples_acc_le_1_8_matched_alt_responses_no_special_flat_clean.jsonl"
CORRECTNESS_PATH = ROOT / "0410_gpt5.4rtn_high_difficulty_samples_acc_le_1_8_correctness_messages_from_matched_responses.jsonl"

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual Verification Viewer</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <style>
    :root {
      --bg: #f6f3ee;
      --card: #ffffff;
      --card-soft: #fbfaf7;
      --line: #e8dfd4;
      --text: #221d19;
      --muted: #6b625c;
      --accent: #0f766e;
      --accent-bg: #e7f4f1;
      --bad: #b42318;
      --bad-bg: #fdecea;
      --shadow: 0 10px 28px rgba(64, 47, 34, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.6;
    }

    .page {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }

    .panel, .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }

    .panel {
      padding: 20px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(1.9rem, 3vw, 2.8rem);
      line-height: 1.05;
      letter-spacing: -0.02em;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      max-width: 80ch;
    }

    .controls {
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) 170px 120px 120px 140px;
      gap: 10px;
      margin-top: 16px;
    }

    input, select, button, .nav-link {
      font: inherit;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 10px 12px;
      background: #fff;
      color: var(--text);
      text-decoration: none;
    }

    button {
      cursor: pointer;
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }

    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }

    .stat {
      border: 1px solid var(--line);
      background: var(--card-soft);
      border-radius: 999px;
      padding: 7px 11px;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .cards {
      display: grid;
      gap: 16px;
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--card-soft);
      flex-wrap: wrap;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 0.84rem;
      color: var(--muted);
      background: #fff;
    }

    .verdict {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }

    .pill {
      border-radius: 999px;
      padding: 6px 10px;
      font-weight: 700;
    }

    .pill.true {
      color: var(--accent);
      background: var(--accent-bg);
    }

    .pill.false {
      color: var(--bad);
      background: var(--bad-bg);
    }

    .section {
      padding: 16px 18px;
      border-top: 1px solid var(--line);
    }

    .section:first-of-type {
      border-top: 0;
    }

    .section h2 {
      margin: 0 0 10px;
      font-size: 0.95rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .math-content p {
      margin: 0 0 1em;
      font-size: 1rem;
      line-height: 1.75;
    }

    .math-content p:last-child {
      margin-bottom: 0;
    }

    .answer { background: #fafdfc; }
    .explanation { background: #fffaf8; }

    .katex-display {
      overflow-x: auto;
      overflow-y: hidden;
      padding: 0.15rem 0;
    }

    .pager {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin: 18px 0;
      flex-wrap: wrap;
    }

    .pager-group {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }

    .empty {
      padding: 26px;
      text-align: center;
      color: var(--muted);
    }

    code {
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      background: rgba(34, 29, 25, 0.06);
      border-radius: 6px;
      padding: 0.12em 0.35em;
    }

    @media (max-width: 860px) {
      .controls {
        grid-template-columns: 1fr 1fr;
      }
    }

    @media (max-width: 620px) {
      .page {
        width: min(100vw - 20px, 1120px);
      }
      .controls {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="panel">
      <h1>Manual Verification Viewer</h1>
      <p class="subtitle">
        Server-rendered pagination for the matched response and correctness files.
        Only the visible page is rendered, so LaTeX can be processed reliably.
      </p>

      <form method="get" action="/" class="controls">
        <input type="search" name="q" value="{{ q }}" placeholder="Search question, answer, solution, explanation, ids">
        <select name="verdict">
          <option value="all" {% if verdict == 'all' %}selected{% endif %}>All verdicts</option>
          <option value="correct" {% if verdict == 'correct' %}selected{% endif %}>Notation: correct</option>
          <option value="incorrect" {% if verdict == 'incorrect' %}selected{% endif %}>Notation: incorrect</option>
        </select>
        <select name="per_page">
          {% for value in [10, 20, 30, 50] %}
          <option value="{{ value }}" {% if per_page == value %}selected{% endif %}>{{ value }}/page</option>
          {% endfor %}
        </select>
        <input type="number" min="1" name="page" value="{{ page }}" placeholder="Page">
        <button type="submit">Apply</button>
      </form>

      <div class="stats">
        <span class="stat">Matched rows: <strong>{{ total }}</strong></span>
        <span class="stat">Filtered rows: <strong>{{ filtered_total }}</strong></span>
        <span class="stat">Page: <strong>{{ page }}</strong> / {{ total_pages }}</span>
        <span class="stat">True on page: <strong>{{ page_true }}</strong></span>
        <span class="stat">False on page: <strong>{{ page_false }}</strong></span>
      </div>
    </section>

    <div class="pager">
      <div class="pager-group">
        {% if prev_url %}
        <a class="nav-link" href="{{ prev_url }}">Previous</a>
        {% endif %}
        {% if next_url %}
        <a class="nav-link" href="{{ next_url }}">Next</a>
        {% endif %}
      </div>
      <div class="pager-group">
        <span>Showing {{ start_index }}-{{ end_index }} of {{ filtered_total }}</span>
      </div>
    </div>

    <main class="cards">
      {% if items %}
        {% for item in items %}
        <article class="card">
          <div class="card-header">
            <div class="chips">
              <span class="chip">Row {{ item.row_index }}</span>
              <span class="chip">Correctness #{{ item.correctness_index }}</span>
              <span class="chip">Problem #{{ item.filtered_index }}</span>
              <span class="chip">Alt #{{ item.alternative_solution_index }}</span>
              <span class="chip">Question accuracy {{ item.question_accuracy }}</span>
            </div>
            <div class="verdict">
              <span class="pill {{ 'true' if item.is_true else 'false' }}">{{ 'True' if item.is_true else 'False' }}</span>
              <span>Notation: <code>{{ item.notation }}</code></span>
            </div>
          </div>

          <section class="section">
            <h2>Original Question</h2>
            {{ item.problem_html|safe }}
          </section>
          <section class="section answer">
            <h2>Reference Answer</h2>
            {{ item.answer_html|safe }}
          </section>
          <section class="section">
            <h2>Candidate Solution</h2>
            {{ item.solution_html|safe }}
          </section>
          <section class="section explanation">
            <h2>Explanation</h2>
            {{ item.explanation_html|safe }}
          </section>
        </article>
        {% endfor %}
      {% else %}
        <section class="panel empty">No rows match the current filter.</section>
      {% endif %}
    </main>

    <div class="pager">
      <div class="pager-group">
        {% if prev_url %}
        <a class="nav-link" href="{{ prev_url }}">Previous</a>
        {% endif %}
        {% if next_url %}
        <a class="nav-link" href="{{ next_url }}">Next</a>
        {% endif %}
      </div>
      <div class="pager-group">
        <span>Use smaller page sizes if your browser still feels slow.</span>
      </div>
    </div>
  </div>

  <script>
    function renderMath() {
      if (!window.renderMathInElement) return;
      document.querySelectorAll('.math-content').forEach((node) => {
        if (node.dataset.rendered === 'true') return;
        window.renderMathInElement(node, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\\\[', right: '\\\\]', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\\\(', right: '\\\\)', display: false }
          ],
          throwOnError: false,
          ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        });
        node.dataset.rendered = 'true';
      });
    }

    document.addEventListener('DOMContentLoaded', renderMath);
    window.addEventListener('load', renderMath);
  </script>
</body>
</html>
"""

app = Flask(__name__)


def normalize_markup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[i\](.*?)\[/i\]", r"<em>\1</em>", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[b\](.*?)\[/b\]", r"<strong>\1</strong>", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def text_to_html(text: str) -> str:
    escaped = escape(text, quote=False)
    escaped = normalize_markup(escaped)
    parts = [part.strip() for part in escaped.split("\n\n") if part.strip()]
    inner = "".join(f"<p>{part.replace(chr(10), '<br>')}</p>" for part in parts) or "<p></p>"
    return f'<div class="math-content">{inner}</div>'


def math_answer_to_html(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return '<div class="math-content"><p></p></div>'
    return f'<div class="math-content"><p>\\[{escape(stripped, quote=False)}\\]</p></div>'


def parse_verdict(item: dict) -> dict:
    raw = (item.get("responses") or ["{}"])[0]
    try:
      parsed = json.loads(raw)
    except json.JSONDecodeError:
      notation_match = re.search(r'"conclusion"\s*:\s*"([^"]+)"', raw)
      explanation_match = re.search(r'"explanation"\s*:\s*"(.*)"\s*}', raw, flags=re.DOTALL)
      parsed = {
          "conclusion": notation_match.group(1).strip() if notation_match else "unknown",
          "explanation": explanation_match.group(1) if explanation_match else raw,
      }
    notation = str(parsed.get("conclusion") or "unknown").strip()
    explanation = str(parsed.get("explanation") or "").replace("\\n", "\n").replace('\\"', '"')
    return {
        "notation": notation,
        "is_true": notation.lower() == "correct",
        "explanation": explanation,
    }


@lru_cache(maxsize=1)
def load_records() -> list[dict]:
    matched_items = [json.loads(line) for line in MATCHED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    correctness_items = [json.loads(line) for line in CORRECTNESS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    joined_count = min(len(matched_items), len(correctness_items))
    records: list[dict] = []
    for idx in range(joined_count):
        matched = matched_items[idx]
        verdict = parse_verdict(correctness_items[idx])
        records.append(
            {
                "row_index": idx + 1,
                "correctness_index": correctness_items[idx].get("index"),
                "filtered_index": matched.get("filtered_index"),
                "alternative_solution_index": matched.get("alternative_solution_index"),
                "question_accuracy": matched.get("question_accuracy"),
                "notation": verdict["notation"],
                "is_true": verdict["is_true"],
                "problem": matched.get("problem", ""),
                "answer": matched.get("answer", ""),
                "alternative_solution": matched.get("alternative_solution", ""),
                "explanation": verdict["explanation"],
            }
        )
    return records


def filter_records(records: list[dict], query: str, verdict: str) -> list[dict]:
    query = query.strip().lower()
    out: list[dict] = []
    for record in records:
        if verdict != "all" and record["notation"].lower() != verdict:
            continue
        if query:
            haystack = " ".join(
                [
                    str(record["filtered_index"]),
                    str(record["alternative_solution_index"]),
                    str(record["notation"]),
                    record["problem"],
                    record["answer"],
                    record["alternative_solution"],
                    record["explanation"],
                ]
            ).lower()
            if query not in haystack:
                continue
        out.append(record)
    return out


def page_url(page: int, q: str, verdict: str, per_page: int) -> str:
    return url_for("index", page=page, q=q, verdict=verdict, per_page=per_page)


@app.get("/")
def index():
    records = load_records()
    q = str(request.args.get("q", ""))
    verdict = str(request.args.get("verdict", "all"))
    per_page = max(1, min(50, request.args.get("per_page", type=int, default=20)))
    page = max(1, request.args.get("page", type=int, default=1))

    filtered = filter_records(records, query=q, verdict=verdict)
    filtered_total = len(filtered)
    total_pages = max(1, math.ceil(filtered_total / per_page))
    page = min(page, total_pages)
    start = (page - 1) * per_page
    end = start + per_page
    items = filtered[start:end]

    prepared = []
    for item in items:
        prepared.append(
            {
                **item,
                "problem_html": text_to_html(item["problem"]),
                "answer_html": math_answer_to_html(item["answer"]),
                "solution_html": text_to_html(item["alternative_solution"]),
                "explanation_html": text_to_html(item["explanation"]),
            }
        )

    prev_url = page_url(page - 1, q, verdict, per_page) if page > 1 else None
    next_url = page_url(page + 1, q, verdict, per_page) if page < total_pages else None
    page_true = sum(1 for item in prepared if item["is_true"])
    page_false = len(prepared) - page_true

    return render_template_string(
        PAGE_TEMPLATE,
        items=prepared,
        q=q,
        verdict=verdict,
        per_page=per_page,
        page=page,
        total=len(records),
        filtered_total=filtered_total,
        total_pages=total_pages,
        start_index=(start + 1 if filtered_total else 0),
        end_index=min(end, filtered_total),
        prev_url=prev_url,
        next_url=next_url,
        page_true=page_true,
        page_false=page_false,
    )


if __name__ == "__main__":
      app.run(host="0.0.0.0", port=8010, debug=False, use_reloader=False)