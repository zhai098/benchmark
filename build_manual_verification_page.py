from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path


ROOT = Path("/Users/zhaipengxiang/Desktop/benchmark")
MATCHED_PATH = ROOT / "high_difficulty_samples_acc_le_1_8_matched_alt_responses_no_special_flat_clean.jsonl"
CORRECTNESS_PATH = ROOT / "0410_gpt5.4rtn_high_difficulty_samples_acc_le_1_8_correctness_messages_from_matched_responses.jsonl"
OUTPUT_PATH = ROOT / "manual_verification_review.html"


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def parse_verdict(item: dict) -> dict:
    responses = item.get("responses") or []
    raw = responses[0] if responses else "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        notation_match = re.search(r'"conclusion"\s*:\s*"([^"]+)"', raw)
        explanation_match = re.search(r'"explanation"\s*:\s*"(.*)"\s*}', raw, flags=re.DOTALL)
        notation = notation_match.group(1).strip() if notation_match else "unknown"
        explanation = explanation_match.group(1) if explanation_match else raw
        explanation = explanation.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        parsed = {"conclusion": notation, "explanation": explanation}
    notation = str(parsed.get("conclusion") or "unknown").strip()
    is_true = notation.lower() == "correct"
    return {
        "notation": notation,
        "is_true": is_true,
        "explanation": str(parsed.get("explanation") or "").strip(),
    }


def normalize_markup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[i\](.*?)\[/i\]", r"<em>\1</em>", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[b\](.*?)\[/b\]", r"<strong>\1</strong>", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def text_to_html(text: str) -> str:
    escaped = escape(text, quote=False)
    escaped = normalize_markup(escaped)
    paragraphs = [part.strip() for part in escaped.split("\n\n")]
    blocks: list[str] = []
    for part in paragraphs:
        if not part:
            continue
        part = part.replace("\n", "<br>")
        blocks.append(f"<p>{part}</p>")
    inner = "".join(blocks) if blocks else "<p></p>"
    return f'<div class="math-content">{inner}</div>'


def build_records(matched_items: list[dict], correctness_items: list[dict]) -> tuple[list[dict], list[dict]]:
    joined_count = min(len(matched_items), len(correctness_items))
    records: list[dict] = []
    extras: list[dict] = []

    for idx in range(joined_count):
        matched = matched_items[idx]
        correctness = correctness_items[idx]
        verdict = parse_verdict(correctness)
        records.append(
            {
                "row_index": idx + 1,
                "correctness_index": correctness.get("index"),
                "filtered_index": matched.get("filtered_index"),
                "alternative_solution_index": matched.get("alternative_solution_index"),
                "problem": matched.get("problem", ""),
                "answer": matched.get("answer", ""),
                "alternative_solution": matched.get("alternative_solution", ""),
                "question_accuracy": matched.get("question_accuracy"),
                "notation": verdict["notation"],
                "is_true": verdict["is_true"],
                "explanation": verdict["explanation"],
            }
        )

    for idx in range(joined_count, len(correctness_items)):
        verdict = parse_verdict(correctness_items[idx])
        extras.append(
            {
                "row_index": idx + 1,
                "correctness_index": correctness_items[idx].get("index"),
                "notation": verdict["notation"],
                "is_true": verdict["is_true"],
                "explanation": verdict["explanation"],
            }
        )

    return records, extras


def card_html(record: dict) -> str:
    verdict_text = "True" if record["is_true"] else "False"
    verdict_class = "is-true" if record["is_true"] else "is-false"
    search_blob = " ".join(
        [
            str(record.get("filtered_index", "")),
            str(record.get("alternative_solution_index", "")),
            str(record.get("notation", "")),
            str(record.get("problem", "")),
            str(record.get("answer", "")),
            str(record.get("alternative_solution", "")),
            str(record.get("explanation", "")),
        ]
    ).lower()

    return f"""
    <article class="card" data-verdict="{escape(record['notation'].lower())}" data-search="{escape(search_blob)}">
      <div class="card-top">
        <div class="meta">
          <span class="chip">Row {record['row_index']}</span>
          <span class="chip">Correctness #{escape(str(record['correctness_index']))}</span>
          <span class="chip">Problem #{escape(str(record['filtered_index']))}</span>
          <span class="chip">Alt #{escape(str(record['alternative_solution_index']))}</span>
          <span class="chip">Question accuracy {escape(str(record['question_accuracy']))}</span>
        </div>
        <div class="verdict-group">
          <span class="verdict-pill {verdict_class}">{verdict_text}</span>
          <span class="notation">Notation: <code>{escape(record['notation'])}</code></span>
        </div>
      </div>

      <section class="section">
        <h2>Original Question</h2>
        {text_to_html(record["problem"])}
      </section>

      <section class="section answer-block">
        <h2>Reference Answer</h2>
        {text_to_html(record["answer"])}
      </section>

      <section class="section">
        <h2>Candidate Solution</h2>
        {text_to_html(record["alternative_solution"])}
      </section>

      <section class="section explanation-block">
        <h2>Explanation</h2>
        {text_to_html(record["explanation"])}
      </section>
    </article>
    """


def extras_html(extras: list[dict]) -> str:
    if not extras:
        return ""
    items = []
    for item in extras:
        verdict_text = "True" if item["is_true"] else "False"
        items.append(
            f"""
            <li>
              <strong>Correctness #{escape(str(item['correctness_index']))}</strong>
              <span class="verdict-inline">{verdict_text}</span>
              <code>{escape(item['notation'])}</code>
              <div>{text_to_html(item['explanation'])}</div>
            </li>
            """
        )
    return f"""
    <details class="extras">
      <summary>Unmatched correctness rows ({len(extras)})</summary>
      <ul>{''.join(items)}</ul>
    </details>
    """


def build_page(records: list[dict], extras: list[dict]) -> str:
    cards = "".join(card_html(record) for record in records)
    true_count = sum(1 for record in records if record["is_true"])
    false_count = len(records) - true_count

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual Verification Review</title>
  <link rel="preconnect" href="https://cdn.jsdelivr.net">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <style>
    :root {{
      --bg: #f7f4ee;
      --paper: #ffffff;
      --paper-strong: #fcfaf6;
      --ink: #1f1a17;
      --muted: #6e625b;
      --accent: #0f766e;
      --accent-soft: #e8f4f2;
      --bad: #b42318;
      --bad-soft: #fdecea;
      --line: #e6ded3;
      --shadow: 0 10px 28px rgba(70, 50, 33, 0.08);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background: var(--bg);
      min-height: 100vh;
    }}

    .shell {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }}

    .hero {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 22px;
      box-shadow: var(--shadow);
      margin-bottom: 20px;
    }}

    .eyebrow {{
      color: var(--accent);
      font-size: 0.85rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin: 0 0 10px;
    }}

    h1 {{
      margin: 0 0 10px;
      font-size: clamp(1.9rem, 3.4vw, 3rem);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}

    .subtitle {{
      margin: 0;
      max-width: 75ch;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.55;
    }}

    .toolbar {{
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) repeat(3, minmax(150px, 220px));
      gap: 12px;
      margin-top: 18px;
    }}

    .stats {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}

    .stat {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: var(--paper-strong);
      font-size: 0.92rem;
    }}

    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 11px 13px;
      background: var(--paper-strong);
      color: var(--ink);
      font: inherit;
      outline: none;
    }}

    input:focus, select:focus {{
      border-color: rgba(15, 118, 110, 0.45);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.08);
    }}

    .grid {{
      display: grid;
      gap: 18px;
    }}

    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      padding: 16px 18px 14px;
      background: var(--paper-strong);
      border-bottom: 1px solid var(--line);
    }}

    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .chip {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.84rem;
      color: var(--muted);
    }}

    .verdict-group {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    .verdict-pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 11px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}

    .is-true {{
      color: var(--accent);
      background: var(--accent-soft);
    }}

    .is-false {{
      color: var(--bad);
      background: var(--bad-soft);
    }}

    .notation {{
      color: var(--muted);
      font-size: 0.92rem;
    }}

    .section {{
      padding: 18px;
      border-top: 1px solid var(--line);
    }}

    .section:first-of-type {{
      border-top: 0;
    }}

    .section h2 {{
      margin: 0 0 14px;
      font-size: 1rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}

    .math-content p {{
      margin: 0 0 1em;
      line-height: 1.75;
      font-size: 1rem;
    }}

    .math-content p:last-child {{
      margin-bottom: 0;
    }}

    .answer-block {{
      background: #fafdfc;
    }}

    .explanation-block {{
      background: #fffaf8;
    }}

    code {{
      font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
      font-size: 0.92em;
      background: rgba(31, 26, 23, 0.06);
      padding: 0.14em 0.4em;
      border-radius: 0.4em;
    }}

    .hidden {{
      display: none !important;
    }}

    .empty {{
      display: none;
      text-align: center;
      padding: 24px;
      color: var(--muted);
      background: var(--paper);
      border: 1px dashed var(--line);
      border-radius: 18px;
      margin-bottom: 18px;
    }}

    .extras {{
      margin-top: 18px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
    }}

    .extras summary {{
      cursor: pointer;
      font-weight: 600;
    }}

    .extras ul {{
      margin: 14px 0 0;
      padding-left: 18px;
    }}

    .extras li {{
      margin-bottom: 14px;
      line-height: 1.6;
    }}

    .verdict-inline {{
      margin: 0 8px;
      font-weight: 700;
    }}

    .katex-display {{
      overflow-x: auto;
      overflow-y: hidden;
      padding: 0.15rem 0;
    }}

    @media (max-width: 1080px) {{
      .toolbar {{
        grid-template-columns: 1fr 1fr;
      }}
    }}

    @media (max-width: 720px) {{
      .shell {{
        width: min(100vw - 24px, 1400px);
        padding-top: 16px;
      }}

      .hero {{
        border-radius: 16px;
        padding: 18px 16px;
      }}

      .toolbar {{
        grid-template-columns: 1fr;
      }}

      .card-top {{
        flex-direction: column;
      }}

      .verdict-group {{
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <p class="eyebrow">Manual Verification</p>
      <h1>Question + Verdict Review Wall</h1>
      <p class="subtitle">
        Joined from <code>{escape(MATCHED_PATH.name)}</code> and <code>{escape(CORRECTNESS_PATH.name)}</code>.
        Each card shows the original question, reference answer, candidate solution, verdict as True or False,
        the raw notation from the correctness response, and the explanation for manual checking.
      </p>

      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="Search by question, solution, explanation, notation, or ids">
        <select id="verdictFilter">
          <option value="all">All verdicts</option>
          <option value="correct">Notation: correct</option>
          <option value="incorrect">Notation: incorrect</option>
        </select>
        <select id="sortMode">
          <option value="row">Sort: source order</option>
          <option value="problem">Sort: problem id</option>
          <option value="verdict">Sort: false first</option>
        </select>
        <input id="jumpBox" type="number" min="1" max="{len(records)}" placeholder="Jump to row">
      </div>

      <div class="stats">
        <span class="stat">Joined rows: <strong id="joinedCount">{len(records)}</strong></span>
        <span class="stat">True: <strong>{true_count}</strong></span>
        <span class="stat">False: <strong>{false_count}</strong></span>
        <span class="stat">Unmatched correctness rows: <strong>{len(extras)}</strong></span>
        <span class="stat">Visible now: <strong id="visibleCount">{len(records)}</strong></span>
      </div>
    </section>

    <div id="emptyState" class="empty">No cards match the current filter.</div>
    <main id="cardGrid" class="grid">
      {cards}
    </main>
    {extras_html(extras)}
  </div>

  <script>
    const searchBox = document.getElementById('searchBox');
    const verdictFilter = document.getElementById('verdictFilter');
    const sortMode = document.getElementById('sortMode');
    const jumpBox = document.getElementById('jumpBox');
    const visibleCount = document.getElementById('visibleCount');
    const emptyState = document.getElementById('emptyState');
    const grid = document.getElementById('cardGrid');

    function applyFilters() {{
      const query = searchBox.value.trim().toLowerCase();
      const verdict = verdictFilter.value;
      const cards = Array.from(grid.querySelectorAll('.card'));
      let visible = 0;

      cards.forEach((card) => {{
        const matchesQuery = !query || card.dataset.search.includes(query);
        const matchesVerdict = verdict === 'all' || card.dataset.verdict === verdict;
        const show = matchesQuery && matchesVerdict;
        card.classList.toggle('hidden', !show);
        if (show) visible += 1;
      }});

      visibleCount.textContent = String(visible);
      emptyState.style.display = visible === 0 ? 'block' : 'none';
    }}

    function applySort() {{
      const cards = Array.from(grid.querySelectorAll('.card'));
      const mode = sortMode.value;
      const sorted = cards.slice().sort((a, b) => {{
        const aRow = Number(a.querySelector('.chip').textContent.replace(/\\D+/g, ''));
        const bRow = Number(b.querySelector('.chip').textContent.replace(/\\D+/g, ''));
        const aProblem = Number(a.querySelectorAll('.chip')[2].textContent.replace(/\\D+/g, ''));
        const bProblem = Number(b.querySelectorAll('.chip')[2].textContent.replace(/\\D+/g, ''));
        const aVerdict = a.dataset.verdict === 'incorrect' ? 0 : 1;
        const bVerdict = b.dataset.verdict === 'incorrect' ? 0 : 1;

        if (mode === 'problem') {{
          return aProblem - bProblem || aRow - bRow;
        }}
        if (mode === 'verdict') {{
          return aVerdict - bVerdict || aProblem - bProblem || aRow - bRow;
        }}
        return aRow - bRow;
      }});

      sorted.forEach((card) => grid.appendChild(card));
    }}

    function jumpToRow() {{
      const row = Number(jumpBox.value);
      if (!row) return;
      const target = Array.from(grid.querySelectorAll('.card')).find((card) => {{
        return card.querySelector('.chip').textContent.trim() === `Row ${{row}}`;
      }});
      if (target) {{
        target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        target.animate(
          [
            {{ boxShadow: '0 0 0 rgba(15,118,110,0)' }},
            {{ boxShadow: '0 0 0 6px rgba(15,118,110,0.18)' }},
            {{ boxShadow: '0 18px 45px rgba(70,50,33,0.12)' }}
          ],
          {{ duration: 1200, easing: 'ease' }}
        );
      }}
    }}

    function renderMath() {{
      if (!window.renderMathInElement) return;
      document.querySelectorAll('.math-content').forEach((node) => {{
        if (node.dataset.mathRendered === 'true') return;
        window.renderMathInElement(node, {{
          delimiters: [
            {{ left: '$$', right: '$$', display: true }},
            {{ left: '\\\\[', right: '\\\\]', display: true }},
            {{ left: '$', right: '$', display: false }},
            {{ left: '\\\\(', right: '\\\\)', display: false }}
          ],
          throwOnError: false,
          ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        }});
        node.dataset.mathRendered = 'true';
      }});
    }}

    searchBox.addEventListener('input', applyFilters);
    verdictFilter.addEventListener('change', applyFilters);
    sortMode.addEventListener('change', () => {{
      applySort();
      applyFilters();
    }});
    jumpBox.addEventListener('change', jumpToRow);
    document.addEventListener('DOMContentLoaded', () => {{
      applySort();
      applyFilters();
      renderMath();
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    matched_items = load_jsonl(MATCHED_PATH)
    correctness_items = load_jsonl(CORRECTNESS_PATH)
    records, extras = build_records(matched_items, correctness_items)
    OUTPUT_PATH.write_text(build_page(records, extras), encoding="utf-8")
    print(f"Wrote {len(records)} joined rows to {OUTPUT_PATH}")
    if extras:
        print(f"Found {len(extras)} unmatched correctness rows")


if __name__ == "__main__":
    main()
