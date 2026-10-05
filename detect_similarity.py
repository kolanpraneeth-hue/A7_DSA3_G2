#!/usr/bin/env python3
"""
Semantic plagiarism / document similarity detector.

Usage:
  python detect_similarity.py --query samples/student_essay.txt --ref samples/source_paper.txt
  python detect_similarity.py -q samples/student_essay.txt -r samples/source_paper.txt --html
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from preprocess import load_and_chunk
from semantic_engine import SemanticEngine, SimilarityResult


def print_report(result: SimilarityResult, query_path: str, ref_path: str) -> None:
    print()
    print("========== SIMILARITY REPORT ==========")
    print(f"Query file                 : {query_path}")
    print(f"Reference file             : {ref_path}")
    print(f"Model                      : {result.model_name}")
    print(f"Query sentences            : {result.query_count}")
    print(f"Reference sentences        : {result.ref_count}")
    print(f"Overall Semantic Similarity: {result.overall_score}%")
    print(f"Decision Class             : {result.decision}")
    print(
        f"Thresholds                 : "
     f">= {result.threshold_high} -> HIGH | "
     f">= {result.threshold_medium} -> MEDIUM | else LOW"
    )
    print()
    print("Top matched sentence pairs:")
    if not result.matches:
        print("  (no strong matches found)")
    for i, m in enumerate(result.matches, 1):
        print(f"  [{i}] score={m.score:.2f}")
        print(f'      Q: "{m.query_sentence}"')
        print(f'      R: "{m.ref_sentence}"')
    print("=======================================")


def save_html_report(
    result: SimilarityResult,
    query_path: str,
    ref_path: str,
    out_path: str,
) -> None:
    rows = []
    for i, m in enumerate(result.matches, 1):
        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td>{m.score:.2f}</td>"
            f"<td>{m.query_sentence}</td>"
            f"<td>{m.ref_sentence}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Similarity Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f6f4ef; color: #1b2420; }}
    h1 {{ color: #3b5d3a; }}
    .card {{ background: white; padding: 1.25rem 1.5rem; border-radius: 8px;
             box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 1.25rem; }}
    .score {{ font-size: 2rem; font-weight: 700; color: #3b5d3a; }}
    .high {{ color: #c26a33; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 0.6rem; border-bottom: 1px solid #e0dcd0; vertical-align: top; }}
    th {{ background: #3b5d3a; color: white; }}
    .meta {{ color: #7a7f76; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Semantic Similarity Report</h1>
  <div class="card">
    <div class="score">{result.overall_score}%
      <span class="high">({result.decision})</span>
    </div>
    <p class="meta">
      Query: {query_path}<br/>
      Reference: {ref_path}<br/>
      Model: {result.model_name}<br/>
      Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </p>
  </div>
  <div class="card">
    <h2>Matched sentence pairs</h2>
    <table>
      <thead>
        <tr><th>#</th><th>Score</th><th>Query sentence</th><th>Reference sentence</th></tr>
      </thead>
      <tbody>
        {"".join(rows) if rows else "<tr><td colspan='4'>No strong matches</td></tr>"}
      </tbody>
    </table>
  </div>
  <p class="meta">
    Similarity is an indicator, not proof of plagiarism. Final decisions should include human review.
  </p>
</body>
</html>
"""
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"[INFO] Report saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic document similarity detector (Sentence-Transformers + FAISS)"
    )
    parser.add_argument("-q", "--query", required=True, help="Query / submitted document path")
    parser.add_argument("-r", "--ref", required=True, help="Reference document path")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument("--high", type=float, default=0.65, help="HIGH threshold (default 0.65)")
    parser.add_argument("--medium", type=float, default=0.40, help="MEDIUM threshold (default 0.40)")
    parser.add_argument("--html", action="store_true", help="Also save an HTML report")
    parser.add_argument(
        "--out",
        default="reports/similarity_report.html",
        help="HTML report output path",
    )
    args = parser.parse_args()

    print(f"[INFO] Extracting & chunking documents ...")
    query_sents = load_and_chunk(args.query)
    ref_sents = load_and_chunk(args.ref)

    engine = SemanticEngine(
        model_name=args.model,
        threshold_high=args.high,
        threshold_medium=args.medium,
    )
    engine.build_index(ref_sents)
    result = engine.compare(query_sents)

    print_report(result, args.query, args.ref)

    if args.html:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_html_report(result, args.query, args.ref, args.out)


if __name__ == "__main__":
    main()
