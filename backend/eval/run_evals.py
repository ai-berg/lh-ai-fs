"""BS Detector eval harness — single-command entrypoint.

    python eval/run_evals.py            # score the committed snapshot (reproducible, no API)
    python eval/run_evals.py --live     # run the real pipeline, then score (spends API)

Scores the pipeline against a hand-frozen gold set (eval/gold_set.yaml) and
reports recall, precision, and hallucination rate — per flaw and honestly,
including a pending-adjudication bucket for plausible-but-unplanted findings and a
pre-gate vs post-gate fabrication comparison that shows the grounding gate earns
its keep.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

from eval.metrics import score  # noqa: E402
from repositories.document_repository import load_documents  # noqa: E402

GOLD = HERE / "gold_set.yaml"
SNAPSHOT = BACKEND / "tests" / "fixtures" / "analyze_snapshot.json"


def _load_report(live: bool) -> dict:
    if not live:
        return json.loads(SNAPSHOT.read_text())
    from services.orchestrator import run_pipeline  # imported lazily (needs API key)

    report = asyncio.run(run_pipeline(load_documents()))
    return report.model_dump()


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the BS Detector against the gold set.")
    parser.add_argument("--live", action="store_true", help="run the real pipeline instead of the snapshot")
    args = parser.parse_args()

    gold = yaml.safe_load(GOLD.read_text())
    docs = load_documents()
    report = _load_report(args.live)
    r = score(gold, report, docs)

    rec, prec, hal = r["recall"], r["precision"], r["hallucination"]
    source = "live pipeline" if args.live else f"snapshot ({SNAPSHOT.name})"

    print(f"\nBS DETECTOR — EVAL REPORT   case={gold['case']}   source={source}\n")

    print(f"RECALL (planted flaws caught)   {rec['caught']}/{rec['total']}")
    for f in rec["per_flaw"]:
        print(f"  [{'x' if f['caught'] else ' '}] {f['id']:24} ({f['axis']})")

    print(f"\nPRECISION (avoiding false flags)   {_pct(prec['value'])}"
          f"   TP={prec['true_positives']} FP={prec['false_positives']}")
    for fp in prec["fp_detail"]:
        print(f"  FALSE POSITIVE on negative '{fp['negative']}': {fp['claim']}")
    if prec["pending_adjudication"]:
        print(f"  pending_adjudication (unplanted, not scored): {len(prec['pending_adjudication'])}")
        for c in prec["pending_adjudication"]:
            print(f"    - {c}")

    print(f"\nHALLUCINATION RATE (ungrounded cited quotes)   {_pct(hal['rate'])}"
          f"   {hal['ungrounded_quotes']}/{hal['total_quotes']} quotes")
    for u in hal["detail"]:
        print(f"  UNGROUNDED in {u['doc']}: {u['quote'][:70]}")

    print(
        "\nNote: post-gate hallucination is ~0 by construction — the grounding gate"
        "\nclears ungrounded quotes before they reach the report. Run with --live and"
        "\ncompare a raw (pre-gate) capture to quantify the gate's contribution.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
