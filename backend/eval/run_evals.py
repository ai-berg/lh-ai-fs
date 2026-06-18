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

from eval.metrics import grounding_consistency_rate, score  # noqa: E402
from repositories.document_repository import load_documents  # noqa: E402

GOLD = HERE / "gold_set.yaml"
SNAPSHOT = BACKEND / "tests" / "fixtures" / "analyze_snapshot.json"
PREGATE = BACKEND / "tests" / "fixtures" / "pregate_snapshot.json"


def _load_report(live: bool) -> dict:
    if not live:
        return json.loads(SNAPSHOT.read_text())
    from services.orchestrator import run_pipeline  # imported lazily (needs API key)

    report = asyncio.run(run_pipeline(load_documents()))
    return report.model_dump()


def _pre_post_gate(docs: dict, live: bool) -> dict:
    """Ablation: grounding consistency in the RAW agent output vs the gated report.

    Quantifies what the grounding gate removes. By default it uses committed
    fixtures (pre-gate + post-gate snapshots) so the ablation appears with no API
    spend; --live recomputes both from a fresh pipeline run.
    """
    if not live:
        pre_flags = json.loads(PREGATE.read_text())["flags"]
        post_flags = json.loads(SNAPSHOT.read_text())["flags"]
        return {
            "pre": grounding_consistency_rate(pre_flags, docs),
            "post": grounding_consistency_rate(post_flags, docs),
        }

    from services.orchestrator import apply_grounding, run_agents

    citations, findings, _ = asyncio.run(run_agents(docs))
    pre = grounding_consistency_rate([f.model_dump() for f in findings], docs)
    _, grounded = apply_grounding(citations, findings, docs)
    post = grounding_consistency_rate([f.model_dump() for f in grounded], docs)
    return {"pre": pre, "post": post}


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

    rec, prec, gc = r["recall"], r["precision"], r["grounding_consistency"]
    source = "live pipeline" if args.live else f"snapshot ({SNAPSHOT.name})"

    def _ci(ci):
        return f"95% CI [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else ""

    print(f"\nBS DETECTOR — EVAL REPORT   case={gold['case']}   source={source}")
    print("(small fixture, n=1 case — read the k/n fractions and CIs, not the point %.)\n")

    print(f"RECALL (planted flaws caught)   {rec['caught']}/{rec['total']}   {_ci(rec['ci95'])}")
    for f in rec["per_flaw"]:
        print(f"  [{'x' if f['caught'] else ' '}] {f['id']:24} ({f['axis']})")

    # Precision as a band: lower bound assumes every pending finding is an FP.
    band = f"[{_pct(prec['value_low'])}, {_pct(prec['value'])}]" if prec["value"] is not None else "n/a"
    print(f"\nPRECISION (avoiding false flags)   {band}"
          f"   TP={prec['true_positives']} FP={prec['false_positives']}"
          f" pending={len(prec['pending_adjudication'])}   {_ci(prec['ci95'])}")
    for fp in prec["fp_detail"]:
        print(f"  FALSE POSITIVE on negative '{fp['negative']}': {fp['claim']}")
    for c in prec["pending_adjudication"]:
        print(f"  pending (unplanted, not scored): {c}")

    print(f"\nGROUNDING CONSISTENCY (cited quotes absent from source)   {_pct(gc['rate'])}"
          f"   {gc['ungrounded_quotes']}/{gc['total_quotes']} quotes")
    print(f"  unsupported assertions (assertive finding, no evidence)   {gc['unsupported_assertions']}")
    print("  (re-runs the pipeline's OWN grounding check — a regression guard, not an"
          " independent hallucination oracle; ~0 post-gate by construction.)")
    for u in gc["detail"]:
        print(f"  UNGROUNDED in {u['doc']}: {u['quote'][:70]}")
    for c in gc["unsupported_detail"]:
        print(f"  UNSUPPORTED ASSERTION: {c}")

    # Ablation always runs: from committed pre/post fixtures by default (no API),
    # recomputed live with --live.
    ab = _pre_post_gate(docs, args.live)
    src = "live" if args.live else "committed pre-gate vs post-gate fixtures"
    print(
        f"\nGROUNDING-GATE ABLATION ({src})"
        f"\n  pre-gate  (raw model):  {_pct(ab['pre']['rate'])}"
        f"   {ab['pre']['ungrounded_quotes']}/{ab['pre']['total_quotes']} quotes"
        f"\n  post-gate (shipped):    {_pct(ab['post']['rate'])}"
        f"   {ab['post']['ungrounded_quotes']}/{ab['post']['total_quotes']} quotes"
        f"\n  -> the gate removed {ab['pre']['ungrounded_quotes'] - ab['post']['ungrounded_quotes']}"
        f" ungrounded quote(s) before they reached the report."
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
