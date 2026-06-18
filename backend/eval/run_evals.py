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


def _load_run(live: bool, docs: dict):
    """Return (report_dict, raw_findings) for one run.

    Default: the committed post-gate snapshot + the committed pre-gate findings —
    no API. Live: run the agents ONCE and derive both the grounded report and the
    raw findings from that single run, so scoring and the ablation describe the
    same pipeline invocation (and the agents aren't called twice).
    """
    if not live:
        from schemas import Finding

        report = json.loads(SNAPSHOT.read_text())
        raw = [Finding(**f) for f in json.loads(PREGATE.read_text())["flags"]]
        return report, raw

    from services.orchestrator import apply_grounding, run_agents

    citations, findings, degraded = asyncio.run(run_agents(docs))
    grounded_citations, grounded = apply_grounding(citations, findings, docs)
    report = {
        "citations": [c.model_dump() for c in grounded_citations],
        "flags": [f.model_dump() for f in grounded],
        "degraded_agents": degraded,
    }
    return report, findings


def _pre_post_gate(docs: dict, raw_findings) -> dict:
    """Ablation: apply the grounding gate to the SAME raw findings and diff.

    Both sides come from one set of raw (pre-gate) findings: the post-gate side is
    that exact input passed through ``apply_grounding`` — never an unrelated
    snapshot — so the delta honestly reflects what the gate did to this run.
    Default uses the committed pre-gate fixture (no API); --live passes fresh
    findings already captured by the scored run (so the agents aren't re-run).
    """
    from services.orchestrator import apply_grounding

    pre = list(raw_findings)
    _, post = apply_grounding([], pre, docs)
    pre_flags = [f.model_dump() for f in pre]
    post_flags = [f.model_dump() for f in post]

    # A finding whose status survives as assertive (not could_not_verify) is one
    # the gate "kept"; the delta of assertive flags shows the gate dropping a
    # fully-fabricated finding, which a quote-only count misses.
    def _assertive(flags):
        return sum(1 for f in flags if f.get("status") in ("contradicted", "verified"))

    return {
        "pre": grounding_consistency_rate(pre_flags, docs),
        "post": grounding_consistency_rate(post_flags, docs),
        "pre_assertive": _assertive(pre_flags),
        "post_assertive": _assertive(post_flags),
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the BS Detector against the gold set.")
    parser.add_argument("--live", action="store_true", help="run the real pipeline instead of the snapshot")
    args = parser.parse_args()

    gold = yaml.safe_load(GOLD.read_text())
    docs = load_documents()
    report, raw_findings = _load_run(args.live, docs)
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

    # Ablation always runs, applying the gate to the SAME raw findings used above
    # (committed pre-gate fixture by default; the live run's raw findings under --live).
    ab = _pre_post_gate(docs, raw_findings)
    src = "live" if args.live else "committed pre-gate fixture -> grounding gate"
    quotes_removed = ab["pre"]["ungrounded_quotes"] - ab["post"]["ungrounded_quotes"]
    flags_dropped = ab["pre_assertive"] - ab["post_assertive"]
    print(
        f"\nGROUNDING-GATE ABLATION ({src})"
        f"\n  pre-gate  (raw model):  {ab['pre']['ungrounded_quotes']} ungrounded quote(s),"
        f" {ab['pre_assertive']} assertive finding(s)"
        f"\n  post-gate (shipped):    {ab['post']['ungrounded_quotes']} ungrounded quote(s),"
        f" {ab['post_assertive']} assertive finding(s)"
        f"\n  -> the gate cleared {quotes_removed} ungrounded quote(s) and downgraded"
        f" {flags_dropped} fabricated finding(s) to could_not_verify."
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
