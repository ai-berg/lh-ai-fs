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
CASES_DIR = HERE / "cases"

# Case registry. Each case = its documents + gold set + a committed snapshot, so
# the default run scores every case offline. The provided Rivera case uses the
# repo's documents/ dir and the existing fixtures; synthetic cases live self-
# contained under eval/cases/ and exist to show the method generalizes past n=1.
CASES = [
    {
        "name": "rivera_v_harmon (provided)",
        "docs": load_documents,                       # backend/documents/
        "gold": GOLD,
        "snapshot": SNAPSHOT,
        "pregate": PREGATE,                           # has the ablation fixtures
    },
    {
        "name": "northgate_v_brightway (synthetic)",
        "docs": lambda: _load_dir(CASES_DIR / "synthetic_contract"),
        "gold": CASES_DIR / "synthetic_contract" / "gold.yaml",
        "snapshot": CASES_DIR / "synthetic_contract" / "snapshot.json",
        "pregate": None,
    },
]


def _load_dir(path: Path) -> dict:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(path.glob("*.txt"))}


def _load_run(case: dict, live: bool, docs: dict):
    """Return (report_dict, raw_findings) for one case's run.

    Default: the case's committed snapshot (+ pre-gate fixture if present) — no
    API. Live: run the agents ONCE and derive both the grounded report and the raw
    findings from that single run, so scoring and the ablation describe the same
    invocation (and the agents aren't called twice).
    """
    if not live:
        from schemas import Finding

        report = json.loads(Path(case["snapshot"]).read_text())
        if case.get("pregate"):
            raw = [Finding(**f) for f in json.loads(Path(case["pregate"]).read_text())["flags"]]
        else:
            raw = None  # no committed pre-gate fixture for this case
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


def _ci(ci):
    return f"95% CI [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else ""


def _run_case(case: dict, live: bool) -> dict:
    """Score one case, print its report, and return its recall/precision tallies."""
    gold = yaml.safe_load(Path(case["gold"]).read_text())
    docs = case["docs"]()
    report, raw_findings = _load_run(case, live, docs)
    r = score(gold, report, docs)
    rec, prec, gc = r["recall"], r["precision"], r["grounding_consistency"]

    print(f"\n── CASE: {case['name']}   ({'live' if live else 'snapshot'}) ──")
    print(f"RECALL (planted flaws caught)   {rec['caught']}/{rec['total']}   {_ci(rec['ci95'])}")
    for f in rec["per_flaw"]:
        print(f"  [{'x' if f['caught'] else ' '}] {f['id']:26} ({f['axis']})")

    band = f"[{_pct(prec['value_low'])}, {_pct(prec['value'])}]" if prec["value"] is not None else "n/a"
    print(f"PRECISION (avoiding false flags)   {band}"
          f"   TP={prec['true_positives']} FP={prec['false_positives']}"
          f" pending={len(prec['pending_adjudication'])}   {_ci(prec['ci95'])}")
    print(f"  checked {prec['negatives_checked']} labeled negatives (true MSJ statements"
          f" that must not be flagged), {prec['false_positives']} incorrectly flagged")
    for fp in prec["fp_detail"]:
        print(f"  FALSE POSITIVE on negative '{fp['negative']}': {fp['claim']}")
    for c in prec["pending_adjudication"]:
        print(f"  pending (unplanted, not scored): {c}")

    # Labeled "hallucination" so the spec term is discoverable; it is the
    # grounding-consistency signal (see header caveat: regression guard, not oracle).
    print(f"HALLUCINATION / GROUNDING   {gc['ungrounded_quotes']}/{gc['total_quotes']} ungrounded quotes,"
          f" {gc['unsupported_assertions']} unsupported assertion(s)")
    for u in gc["detail"]:
        print(f"  UNGROUNDED in {u['doc']}: {u['quote'][:70]}")

    # Ablation only where a pre-gate fixture (or a live run) provides raw findings.
    if raw_findings is not None:
        ab = _pre_post_gate(docs, raw_findings)
        print(
            f"GROUNDING-GATE ABLATION:"
            f" pre-gate {ab['pre']['ungrounded_quotes']} ungrounded / {ab['pre_assertive']} assertive"
            f" -> post-gate {ab['post']['ungrounded_quotes']} / {ab['post_assertive']}"
            f"  (gate cleared {ab['pre']['ungrounded_quotes'] - ab['post']['ungrounded_quotes']} quote(s),"
            f" downgraded {ab['pre_assertive'] - ab['post_assertive']} finding(s))"
        )

    return {"caught": rec["caught"], "total": rec["total"],
            "tp": prec["true_positives"], "fp": prec["false_positives"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the BS Detector against the gold set(s).")
    parser.add_argument("--live", action="store_true", help="run the real pipeline instead of the snapshots")
    args = parser.parse_args()

    print("\nBS DETECTOR — EVAL REPORT")
    print("(small gold sets — read the k/n fractions and CIs, not the point %. The"
          " grounding-consistency rate re-runs the pipeline's own check: a regression"
          " guard, not an independent hallucination oracle.)")

    tallies = [_run_case(c, args.live) for c in CASES]

    caught = sum(t["caught"] for t in tallies)
    total = sum(t["total"] for t in tallies)
    tp = sum(t["tp"] for t in tallies)
    fp = sum(t["fp"] for t in tallies)
    from eval.metrics import wilson_ci

    print(f"\n══ AGGREGATE over {len(CASES)} cases ══")
    print(f"RECALL    {caught}/{total}   95% CI [{_pct(wilson_ci(caught, total)[0])},"
          f" {_pct(wilson_ci(caught, total)[1])}]")
    prec = tp / (tp + fp) if (tp + fp) else None
    print(f"PRECISION {_pct(prec)}   TP={tp} FP={fp}")
    print("(still small-N — two cases prove the method generalizes past one fixture,"
          " not that the rates are statistically settled.)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
