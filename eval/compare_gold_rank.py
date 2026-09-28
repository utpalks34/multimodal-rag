"""Compare gold_rank / gold_hit@5 / gold_hit@10 between two eval.run_eval --stage retrieval
outputs (retrieval_only.json), per question and per type. Diagnostic only, no pipeline call.

Usage: python -m eval.compare_gold_rank <before.json> <after.json>
"""
import json
import statistics
import sys


def _load(path: str) -> dict:
    return {r["id"]: r for r in json.loads(open(path, encoding="utf-8").read())}


def main(before_path: str, after_path: str) -> None:
    before, after = _load(before_path), _load(after_path)
    ids = sorted(set(before) | set(after))
    print(f"{'id':8s} {'type':6s} {'rank before':>12s} {'rank after':>11s} {'hit5 b->a':>10s} {'hit10 b->a':>11s}")
    for i in ids:
        b, a = before.get(i, {}), after.get(i, {})
        t = a.get("type") or b.get("type")
        print(f"{i:8s} {t:6s} {str(b.get('gold_rank')):>12s} {str(a.get('gold_rank')):>11s} "
              f"{str(b.get('gold_hit_at_5'))}->{a.get('gold_hit_at_5'):>10} "
              f"{str(b.get('gold_hit_at_10'))}->{a.get('gold_hit_at_10')}")

    print("\nBy type (mean hit@5 / hit@10, before -> after):")
    for t in sorted({r.get("type") for r in {**before, **after}.values()}):
        for label, data in (("before", before), ("after", after)):
            h5 = [r["gold_hit_at_5"] for r in data.values() if r["type"] == t and r["gold_hit_at_5"] is not None]
            h10 = [r["gold_hit_at_10"] for r in data.values() if r["type"] == t and r["gold_hit_at_10"] is not None]
            m5 = round(statistics.mean(h5), 3) if h5 else None
            m10 = round(statistics.mean(h10), 3) if h10 else None
            print(f"  {t:6s} {label:7s} hit@5={m5} hit@10={m10}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m eval.compare_gold_rank <before.json> <after.json>")
    main(sys.argv[1], sys.argv[2])
