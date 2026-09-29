"""Merge partial run summaries of the same experiment (e.g. a budget-capped run plus its completion)
into one summary.json, recomputing the aggregates over all tasks.

    python scripts/merge_summaries.py results/9b_test_opencode_v2 \
        summary_part1_99.json summary_part2_69.json
"""

import json
import sys
from pathlib import Path


def main(results_dir: Path, parts: list[str]) -> None:
    summaries = [json.loads((results_dir / p).read_text()) for p in parts]
    rows = [row for s in summaries for row in s["tasks"]]
    ids = [r["task_id"] for r in rows]
    assert len(ids) == len(set(ids)), "a task appears in more than one part"
    n = len(rows)
    merged = {k: v for k, v in summaries[0].items() if k not in ("tasks", "skipped_by_budget_guard")}
    merged.update({
        "num_tasks": n,
        "parts": parts,
        "pass_at_1": round(100 * sum(r["passed"] for r in rows) / n, 1),
        "test_pass_rate": round(100 * sum(r["tests_passed"] for r in rows)
                                / max(1, sum(r["tests_total"] for r in rows)), 1),
        "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / n),
        "avg_output_tokens": round(sum(r["output_tokens"] for r in rows) / n),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 1),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
        "timeouts": sum(r["timeout"] for r in rows),
        "wall_seconds": sum(s["wall_seconds"] for s in summaries),
        "tasks": rows,
    })
    (results_dir / "summary.json").write_text(json.dumps(merged, indent=2))
    (results_dir / "task_ids.txt").write_text("\n".join(ids) + "\n")
    print(f"merged {n} tasks: pass@1 {merged['pass_at_1']}%, tests {merged['test_pass_rate']}%, "
          f"{merged['avg_input_tokens']:,} input tok/task, timeouts {merged['timeouts']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
