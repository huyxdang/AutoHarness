"""Compare two finished runs on the same AppWorld tasks.

    python scripts/compare.py 27b_test_react 27b_test_auto

Prints pass@1, scenario goal completion (SGC), unit tests, tokens and steps for both runs, then
the paired comparison: tasks solved by only one run, the exact McNemar test, 95% bootstrap CIs for
the pass@1 difference (resampling tasks, and resampling whole scenarios since a scenario's 3 task
variants are correlated), and how many failures are "answer-only" (every check passed except the
final answer check).
"""

import argparse
import json
import random
import re
from collections import defaultdict
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "appworld/experiments/outputs"


def load(experiment: str) -> dict[str, dict]:
    summary = json.loads((PROJECT_ROOT / "results" / experiment / "summary.json").read_text())
    return {row["task_id"]: row for row in summary["tasks"]}


def scenario(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]


def failed_checks(experiment: str, task_id: str) -> list[str]:
    report = (OUTPUTS / experiment / "tasks" / task_id / "evaluation/report.md").read_text()
    fails = report.split("── Fails ──", 1)[1] if "── Fails ──" in report else ""
    return [m.group(1).strip() for m in re.finditer(r">> Failed Requirement\n(.+)", fails)]


def answer_only(experiment: str, task_id: str) -> bool:
    return failed_checks(experiment, task_id) == ["assert answers match."]


def sgc(rows: dict[str, dict]) -> float:
    by_scenario = defaultdict(list)
    for task_id, row in rows.items():
        by_scenario[scenario(task_id)].append(row["passed"])
    return 100 * sum(all(v) for v in by_scenario.values()) / len(by_scenario)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)


def bootstrap_ci(diffs: dict[str, int], by_scenario: bool, n: int = 10000, seed: int = 0):
    groups = defaultdict(list)
    for task_id, d in diffs.items():
        groups[scenario(task_id) if by_scenario else task_id].append(d)
    units = list(groups.values())
    rng = random.Random(seed)
    stats = sorted(100 * sum(sum(u) for u in s) / sum(len(u) for u in s)
                   for s in ([rng.choice(units) for _ in units] for _ in range(n)))
    return stats[int(0.025 * n)], stats[int(0.975 * n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("new")
    args = parser.parse_args()
    base, new = load(args.base), load(args.new)
    assert set(base) == set(new), "runs cover different tasks"
    n = len(base)

    print(f"{'':28}{'pass@1':>16}{'SGC':>8}{'tests':>8}{'input tok':>11}{'steps':>7}")
    for name, rows in [(args.base, base), (args.new, new)]:
        passed = sum(r["passed"] for r in rows.values())
        tests = 100 * sum(r["tests_passed"] for r in rows.values()) / sum(r["tests_total"] for r in rows.values())
        print(f"{name:28}{100 * passed / n:>7.1f}% ({passed:>3}/{n}){sgc(rows):>7.1f}%{tests:>7.1f}%"
              f"{sum(r['input_tokens'] for r in rows.values()) / n:>11,.0f}"
              f"{sum(r['steps'] for r in rows.values()) / n:>7.1f}")

    only_new = [t for t in base if new[t]["passed"] and not base[t]["passed"]]
    only_base = [t for t in base if base[t]["passed"] and not new[t]["passed"]]
    diffs = {t: int(new[t]["passed"]) - int(base[t]["passed"]) for t in base}
    delta = 100 * sum(diffs.values()) / n
    lo_t, hi_t = bootstrap_ci(diffs, by_scenario=False)
    lo_s, hi_s = bootstrap_ci(diffs, by_scenario=True)
    print(f"\n{args.new} - {args.base}: {delta:+.1f} points")
    print(f"  solved only by {args.new}: {len(only_new)}, only by {args.base}: {len(only_base)}, "
          f"exact McNemar p = {mcnemar_exact(len(only_new), len(only_base)):.2g}")
    print(f"  95% CI: {lo_t:+.1f} to {hi_t:+.1f} (tasks resampled), {lo_s:+.1f} to {hi_s:+.1f} (scenarios resampled)")

    for name, rows in [(args.base, base), (args.new, new)]:
        failed = [t for t in rows if not rows[t]["passed"]]
        print(f"  {name}: {sum(answer_only(name, t) for t in failed)} of {len(failed)} failures are answer-only")
    print(f"  gains that were answer-only failures in {args.base}: "
          f"{sum(answer_only(args.base, t) for t in only_new)} of {len(only_new)}")


if __name__ == "__main__":
    main()
