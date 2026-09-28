# Results: appworld-qwen3.5-9b

Held-out evaluation on AppWorld `test_normal` (168 tasks, one attempt each).
Same model and sampling settings for both rows; only the harness differs.

| Harness | pass@1 | Unit tests passed | Input tokens / task | Output tokens / task | Steps / task |
|---|---|---|---|---|---|
| Baseline (AppWorld ReAct) | **19.6%** (33/168) | 67.4% | 171,269 | 2,414 | 19.2 |
| Optimized (this harness) | **49.4%** (83/168) | 75.2% | 166,088 | 2,155 | 18.5 |

Task by task: 58 tasks solved only with the optimized harness, 8 only with the baseline.

See CHANGELOG.md for every harness edit that was tried and why it was kept or rejected.
