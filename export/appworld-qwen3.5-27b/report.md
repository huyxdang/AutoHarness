# Results: appworld-qwen3.5-27b

Held-out evaluation on AppWorld `test_normal` (168 tasks, one attempt each).
Same model and sampling settings for both rows; only the harness differs.

| Harness | pass@1 | Unit tests passed | Input tokens / task | Output tokens / task | Steps / task |
|---|---|---|---|---|---|
| Baseline (AppWorld ReAct) | **35.1%** (59/168) | 80.4% | 139,459 | 1,513 | 16.4 |
| Optimized (this harness) | **72.0%** (121/168) | 87.3% | 152,063 | 1,657 | 16.6 |

Task by task: 70 tasks solved only with the optimized harness, 8 only with the baseline.

See CHANGELOG.md for every harness edit that was tried and why it was kept or rejected.
