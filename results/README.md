# Results

One folder per run, each with `summary.json` (per-task pass/fail, unit tests, tokens, steps, time)
and `task_ids.txt`. Run names are `<model>_<stage>_<split>`.

## Runs behind the README numbers

| Number in the README | Run |
|---|---|
| 9B · ReAct baseline, test | `9b_test_react` |
| 9B · OpenCode, test | `9b_test_opencode_v2` (the audited rerun; the first run is in `archive/`) |
| 9B · AutoHarness, test | `9b_test_auto` |
| 27B · ReAct baseline, test | `27b_test_react` |
| 27B · OpenCode, test | `27b_test_opencode` (three budget-limited parts merged with `scripts/merge_summaries.py`) |
| 27B · AutoHarness, test | `27b_test_auto` |
| Dev pass@1 per round (timeline chart) | `9b_round{0..3}_dev`, `27b_round{0..3}_dev` |
| Train failures the optimizer read | `9b_round{1..3}_train`, `27b_round{1..3}_train` |
| Dev noise check (9B) | `9b_noise_react_dev`, `9b_noise_auto_dev` |

Task lists: `test_normal_task_ids.txt` (all 168 test tasks) and `dev20_task_ids.txt` (the fixed dev set).

Compare any two runs on the same tasks:

```bash
python scripts/compare.py 27b_test_react 27b_test_auto
```

`archive/` holds runs that no reported number depends on: the MVP, pilots, the pre-audit 9B
OpenCode run, the parts of the merged 27B OpenCode run, and a 9B round-4 train run that is not part
of the reported three-round loop.
