# Harness optimization history

Each round: an edit proposed by the optimizer from train-set failures, then kept or rejected
based on a fixed 20-task dev set (pass@1, tiebreak: unit-test pass rate; input tokens may rise at most 20%).

## qwen3.5-9b — round 0 (starting harness)
Dev: pass@1 15.0%, tests 59.6%, 154,523 input tok/task, 20.4 steps/task

### Round 1 — KEPT (better on dev and within token budget)
- Edit: 11 of 14 round-1 train failures (6 of them solely) failed the `assert answers match (== null)` requirement by passing a confirmation/summary/count string to `complete_task` on action tasks; I rewrote section D of prompt.txt into an explicit information-seeking-vs-action decision rule that mandates `complete_task()` with no answer (and forbids status/summary/count values) for action tasks.
- Dev after edit: pass@1 45.0%, tests 78.9%, 150,510 input tok/task, 17.0 steps/task
- Best dev so far: pass@1 45.0%, tests 78.9%, 150,510 input tok/task, 17.0 steps/task
