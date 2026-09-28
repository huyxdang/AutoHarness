# Harness optimization history

Each round: an edit proposed by the optimizer from train-set failures, then kept or rejected
based on a fixed 20-task dev set (pass@1, tiebreak: unit-test pass rate; input tokens may rise at most 20%).

## qwen3.5-9b — round 0 (starting harness)
Dev: pass@1 15.0%, tests 59.6%, 154,523 input tok/task, 20.4 steps/task

### Round 1 — KEPT (better on dev and within token budget)
- Edit: 11 of 14 round-1 train failures (6 of them solely) failed the `assert answers match (== null)` requirement by passing a confirmation/summary/count string to `complete_task` on action tasks; I rewrote section D of prompt.txt into an explicit information-seeking-vs-action decision rule that mandates `complete_task()` with no answer (and forbids status/summary/count values) for action tasks.
- Dev after edit: pass@1 45.0%, tests 78.9%, 150,510 input tok/task, 17.0 steps/task
- Best dev so far: pass@1 45.0%, tests 78.9%, 150,510 input tok/task, 17.0 steps/task

### Round 2 — REJECTED (no improvement on dev)
- Edit: Pagination under-fetch directly caused 2 of 9 round-2 train failures (229360a_2 broke its remove/keep set via `if len(page) < 20: break` with default page_limit=5; e85d92a_2 read only the first `search_songs` page) and left the 6104387 CSV tasks incomplete, so I replaced the vague "process all results" bullet with a concrete recipe (pass `page_limit=20`, loop from page_index=0, stop only on an empty page, never break on a guessed size like `len(page)<20` or `while page_index<10`, and apply it to search APIs too) and aligned the few-shot demo's loop to that pattern.
- Dev after edit: pass@1 45.0%, tests 75.4%, 140,843 input tok/task, 16.9 steps/task
- Best dev so far: pass@1 45.0%, tests 78.9%, 150,510 input tok/task, 17.0 steps/task

### Round 3 — REJECTED (no improvement on dev)
- Edit: The top harness-fixable pattern (2 of 4 round-3 failures — 229360a_3 and afc0fce_2) was the agent building a filter on a wrong response-field assumption that silently produced a wrong-but-non-erroring result (`.get("song_ids", [])`→`[]`→vacuously-true `all()`; `"friends_since" in profile` always true), so I added three section-C bullets telling it to read values through the exact schema key (not guessed keys with `.get(default)`), test a field's value rather than its mere presence, and sanity-check any filter that keeps all-or-zero items by printing a kept and a dropped record before acting.
- Dev after edit: pass@1 20.0%, tests 67.5%, 149,303 input tok/task, 17.1 steps/task
- Best dev so far: pass@1 45.0%, tests 78.9%, 150,510 input tok/task, 17.0 steps/task
