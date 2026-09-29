
## qwen3.5-27b — round 0 (starting harness)
Dev: pass@1 20.0%, tests 74.6%, 92,739 input tok/task, 13.6 steps/task

### Round 1 — KEPT (better on dev and within token budget)
- Edit: 8 of 10 failed tasks failed the `assert answers match … == 'null'` test because the agent passed a spurious `answer` (often a status summary like "Money sent back to Robert") to `complete_task` on action tasks whose expected answer is null — with 6 of those failing on that test alone — so I rewrote section D of the prompt to make the agent decide answer-vs-no-answer up front, provide `answer` only for explicit questions, and never pass a confirmation/summary/count as the answer on action/state-change tasks.
- Dev after edit: pass@1 55.0%, tests 81.6%, 107,650 input tok/task, 13.8 steps/task
- Best dev so far: pass@1 55.0%, tests 81.6%, 107,650 input tok/task, 13.8 steps/task

### Round 2 — KEPT (better on dev and within token budget)
- Edit: 1 of the 5 failed train tasks (d0b1f43_1) showed the repeated-failing-call pattern — it re-ran a byte-identical failing code block 11 times to the 50-step cap, burning 827k input tokens (5× average) and never calling complete_task — so I added a loop guard in react_agent.py that detects consecutive identical code submissions, injects a course-correction notice urging a different approach, and terminates the episode after 3 repeats.
- Dev after edit: pass@1 70.0%, tests 88.6%, 100,576 input tok/task, 14.1 steps/task
- Best dev so far: pass@1 70.0%, tests 88.6%, 100,576 input tok/task, 14.1 steps/task

### Round 3 — KEPT (better on dev and within token budget)
- Edit: 1 of the 3 failed train tasks (afc0fce_2) failed because the agent called a search/list API (`search_friends`) once and relied on the default `page_limit=5`, silently truncating the friends list and producing the wrong set of records to modify — a latent bug in the broader completeness/pagination failure class — so I rewrote the pagination bullet in section B to forcefully state that every search/list API (friends/contacts, users, transactions, library, notes, etc.) defaults to only 5 items per call and must always be fully paginated with `page_limit=20` looping `page_index` until an empty page before counting, filtering, or acting.
- Dev after edit: pass@1 80.0%, tests 91.2%, 107,747 input tok/task, 14.1 steps/task
- Best dev so far: pass@1 80.0%, tests 91.2%, 107,747 input tok/task, 14.1 steps/task

### Post-run analysis (added by hand after the run; not seen by the optimizer)
- Round 2's loop guard never fired on any dev task (its notice appears in no dev trajectory), so its
  +3-task dev gain was run-to-run noise, not the edit. The sequential loop has no noise margin, so it kept it.
  The guard is inert unless the model repeats itself, so it is harmless.
- Round 3 flipped 4 dev tasks to pass and 2 to fail: a net +2, within the noise seen in round 2.
- Test (168 tasks, once): ReAct 35.1% -> final harness (all three edits) 72.0%; 52 of the 70 gains were
  answer-only fixes (round 1's rule).
