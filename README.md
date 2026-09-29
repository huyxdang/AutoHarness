# AutoHarness

**Automatic harness optimization for small open models.** Give it a model and a task with a
grader; it reads the model's failures, rewrites the harness around the model (prompt and agent
loop), and keeps only the changes that hold up on tasks it never studied.

Adaption Labs showed that a better harness can take a 27B open model from 67% to 86% on a legal
agent benchmark with no weight changes, from harness fixes found by hand
([post](https://adaptionlabs.ai/blog/a-better-harness-can-unlock-smaller-models)).
AutoHarness automates that loop and tests it on a public benchmark: on AppWorld's held-out test
split it takes Qwen3.5-9B from 19.6% to 49.4% and Qwen3.5-27B from 35.1% to 72.0%, with no weight changes.
Against a strong off-the-shelf agent (OpenCode), it wins clearly on 9B (49.4% vs. 28.0%) and is
level on 27B (72.0% vs. 76.8%, not significant), where OpenCode uses 1.8× the input tokens.

## Result

On AppWorld's held-out `test_normal` split (168 tasks, one attempt each), same tasks and grader for
every row:

| Model | Harness | pass@1 | Scenarios fully solved | Unit tests passed | Input tokens / task | Steps / task |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | AppWorld ReAct (baseline) | 19.6% (33/168) | 7.1% | 67.4% | 171k | 19.2 |
| Qwen3.5-9B | **AutoHarness** | **49.4% (83/168)** | **25.0%** | **75.2%** | **166k** | 18.5 |
| Qwen3.5-27B | AppWorld ReAct (baseline) | 35.1% (59/168) | 26.8% | 80.4% | 139k | 16.4 |
| Qwen3.5-27B | **AutoHarness** | **72.0% (121/168)** | **55.4%** | **87.3%** | 152k | 16.6 |

- **On this benchmark the harness matters more than model size.** AutoHarness adds 29.8 points on 9B
  (58 tasks solved only with it, 8 only with ReAct; exact McNemar p ≈ 2×10⁻¹⁰) and 36.9 points on 27B
  (70 vs. 8, p ≈ 2×10⁻¹³, 95% CI +28.0 to +45.8). Going from 9B to 27B under the same ReAct harness adds 15.5.
- **A 3× smaller model with an optimized harness beats the bigger model without one:** 9B + AutoHarness
  49.4% vs. 27B + ReAct 35.1% (39 vs. 15 tasks, p = 0.0015), though on scenarios fully solved the two
  are level (25.0% vs. 26.8%). With both optimized, size still counts: 72.0% vs. 49.4% (p ≈ 6×10⁻⁶).
- Each model gets its own harness, optimized separately from the same start (AppWorld's ReAct agent),
  and both runs found the same top fix on their own (below). Input tokens: 3% fewer on 9B, 9% more on 27B.
- **Most of the gain is one rule.** Most gained tasks are ones where ReAct did every action right, then
  gave an answer on a task that expects none (36 of 58 gains on 9B, 52 of 70 on 27B). That rule is part
  of the task spec, so these are real failures, but it is one fix. Counting only the other gains against
  ReAct's wins: 9B 22 vs. 8 (p = 0.016), 27B 18 vs. 8 (p = 0.08, not significant).
- Statistics: `python scripts/compare.py <baseline run> <new run>` (McNemar, bootstrap CIs over tasks
  and over scenarios, answer-only breakdown).

### vs. an off-the-shelf agent harness: OpenCode

[OpenCode](https://github.com/anomalyco/opencode) 1.18.33 with the same models, on the same
168 test tasks:

| Model | Harness | pass@1 | Scenarios fully solved | Unit tests passed | Input tokens / task | Steps / task |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | AppWorld ReAct (baseline) | 19.6% (33/168) | 7.1% | 67.4% | 171k | 19.2 |
| Qwen3.5-9B | OpenCode | 28.0% (47/168) | 8.9% | 61.9% | 446k | 20.3 |
| Qwen3.5-9B | **AutoHarness** | **49.4% (83/168)** | **25.0%** | **75.2%** | **166k** | 18.5 |
| Qwen3.5-27B | AppWorld ReAct (baseline) | 35.1% (59/168) | 26.8% | 80.4% | 139k | 16.4 |
| Qwen3.5-27B | AutoHarness | 72.0% (121/168) | 55.4% | 87.3% | **152k** | 16.6 |
| Qwen3.5-27B | **OpenCode** | **76.8% (129/168)** | **62.5%** | **89.0%** | 268k | 12.6 |

- **On 9B, AutoHarness wins clearly:** 50 tasks solved only by AutoHarness, 14 only by OpenCode
  (+21.4 points, 95% CI +12.5 to +30.4, p ≈ 7×10⁻⁶), with 2.7× fewer input tokens per task.
  OpenCode is ahead of plain ReAct (32 vs. 18), but that difference is not significant (p ≈ 0.07).
- **On 27B, OpenCode is level or slightly ahead:** 26 tasks solved only by OpenCode, 18 only by
  AutoHarness (+4.8 points for OpenCode, 95% CI −3.0 to +12.5, p = 0.29, not significant), while
  sending 1.8× the input tokens per task. Both are far ahead of ReAct (OpenCode vs. ReAct: 80 vs. 10).
- **Why the gap closes with size:** the main failure AutoHarness fixes (every action done, then an
  answer given where none is expected) is covered by a one-line rule in OpenCode's prompt too. The
  9B model does not follow it reliably (29 of OpenCode's 121 failures are this mistake; crediting
  them would put it at 45.2%); the 27B model does (1 of 39 failures). OpenCode is built for frontier
  models: ~11k tokens of system prompt and tool schemas with every request (vs. ~3.6k for
  AutoHarness) and a 64k context. The bigger the model, the more of that general-purpose design it
  can use; a harness tailored to the model matters most for small models.
- OpenCode's own scaling: 9B → 27B is 28.0% → 76.8% (83 vs. 1 tasks), far steeper than ReAct's
  19.6% → 35.1% or AutoHarness's 49.4% → 72.0%.
- For context, the public [AppWorld leaderboard](https://appworld.dev/appworld/leaderboard) lists
  GPT-4o + ReAct at 48.8% and Llama3-70B + ReAct at 20.8% on the same split (2024 entries, so not
  identical conditions).

<details>
<summary>How OpenCode was run (and what an audit of the first run changed)</summary>

OpenCode's own agent loop, planning, and context compaction ran as shipped. The adapter
(`scripts/external_eval.py`) only decides how OpenCode reaches AppWorld:

- **API access:** AppWorld's MCP server exposes each API as a tool, but every task allows all 9 apps
  (~450 APIs, ~84k tokens of tool schemas), too many for any small-context model. AppWorld's API
  predictor (same model, AppWorld's prompt and 3 train-split examples) picks up to 20 APIs, which
  OpenCode gets as direct tools. On top of that, a proxy (`scripts/mcp_filter_proxy.py`) gives it
  `api_docs__show_api_descriptions`, `api_docs__show_api_doc`, and `call_api`, so it can look up and
  call **any** API on demand, like ReAct and AutoHarness do through `apis.api_docs` in code.
- **Prompt:** AppWorld's official function-calling instructions, with the same task-completion rules
  the ReAct baseline gets. Its worked example is not included (it is written as Python code for ReAct).
- **Sampling:** temperature 0, replies capped at 1,500 tokens, as for the other harnesses.
- **Context:** 64k (vs. 32k for ReAct and AutoHarness), because OpenCode's own system prompt and tool
  schemas take ~11k tokens per request. In the first run, 23 tasks sent a request over 32k and
  OpenCode solved 1 of them, so the extra room did not decide the comparison.
- **Safety:** it runs on the local machine, so OpenCode's shell, file-editing, web, and filesystem
  tools are off. A 20-minute wall-clock cap per task applied (never hit in the final runs).
- **27B:** the same adapter, on a separate 64k-context H100 server with the same model revision.

**First run, before the audit:** 22.6% on all 168 tasks (38/168). An audit found two adapter
problems, not OpenCode problems: (1) with only the predicted APIs, half the tasks were missing an
API they needed (162 attempts to call unavailable tools); (2) OpenCode's read-only file tools were
still on, and in 21 tasks the model searched the *host's* files for AppWorld's file-system app,
solving none of them. Fixing both (and pinning temperature 0) moved OpenCode from 22.6% to 28.0% on
all 168 tasks. Tool-call parsing, the task clock, and grading were checked and were correct. (The
final 9B and 27B runs were each completed in parts because of GPU budget; the parts are merged with
`scripts/merge_summaries.py`.)

</details>

The winning change was found automatically from training-set failures. In the 9B run, the optimizer
noticed that 11 of 14 failed tasks were *action* tasks ("send…", "delete…") where the model reported
back what it did (`"Done"`, `"15"`), while the grader expects no answer. It rewrote the
task-completion section of the prompt:

```diff
-- If an answer is needed, e.g., for "How many songs are in the Spotify queue?", call it with the appropriate answer argument value.
-- If no answer is required, e.g., for "Start my Spotify music player.", omit the answer argument (or set it to None/null).
+1. INFORMATION-SEEKING task — it asks you to find out or report something ("How many...", "What is...", "List..."). These expect an answer.
+2. ACTION task — it asks you to DO something (send, pay, delete, follow, create, ...). These expect NO answer.
+   Call `apis.supervisor.complete_task()` with no answer argument.
+   - Do NOT report back what you did here. Passing a confirmation, a summary, a status word, or a count
+     (e.g. "Done", "Sent", "15") will FAIL the task even when every action you performed was correct.
```

The 27B run started over from AppWorld's ReAct prompt and found the same failure in its own training
tasks (8 of 10 failures, e.g. answering "Money sent back to Robert" after paying Robert back), then
wrote its own version of the rule.

## How it works

<p align="center">
  <img src="assets/autoharness-loop.svg" width="100%"
       alt="AutoHarness loop: set current_harness to the baseline; collect training-set trajectories; the optimizer LLM analyzes errors and edits the harness; if the new harness beats current_harness on the dev split it becomes current_harness, otherwise it is discarded; stop after N rounds or 2 rejections in a row; report benchmark results on the test split and output the best harness.">
</p>

- **Solver:** Qwen3.5-9B (one Modal L40S) or Qwen3.5-27B (one H100), served with SGLang, thinking disabled.
- **Optimizer:** Claude Opus 4.8 via headless Claude Code (`claude -p`). It sees only failed *train*
  trajectories and its run's history file (`optimizer/history_<tag>.md`: earlier edits in this run and their
  verdicts; each run starts fresh), and may edit only that model's harness (`harness/` for 9B,
  `harnesses/27b/` for 27B).
- **Keep rule:** dev pass@1 must go up (ties broken by unit-test pass rate), and input tokens may rise at most 20%.
- **Splits:** train (fresh 15 tasks per round) → dev (fixed 20 tasks) → test (168 tasks, touched once).
- **Limits:** the loop runs **N = 3 rounds** (`MAX_ROUNDS`, override with `--rounds`). The parallel
  version also stops early after 2 rounds in a row without a kept edit. Each task gets one attempt
  (pass@1) of at most 50 agent steps, with replies capped at 1,500 tokens and a 32k context.

### What the loop tried

**Qwen3.5-9B**

| Round | Edit proposed from train failures | Dev pass@1 | Verdict |
|---|---|---|---|
| 0 | AppWorld ReAct baseline | 15% | start |
| 1 | Don't return an answer on action tasks | **45%** | kept |
| 2 | Concrete recipe for fetching every page of API results | 45% (fewer unit tests passed) | rejected |
| 3 | Read API fields by exact name; sanity-check filters | 20% | rejected |

Round 3 looked sensible on its training failures but dropped dev accuracy from 45% to 20%. The dev
gate is what stopped it from shipping.

Dev noise check: re-running both harnesses on the same 20 dev tasks gave ReAct 15% → 5% and
AutoHarness 45% → 35%. Single runs move by about ±2 tasks; the ~30-point gap held.

**Qwen3.5-27B**

| Round | Edit proposed from train failures | Dev pass@1 | Verdict |
|---|---|---|---|
| 0 | AppWorld ReAct baseline | 20% | start |
| 1 | Answer only explicit questions; never a confirmation or summary | **55%** | kept |
| 2 | Loop guard in the agent code: warn on an identical repeated code block, stop after 3 repeats | 70% | kept (noise) |
| 3 | Always page through search/list results (APIs return 5 items per call by default) | 80% | kept (within noise) |

The 20-task dev set can't tell rounds 2 and 3 from noise. Round 2's guard never fired on any dev
task, yet 3 tasks flipped to passing: run-to-run variation, not the edit. Round 3 flipped 4 tasks up
and 2 down. The sequential loop has no noise margin, so it kept both (`optimizer/loop_parallel.py`
requires a 2-task margin and re-scores the current best in the same batch). The guard only acts
when the model repeats itself, so it is harmless either way. The 72.0% test score is for all three
edits together, and the final dev score (80%) overstates it, as expected when keeping the best of
noisy dev runs.

## What you get: the export

`scripts/export.py` packages everything a user of the optimized harness needs into one folder per
model ([`export/appworld-qwen3.5-9b/`](export/appworld-qwen3.5-9b),
[`export/appworld-qwen3.5-27b/`](export/appworld-qwen3.5-27b)):

| File | What it's for |
|---|---|
| `harness/prompt.txt`, `harness/react_agent.py` | The optimized harness: drop-in replacement for the agent you run today |
| `config.yaml` | The exact model and sampling settings it was validated with |
| `report.md` | Baseline vs. optimized on held-out tasks: accuracy, tokens, steps |
| `CHANGELOG.md` | Every edit that was tried, the failures behind it, and why it was kept or rejected |

The benchmark splits and grader double as a regression test when the model changes, and the
failures the harness could not fix are the natural targets for post-training.

## Repo layout

```
serve/sglang_server.py      SGLang on Modal (one app per solver model, API-key protected)
scripts/run_eval.py         run an agent on tasks → pass@1, unit tests, tokens, steps, time
scripts/modal_eval.py       run each AppWorld task in its own Modal CPU container
scripts/final_eval.sh       dev noise check + test_normal comparison
scripts/external_eval.py    run an external harness (OpenCode) on AppWorld tasks, graded the same way
scripts/mcp_filter_proxy.py MCP proxy: predicted APIs as tools + on-demand api_docs / call_api
scripts/merge_summaries.py  combine partial runs of one experiment into one summary
scripts/compare.py          paired comparison of two runs: McNemar, bootstrap CIs, answer-only breakdown
scripts/run_27b.sh          the 27B experiment end to end: ReAct test, loop, final test (resumable)
scripts/export.py           package the optimized harness + report + change log
harness/                    the 9B harness being optimized (the only thing its optimizer may edit)
harnesses/27b/              the 27B harness (its own copy, started from AppWorld's ReAct agent)
optimizer/round.py          one round: train run → claude -p edits harness/ → commit
optimizer/loop.py           rounds with automatic keep/reject on dev
optimizer/candidates.py     parallel version: one diagnosis call, k editors, k candidates
optimizer/loop_parallel.py  scores current best + k candidates together, keeps winner by a margin
optimizer/history_*.md      every edit and its verdict, one file per run (9b, 27b)
results/                    summary.json for every run
```

## Reproduce

Requires Python 3.11, a [Modal](https://modal.com) account, and Claude Code.

```bash
# 1. AppWorld from source (the agents package needs it); then the data
git clone https://github.com/StonyBrookNLP/appworld && cd appworld
pip install -e . -e "experiments[simplified]" && appworld install --repo && appworld download data && cd ..

# 2. Model server (creates an OpenAI-compatible endpoint on Modal)
modal secret create autoharness-sglang SGLANG_API_KEY=<random-key>   # also put it in .env
SOLVER=qwen3.5-9b modal deploy serve/sglang_server.py

# 3. Optimize, then evaluate on test
python optimizer/loop.py --rounds 3 --model qwen3.5-9b --tag 9b_
bash scripts/final_eval.sh
python scripts/export.py --name appworld-qwen3.5-9b --baseline results/9b_test_react --optimized results/9b_test_auto

# Qwen3.5-27B on an H100: ReAct test, loop, and final test in one resumable script
SOLVER=qwen3.5-27b modal deploy serve/sglang_server.py
bash scripts/run_27b.sh
python scripts/compare.py 27b_test_react 27b_test_auto
```

Setup notes that cost time (AppWorld's Git LFS quota, SGLang vs. AppWorld response format,
runaway generations, AppWorld's frozen clock stalling Modal's container heartbeat) are in [PLAN.md](PLAN.md).

## Limitations

- One benchmark (AppWorld) and two solvers from one family (Qwen3.5-9B and 27B), 3 optimization
  rounds each. Both times the big win was one fix found in round 1; later rounds added nothing the
  dev set could measure, so it is not yet shown whether gains keep compounding.
- The dev set has 20 tasks, so each is worth 5 points and single runs move by about ±2–3 tasks. The
  sequential loop has no noise margin: in the 27B run it kept an edit that never fired on dev. The
  test split (168 tasks) is the number to trust.
- Test results are one run per harness. Sampling is temperature 0, but batched inference is not
  perfectly deterministic.
- Time per task is not compared for 27B: its two test runs ran under different GPU load.
- Leaderboard comparisons are context only: those entries used AppWorld's 2024 setup.
- AutoHarness does not beat a strong off-the-shelf harness at every size: on 27B, OpenCode scores
  4.8 points higher (not significant) at 1.8× the tokens. The loop started from AppWorld's ReAct
  agent, not from OpenCode; starting it from a stronger harness is untested.
- The OpenCode adapter is ours: API access goes through AppWorld's predictor plus on-demand lookup,
  its prompt lacks the worked example the ReAct-style harnesses get, and it runs with a 64k context
  (vs. 32k). A different adapter could score differently. Prime
  Agent was not compared: its Python REPL runs unsandboxed on the host, and it was left out of scope.
- The harness is prompt + agent loop only. Tool wrappers, memory, and context compaction are not
  yet in the search space (two dev tasks still overflowed the 32k context).

## Where this goes

The product version starts from a customer's task description and a few examples instead of a
public benchmark: generate a preview set of tasks with graders, have the customer approve it,
generate the full train/dev/test benchmark (for example with Adaption's Adaptive Data), then run
this loop and return the export. The failures the harness can't fix become the data for
post-training.
