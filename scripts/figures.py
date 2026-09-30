"""Render the README's charts as static SVGs from the run summaries in results/.

    python scripts/figures.py        # writes assets/results.svg, timeline.svg, architecture.svg

Each chart sits on its own white card so it stays readable in GitHub's dark theme. Fonts are the
system UI stack (an <img> SVG on GitHub cannot load web fonts). Colours: one saturated blue for
AutoHarness, a light tint of it for the frontier general-purpose harness (OpenCode), grey for the
baseline (AppWorld's ReAct agent).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
INK, MUTED, FAINT, GRID = "#1f2328", "#59636e", "#8c959f", "#e6e8eb"
BLUE, TINT, GREY = "#2f6ff0", "#bcd2fb", "#e1e3e6"

MODELS = [("Qwen3.5-9B", "9b"), ("Qwen3.5-27B", "27b")]
HARNESSES = [  # label, run suffix, bar colour
    ("Baseline (ReAct)", "test_react", GREY),
    ("OpenCode", "test_opencode", TINT),
    ("AutoHarness", "test_auto", BLUE),
]
# The reported 9B OpenCode run is the audited rerun (see README, "How OpenCode was run").
RUN_OVERRIDES = {("9b", "test_opencode"): "9b_test_opencode_v2"}


def summary(run: str) -> dict:
    return json.loads((RESULTS / run / "summary.json").read_text())


def card(width: int, height: int, body: str, title: str, subtitle: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="{FONT}">'
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="16" fill="#ffffff" stroke="#d8dee4"/>'
        f'<text x="32" y="46" font-size="20" font-weight="600" fill="{INK}">{title}</text>'
        f'<text x="32" y="72" font-size="14" fill="{MUTED}">{subtitle}</text>'
        f"{body}</svg>\n"
    )


def bar_panel(x0: int, y0: int, w: int, h: int, heading: str, note: str, rows, ticks, fmt, tick_fmt) -> str:
    """One column chart: a model group of three bars per model, value printed on top."""
    top, base = y0 + 40, y0 + h - 58
    scale = lambda v: base - (base - top) * v / ticks[-1]
    out = [f'<text x="{x0}" y="{y0 + 10}" font-size="15" font-weight="600" fill="{INK}">{heading}'
           f'<tspan font-weight="400" fill="{FAINT}" dx="8">{note}</tspan></text>']
    plot_x0 = x0 + 44
    for t in ticks:
        y = scale(t)
        dash = "" if t == 0 else ' stroke-dasharray="4 5"'
        colour = "#c4cad1" if t == 0 else GRID
        out.append(f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" stroke="{colour}"{dash}/>')
        out.append(f'<text x="{plot_x0 - 10}" y="{y + 4:.1f}" font-size="12" fill="{FAINT}" text-anchor="end">{tick_fmt(t)}</text>')
    bar, gap, between = 46, 22, 36
    group = 3 * bar + 2 * gap
    gx = plot_x0 + (x0 + w - plot_x0 - (2 * group + between)) / 2
    for model, values in rows:
        for i, ((label, _, colour), v) in enumerate(zip(HARNESSES, values)):
            bx = gx + i * (bar + gap)
            by = scale(v)
            auto = label == "AutoHarness"
            out.append(f'<path d="M{bx:.1f} {base:.1f}V{by + 4:.1f}Q{bx:.1f} {by:.1f} {bx + 4:.1f} {by:.1f}'
                       f'H{bx + bar - 4:.1f}Q{bx + bar:.1f} {by:.1f} {bx + bar:.1f} {by + 4:.1f}V{base:.1f}Z" fill="{colour}"/>')
            weight, fill = ("700", BLUE) if auto else ("500", "#3d444d")
            out.append(f'<text x="{bx + bar / 2:.1f}" y="{by - 7:.1f}" font-size="13" font-weight="{weight}" '
                       f'fill="{fill}" text-anchor="middle">{fmt(v)}</text>')
            short = {"Baseline (ReAct)": "ReAct", "OpenCode": "OpenCode", "AutoHarness": "AutoHarness"}[label]
            out.append(f'<text x="{bx + bar / 2:.1f}" y="{base + 18:.1f}" font-size="11.5" '
                       f'fill="{BLUE if auto else MUTED}" font-weight="{600 if auto else 400}" text-anchor="middle">{short}</text>')
        out.append(f'<text x="{gx + group / 2:.1f}" y="{base + 42:.1f}" font-size="13.5" font-weight="600" '
                   f'fill="{INK}" text-anchor="middle">{model}</text>')
        gx += group + between
    return "".join(out)


def results_svg() -> str:
    solved, tokens = [], []
    for model, tag in MODELS:
        runs = [summary(RUN_OVERRIDES.get((tag, suffix), f"{tag}_{suffix}")) for _, suffix, _ in HARNESSES]
        solved.append((model, [r["pass_at_1"] for r in runs]))
        tokens.append((model, [sum(t["input_tokens"] for t in r["tasks"]) / 1e6 for r in runs]))
    W, H = 1000, 470
    body = bar_panel(32, 112, 450, 330, "Tasks solved (pass@1)", "higher is better ↑", solved,
                     [0, 25, 50, 75, 100], lambda v: f"{v:.1f}%", lambda t: f"{t}%")
    body += f'<line x1="500" y1="112" x2="500" y2="{H - 30}" stroke="{GRID}"/>'
    body += bar_panel(518, 112, 450, 330, "Total input tokens", "lower is better ↓", tokens,
                      [0, 20, 40, 60, 80], lambda v: f"{v:.1f}M", lambda t: f"{t}M")
    return card(W, H, body, "AutoHarness doubles pass@1 without the token bill",
                "AppWorld test_normal, 168 held-out tasks, one attempt each. Only the harness differs within a model.")


def timeline_svg() -> str:
    # (dev pass@1 per round, verdict per round); round 0 is the starting ReAct harness.
    verdicts = {"9b": ["start", "kept", "rejected", "rejected"], "27b": ["start", "kept", "kept", "kept"]}
    colours = {"9b": "#8c959f", "27b": BLUE}
    W, H = 1000, 400
    x0, x1, top, base = 110, 760, 120, 330
    xs = lambda r: x0 + (x1 - x0) * r / 3
    ys = lambda v: base - (base - top) * v / 100
    out = []
    for t in (0, 25, 50, 75, 100):
        dash = "" if t == 0 else ' stroke-dasharray="4 5"'
        out.append(f'<line x1="{x0 - 20}" y1="{ys(t)}" x2="{x1 + 20}" y2="{ys(t)}" stroke="{"#c4cad1" if t == 0 else GRID}"{dash}/>')
        out.append(f'<text x="{x0 - 32}" y="{ys(t) + 4}" font-size="12" fill="{FAINT}" text-anchor="end">{t}%</text>')
    for r in range(4):
        out.append(f'<text x="{xs(r)}" y="{base + 24}" font-size="12.5" fill="{MUTED}" text-anchor="middle">'
                   f'{"Start" if r == 0 else f"Round {r}"}</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{base + 50}" font-size="12.5" fill="{FAINT}" text-anchor="middle">'
               f'Dev pass@1 of each round&#8217;s candidate harness (20 fixed dev tasks)</text>')
    for model, tag in MODELS:
        scores = [summary(f"{tag}_round{r}_dev")["pass_at_1"] for r in range(4)]
        c = colours[tag]
        # The line follows the harness actually kept after each round; rejected candidates hang off it.
        kept, best = [], scores[0]
        for s, v in zip(scores, verdicts[tag]):
            best = s if v in ("start", "kept") else best
            kept.append(best)
        out.append('<polyline fill="none" stroke="{}" stroke-width="2.5" stroke-linejoin="round" points="{}"/>'.format(
            c, " ".join(f"{xs(r)},{ys(v)}" for r, v in enumerate(kept))))
        for r, (s, v) in enumerate(zip(scores, verdicts[tag])):
            if v == "rejected":
                out.append(f'<line x1="{xs(r)}" y1="{ys(kept[r])}" x2="{xs(r)}" y2="{ys(s)}" stroke="{c}" stroke-dasharray="3 4"/>')
                out.append(f'<circle cx="{xs(r)}" cy="{ys(s)}" r="6" fill="#ffffff" stroke="{c}" stroke-width="2"/>')
                out.append(f'<text x="{xs(r)}" y="{ys(s) + 24}" font-size="12" fill="{MUTED}" '
                           f'text-anchor="middle">{s:.0f}% · rejected</text>')
            else:
                # 27B labels sit above its line, 9B labels below its line, so the two never collide.
                dy = -12 if tag == "27b" else 24
                out.append(f'<circle cx="{xs(r)}" cy="{ys(s)}" r="6" fill="{c}"/>')
                out.append(f'<text x="{xs(r)}" y="{ys(s) + dy}" font-size="12.5" font-weight="600" fill="{c}" '
                           f'text-anchor="middle">{s:.0f}%</text>')
        out.append(f'<text x="{x1 + 34}" y="{ys(kept[-1]) + 5}" font-size="14" font-weight="600" fill="{c}">{model}</text>')
    # Legend and the one-line reading of the chart.
    lx = 780
    out.append(f'<circle cx="{lx}" cy="40" r="6" fill="{INK}"/><text x="{lx + 14}" y="45" font-size="13" fill="{MUTED}">edit kept</text>')
    out.append(f'<circle cx="{lx}" cy="66" r="6" fill="#fff" stroke="{INK}" stroke-width="2"/>'
               f'<text x="{lx + 14}" y="71" font-size="13" fill="{MUTED}">edit rejected by the dev gate</text>')
    return card(W, H, "".join(out), "Round 1 finds the big fix on both models",
                "Every edit is scored on held-out dev tasks and kept only if it improves on the current harness.")


MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def box(x, y, w, h, title, lines, stroke="#8c959f", fill="#f6f8fa", title_fill=INK) -> str:
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
           f'<text x="{x + 16}" y="{y + 28}" font-size="14.5" font-weight="600" fill="{title_fill}">{title}</text>']
    for i, (text, mono) in enumerate(lines):
        font = f' font-family="{MONO}" font-size="12"' if mono else ' font-size="12.5"'
        out.append(f'<text x="{x + 16}" y="{y + 52 + i * 19}"{font} fill="{MUTED}">{text}</text>')
    return "".join(out)


def arrow(points, label="", lx=0, ly=0, anchor="middle") -> str:
    path = "M" + " L".join(f"{x} {y}" for x, y in points)
    out = f'<path d="{path}" fill="none" stroke="#57606a" stroke-width="1.6" marker-end="url(#arr)"/>'
    if label:
        out += f'<text x="{lx}" y="{ly}" font-size="12" fill="{MUTED}" text-anchor="{anchor}">{label}</text>'
    return out


def architecture_svg() -> str:
    """Who runs where: the optimizer edits the harness; each AppWorld task runs in its own Modal CPU
    container against a SGLang server on a Modal GPU; results decide keep or revert."""
    W, H = 1000, 470
    defs = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke="#57606a" stroke-width="1.6"/></marker></defs>')
    top, bh = 110, 132
    body = [defs,
            box(32, top, 200, bh, "Optimizer", [("Claude Code, headless", False), ("claude -p", True),
                                                 ("reads failed train runs", False), ("+ its edit history", False)],
                stroke="#d9695f", fill="#fdf0ee"),
            box(282, top, 200, bh, "Harness", [("prompt.txt", True), ("react_agent.py", True),
                                               ("the only files the", False), ("optimizer may edit", False)],
                stroke=BLUE, fill="#eef4ff"),
            box(532, top, 210, bh, "AppWorld tasks", [("Modal CPU containers,", False), ("one per task", False),
                                                      ("agent loop + app APIs", False), ("graded by unit tests", False)]),
            box(792, top, 176, bh, "Solver model", [("SGLang on Modal GPU", False), ("Qwen3.5-9B · L40S", False),
                                                    ("Qwen3.5-27B · H100", False), ("OpenAI-compatible API", False)]),
            box(282, 322, 200, 96, "Dev gate", [("loop.py: keep only if", False), ("dev pass@1 improves", False)],
                stroke=BLUE, fill="#ffffff"),
            box(532, 322, 210, 96, "Run results", [("results/*/summary.json", True),
                                                   ("pass@1, tests, tokens", False)]),
            arrow([(232, 176), (278, 176)], "edits", 255, 168),
            arrow([(482, 176), (528, 176)], "runs", 505, 168),
            arrow([(742, 164), (788, 164)]),
            arrow([(788, 190), (742, 190)]),
            f'<text x="765" y="156" font-size="11" fill="{MUTED}" text-anchor="middle">chat</text>',
            arrow([(637, top + bh), (637, 318)], "trajectories", 646, 290, "start"),
            arrow([(532, 370), (486, 370)], "dev", 509, 362),
            arrow([(382, 322), (382, top + bh + 4)], "keep / revert", 391, 290, "start"),
            arrow([(637, 418), (637, 446), (132, 446), (132, top + bh + 4)]),
            f'<text x="384" y="440" font-size="12" fill="{MUTED}" text-anchor="middle">failed train trajectories</text>',
            f'<text x="832" y="360" font-size="12.5" fill="{MUTED}">After the last round:</text>',
            f'<text x="832" y="380" font-family="{MONO}" font-size="12" fill="{INK}">scripts/export.py</text>',
            f'<text x="832" y="400" font-size="12.5" fill="{MUTED}">harness + config +</text>',
            f'<text x="832" y="418" font-size="12.5" fill="{MUTED}">report + changelog</text>']
    return card(W, H, "".join(body), "How a round runs",
                "Train tasks show the optimizer what failed; dev tasks decide whether its edit is kept. Test tasks are run once at the end.")


def main() -> None:
    (ASSETS / "results.svg").write_text(results_svg())
    (ASSETS / "timeline.svg").write_text(timeline_svg())
    (ASSETS / "architecture.svg").write_text(architecture_svg())
    print("wrote assets/results.svg, assets/timeline.svg, assets/architecture.svg")


if __name__ == "__main__":
    main()
