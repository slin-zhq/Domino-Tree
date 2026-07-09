#!/usr/bin/env python3
"""Replay one or more recorded casts side by side on a shared wall clock.

Each pane streams its decoded text at the pace it was actually measured at
(``record.py``), so faster methods visibly pull ahead and finish first --- the
same record-then-replay design as the released DFlash/Domino demo. Per-pane
badges show live TPS, token count, round count, mean accepted length (tau), and
speedup over the baseline pane; a global timeline runs along the bottom.

    python play.py runs/ar.json runs/domino.json runs/dominotree.json

Only depends on ``rich`` (no torch), so it runs anywhere the casts are copied.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("casts", nargs="+", help="Cast JSON files (one per pane, left to right).")
    p.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier (1.0 = real time).")
    p.add_argument("--fps", type=float, default=20.0, help="Render frames per second.")
    p.add_argument("--baseline", default=None,
                   help="Method name to normalize speedup against (default: 'ar' if present, else the slowest).")
    p.add_argument("--hold", type=float, default=2.5, help="Seconds to hold the final frame.")
    return p.parse_args()


def load_casts(paths):
    casts = []
    for path in paths:
        with open(path) as f:
            c = json.load(f)
        c["_events"] = c.get("events", [])
        casts.append(c)
    return casts


def pane_state(cast, vt):
    """State of one pane at virtual decode time ``vt`` (s)."""
    events = cast["_events"]
    decode_time = cast["summary"]["decode_time_s"]
    shown = [e for e in events if e["t"] <= vt]
    if shown:
        cum = shown[-1]["cum_tokens"]
        rnd = shown[-1]["round"]
        text = "".join(e["chunk"] for e in shown)
    else:
        cum, rnd, text = 0, 0, ""
    done = vt >= decode_time or (events and shown and shown[-1] is events[-1])
    elapsed = min(vt, decode_time) if not done else decode_time
    tps = cum / elapsed if elapsed > 1e-9 else 0.0
    tau = cum / rnd if rnd else 0.0
    frac = min(1.0, cum / max(1, cast["summary"]["num_output_tokens"]))
    return {"text": text, "cum": cum, "round": rnd, "tps": tps, "tau": tau, "done": done, "frac": frac}


def bar(frac, width, color):
    filled = int(round(frac * width))
    return Text("█" * filled, style=color) + Text("─" * (width - filled), style="grey37")


def tail_wrapped(text, width, height):
    """Last ``height`` display lines of ``text`` wrapped to ``width`` (newest at bottom)."""
    lines = []
    for para in text.split("\n"):
        wrapped = textwrap.wrap(para, width=max(4, width)) or [""]
        lines.extend(wrapped)
    return "\n".join(lines[-height:])


def render(casts, vt, max_dur, size, baseline_tps, cursor_on):
    width, height = size
    n = len(casts)
    body_h = max(3, height - 11)
    pane_w = max(12, width // n - 4)

    prompt = casts[0].get("prompt_preview", "")
    header = Panel(Text(prompt, style="white"), title="[bold yellow]Prompt", border_style="yellow", padding=(0, 1))

    row = Table.grid(expand=True)
    for _ in casts:
        row.add_column(ratio=1)
    cells = []
    for c in casts:
        st = pane_state(c, vt)
        color = c.get("color", "white")
        tps = st["tps"] if not st["done"] else c["summary"]["tps"]
        speed = (c["summary"]["tps"] / baseline_tps) if baseline_tps > 0 else 0.0
        title = Text.assemble((c["label"], f"bold {color}"))
        badges = Text.assemble(
            (f" {tps:6.1f} tps ", f"black on {color}"), "  ",
            (f" {speed:.2f}x ", "bold white on grey23"), "  ",
            (f"{st['cum']}/{c['summary']['num_output_tokens']} tok", "grey70"),
        )
        stats = Text.assemble(
            (f"round {st['round']}/{c['summary']['rounds']}", "grey62"), "   ",
            (f"τ {st['tau']:.2f}", "grey62"),
            ("   ✓ done" if st["done"] else "", f"bold {color}"),
        )
        body = tail_wrapped(st["text"], pane_w, body_h)
        body_text = Text(body, style="white")
        if not st["done"] and cursor_on:
            body_text.append("▌", style=color)
        progress = bar(st["frac"], pane_w, color)
        pane = Panel(
            Group(badges, stats, Text(""), body_text, Text(""), progress),
            title=title, border_style=color if not st["done"] else "grey37",
            padding=(0, 1), height=body_h + 7,
        )
        cells.append(pane)
    row.add_row(*cells)

    tl_w = max(10, width - 20)
    tl_frac = min(1.0, vt / max_dur) if max_dur > 0 else 1.0
    timeline = Text.assemble(
        (f"{min(vt, max_dur):05.2f}", "bold white"), " ",
        ("█" * int(tl_frac * tl_w), "cyan"), ("─" * (tl_w - int(tl_frac * tl_w)), "grey37"),
        " ", (f"{max_dur:05.2f}s", "grey62"),
    )
    return Group(header, row, Panel(timeline, border_style="grey37", padding=(0, 1)))


def main():
    args = parse_args()
    casts = load_casts(args.casts)
    console = Console()

    # Baseline for speedup normalization.
    if args.baseline:
        base = next((c for c in casts if c["method"] == args.baseline), None)
    else:
        base = next((c for c in casts if c["method"] == "ar"), None)
    if base is None:
        base = min(casts, key=lambda c: c["summary"]["tps"])
    baseline_tps = base["summary"]["tps"]

    max_dur = max(c["summary"]["decode_time_s"] for c in casts)
    frame_dt = 1.0 / args.fps
    start = time.perf_counter()

    with Live(console=console, refresh_per_second=args.fps, screen=True) as live:
        while True:
            real = time.perf_counter() - start
            vt = real * args.speed
            cursor_on = int(real * 3) % 2 == 0
            live.update(render(casts, vt, max_dur, console.size, baseline_tps, cursor_on))
            if vt >= max_dur:
                break
            time.sleep(frame_dt)
        # Hold final frame.
        end = time.perf_counter()
        while time.perf_counter() - end < args.hold:
            live.update(render(casts, max_dur, max_dur, console.size, baseline_tps, True))
            time.sleep(frame_dt)

    # Final summary to normal scrollback.
    console.print()
    tbl = Table(title="Replay summary", show_edge=True)
    for col in ("method", "tps", "speedup", "tokens", "rounds", "τ", "decode_s"):
        tbl.add_column(col, justify="right")
    for c in casts:
        s = c["summary"]
        speed = s["tps"] / baseline_tps if baseline_tps > 0 else 0.0
        tbl.add_row(c["label"], f"{s['tps']:.1f}", f"{speed:.2f}x",
                    str(s["num_output_tokens"]), str(s["rounds"]),
                    f"{s['mean_accept']:.2f}", f"{s['decode_time_s']:.2f}")
    console.print(tbl)


if __name__ == "__main__":
    main()
