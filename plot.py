#!/usr/bin/env python3
"""Reliability diagram + auto-route trade-off. Light background, phone-readable."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
NAVY = "#1d4f91"
INK = "#1a1a1a"
MUTED = "#5c6570"
BAR_SAID = "#c5ced6"
BAR_RIGHT = NAVY
RED = "#b33a3a"


def main() -> None:
    s = json.loads((ROOT / "results" / "metrics.json").read_text())
    bins = [b for b in s["bins"] if b["n"] >= 1]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 10.2), dpi=160, gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.set_facecolor("white")
    x = list(range(len(bins)))
    w = 0.36
    said = [b["mean_conf"] * 100 for b in bins]
    right = [b["accuracy"] * 100 for b in bins]
    ax.bar([i - w / 2 for i in x], said, w, color=BAR_SAID, label="How sure d1-2B said it was")
    ax.bar([i + w / 2 for i in x], right, w, color=BAR_RIGHT, label="How often it was actually right")
    for i, b in enumerate(bins):
        label = f"{right[i]:.1f}%" if right[i] >= 99 else f"{right[i]:.0f}%"
        ax.text(i + w / 2, right[i] + 1.6, label, ha="center", va="bottom",
                fontsize=11, color=NAVY, fontweight="bold")
        ax.text(i, -9.5, f"n={b['n']}", ha="center", va="top", fontsize=9, color=MUTED)
    ax.set_xticks(x, [b["bin"].replace("0.", ".") for b in bins], fontsize=10)
    ax.set_xlabel("Confidence bucket (d1-2B’s own probability)", fontsize=11, color=INK, labelpad=18)
    ax.set_ylabel("Percent", fontsize=11, color=INK)
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.tick_params(colors=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    acc = s["accuracy"] * 100
    mean_c = s["mean_conf"] * 100
    ece = s["ece_equal_10"] * 100
    ax.set_title(
        "When d1-2B says it is 90% sure, is it right 90% of the time?\n"
        f"Zero-shot on 3,080 BANKING77 bank tickets  ·  "
        f"accuracy {acc:.1f}%  ·  avg confidence {mean_c:.1f}%  ·  ECE {ece:.1f} pp",
        loc="left",
        fontsize=12.5,
        color=INK,
        pad=10,
    )

    ax2 = axes[1]
    ax2.set_facecolor("white")
    route = s["route"]
    ts = [q["threshold"] for q in route]
    cov = [q["coverage"] * 100 for q in route]
    prec = [q["accuracy"] * 100 for q in route]
    ax2.plot(ts, cov, "o-", color=NAVY, lw=2.4, ms=8, label="Share of tickets auto-routed")
    ax2.plot(ts, prec, "s-", color=RED, lw=2.4, ms=8, label="Accuracy on those tickets")
    # Annotate through 0.95. 0.98/0.99 sit on top of each other at phone width.
    labelled = {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
    for q in route:
        if round(q["threshold"], 2) not in labelled:
            continue
        ax2.annotate(f"{q['coverage'] * 100:.0f}%", (q["threshold"], q["coverage"] * 100),
                     textcoords="offset points", xytext=(0, -18), ha="center", color=NAVY, fontsize=9)
        ax2.annotate(f"{q['accuracy'] * 100:.1f}%", (q["threshold"], q["accuracy"] * 100),
                     textcoords="offset points", xytext=(0, 8), ha="center", color=RED, fontsize=9)
    ax2.set_xlabel("Only auto-route a ticket when confidence is at least …", fontsize=11, color=INK)
    ax2.set_ylabel("Percent", fontsize=11, color=INK)
    ax2.set_xlim(0.47, 1.02)
    ax2.set_ylim(0, 112)
    ax2.spines[["top", "right"]].set_visible(False)
    for sp in ax2.spines.values():
        sp.set_color("#333")
    ax2.tick_params(colors=INK)
    ax2.legend(loc="lower left", frameon=False, fontsize=10)
    ax2.set_title(
        "Production cut: how many tickets can you skip a human on?",
        loc="left",
        fontsize=12.5,
        color=INK,
        pad=8,
    )

    fig.tight_layout(h_pad=3.2)
    out = ROOT / "results" / "reliability.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
