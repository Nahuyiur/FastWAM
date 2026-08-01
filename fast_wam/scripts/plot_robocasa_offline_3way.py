#!/usr/bin/env python3
"""Plot the formal RoboCasa offline-input benchmark without hiding repeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODES = (
    ("ordinary_online", "Ordinary\nonline VAE"),
    ("ordinary_offline", "Ordinary\nBF16 mmap"),
    ("webdataset_offline", "WebDataset\nBF16 tar"),
)
COLORS = ("#6B7280", "#0072B2", "#009E73")
HATCHES = ("", "//", "xx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--flat-cache-bytes", type=int, required=True)
    parser.add_argument("--wds-tar-bytes", type=int, required=True)
    parser.add_argument("--benchmark-samples", type=int, default=1024)
    parser.add_argument("--full-samples", type=int, default=286101)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    labels = [label for _, label in MODES]
    means_ms = [payload["modes"][mode]["step_seconds_mean"] * 1000 for mode, _ in MODES]
    repeat_ms = [
        [value * 1000 for value in payload["modes"][mode]["repeat_step_seconds_mean"]]
        for mode, _ in MODES
    ]
    full_scale = args.full_samples / args.benchmark_samples
    storage_gib = [
        args.flat_cache_bytes * full_scale / 2**30,
        args.wds_tar_bytes * full_scale / 2**30,
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (ax_latency, ax_storage) = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.2),
        layout="constrained",
        gridspec_kw={"width_ratios": (1.45, 1)},
    )

    x = np.arange(len(MODES))
    bars = ax_latency.bar(
        x,
        means_ms,
        color=COLORS,
        edgecolor="#202124",
        linewidth=0.8,
        width=0.62,
    )
    for bar, hatch in zip(bars, HATCHES):
        bar.set_hatch(hatch)
    offsets = (-0.12, 0.0, 0.12)
    for mode_index, observations in enumerate(repeat_ms):
        ax_latency.scatter(
            [mode_index + offset for offset in offsets],
            observations,
            marker="o",
            s=34,
            facecolor="white",
            edgecolor="#111827",
            linewidth=1,
            zorder=3,
            label="repeat mean" if mode_index == 0 else None,
        )
    for index, value in enumerate(means_ms):
        ax_latency.text(index, value + 8, f"{value:.1f}", ha="center", va="bottom")
    ax_latency.set(
        title="A. Steady-state training latency",
        ylabel="Iteration latency (ms, lower is better)",
        xticks=x,
        xticklabels=labels,
        ylim=(0, max(max(values) for values in repeat_ms) * 1.18),
    )
    ax_latency.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    ax_latency.legend(frameon=False, loc="upper right")

    storage_labels = ("BF16 mmap", "BF16 tar")
    storage_colors = (COLORS[1], COLORS[2])
    storage_bars = ax_storage.bar(
        np.arange(2),
        storage_gib,
        color=storage_colors,
        edgecolor="#202124",
        linewidth=0.8,
        width=0.58,
    )
    storage_bars[0].set_hatch("//")
    storage_bars[1].set_hatch("xx")
    for index, value in enumerate(storage_gib):
        ax_storage.text(index, value + 0.45, f"{value:.2f} GiB", ha="center")
    overhead = (storage_gib[1] / storage_gib[0] - 1) * 100
    ax_storage.text(
        0.5,
        max(storage_gib) * 0.58,
        f"tar overhead\n+{overhead:.1f}%",
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax_storage.set(
        title=f"B. Extrapolated latent storage\n({args.full_samples:,} windows)",
        ylabel="GiB (context cache excluded)",
        xticks=np.arange(2),
        xticklabels=storage_labels,
        ylim=(0, max(storage_gib) * 1.18),
    )
    ax_storage.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)

    fig.suptitle(
        "RoboCasa offline VAE input: tar does not provide a robust mmap speed advantage",
        fontsize=13,
        fontweight="bold",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
