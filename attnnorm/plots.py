import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.grid": True, "grid.alpha": 0.3,
})

COLORS = {"true": "#2ecc40", "expected": "#39cccc", "bound": "#ff7f50",
          "ns_ratio": "#2ecc40", "ns_lower": "#39cccc", "ns_upper": "#ff7f50"}
LABELS = {"true": "True distance", "expected": "Expectation", "bound": "Upper bound",
          "ns_ratio": r"True $N_s/N$", "ns_lower": "Lower bound", "ns_upper": "Upper bound"}


def grouped_boxes(ax, groups, series):
    k = len(series)
    width = 0.8 / k
    for j, (name, data) in enumerate(series.items()):
        pos = [i + (j - (k - 1) / 2) * width for i in range(len(groups))]
        clean = [d[~np.isnan(d)] for d in data]
        bp = ax.boxplot(clean, positions=pos, widths=width * 0.9, patch_artist=True,
                        flierprops=dict(markersize=3))
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[name])
        for med in bp["medians"]:
            med.set_color("black")
        ax.plot([], [], "s", color=COLORS[name], markersize=12, label=LABELS[name])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
