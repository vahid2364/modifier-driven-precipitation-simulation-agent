"""
Model vs experiment comparison: dissolved metal concentration vs NaOH equivalents.

Three model scenarios (open / partial / closed) plotted as continuous curves;
CC-NRCan-078 ICP-OES data overlaid as symbols at 0, 0.3, 0.7, 1.0 equiv.

Usage:
    python src/plot_model_vs_exp.py \
        --base experiments/single \
        --out  experiments/model_vs_exp.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ── Metadata ──────────────────────────────────────────────────────────────────

METALS   = ["Al", "Fe", "Mn", "Cu", "Co", "Ni"]
MW       = {"Al": 26.982, "Fe": 55.845, "Cu": 63.546,
            "Co": 58.933, "Ni": 58.693, "Mn": 54.938}

# Initial dissolved concentration in vial (mg/L) — CC-NRCan-078, 0 NaOH equiv
INITIAL_MGL = {"Al": 294.8, "Fe": 289.2, "Cu": 1328.2,
                "Co": 4434.2, "Ni": 7426.8, "Mn": 3310.6}

# Experimental ICP-OES data: dissolved concentration (mg/L) at each NaOH equiv
# Source: CC-NRCan-078 sheet, rows 0 / 0.3 / 0.7 / 1.0 equiv
EXP_EQUIV = [0.0, 0.3, 0.7, 1.0]
EXP_MGL = {
    "Al": [294.8,   0.0,  82.6, 111.0],
    "Fe": [289.2,   0.0,   0.0,   0.0],
    "Cu": [1328.2, 844.0, 865.8, 913.2],
    "Co": [4434.2, 3281.0, 2993.0, 2742.2],
    "Ni": [7426.8, 6008.4, 6277.6, 6332.2],
    "Mn": [3310.6, 1572.4,   3.4,    2.4],
}

# Max NaOH dose (mol/kgw) = 1.0 equiv
NAOH_MAX = 0.627

METAL_COLORS = {
    "Al": "#4c72b0", "Fe": "#c44e52", "Cu": "#dd8452",
    "Co": "#55a868", "Ni": "#8172b3", "Mn": "#937860",
}
METAL_STYLE = {
    "Al": {"linestyle": "--", "lw": 1.8},
    "Fe": {"linestyle": "--", "lw": 1.8},
    "Cu": {"linestyle": "--", "lw": 1.8},
    "Co": {"linestyle": "-",  "lw": 1.8},
    "Ni": {"linestyle": "-",  "lw": 1.8},
    "Mn": {"linestyle": "-",  "lw": 2.2},   # thicker — key comparison metal
}

SCENARIOS = [
    {"key": "redox_open",    "label": "Open system\n(atmospheric O₂, 10 mol)"},
    {"key": "redox_partial", "label": "Partially open\n(dissolved O₂, 0.001 mol)"},
    {"key": "redox_closed",  "label": "Closed system\n(no O₂)"},
]

# Marker per scenario for experimental points
EXP_MARKERS = ["o", "s", "^"]
EXP_MARKER_SIZE = 7


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(path):
    df = pd.read_csv(path, sep="\t", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.replace(1e-99, np.nan, inplace=True)
    # Convert reaction (mol/kgw NaOH) to equivalents
    df["equiv"] = df["reaction"] / NAOH_MAX
    # Convert model mol/kgw to mg/L (mol/kgw × MW × 1000 for dilute vial ≈ mg/L)
    for m in METALS:
        if m in df.columns:
            df[f"{m}_mgl"] = df[m] * MW[m] * 1000
    return df


def pct_removed(mgl, initial):
    return (1 - np.array(mgl) / initial) * 100


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(base_dir, out_path):
    # Load models
    models = []
    for sc in SCENARIOS:
        csv = Path(base_dir) / sc["key"] / "neutralization.csv"
        if not csv.exists():
            raise FileNotFoundError(csv)
        models.append(load_model(csv))

    n_metals = len(METALS)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=False)
    axes = axes.flatten()

    fig.suptitle(
        "Model vs experiment: dissolved metal fraction vs NaOH equivalents\n"
        "NMC532 leachate + EDTA 0.20 mol/kgw (Na₄EDTA, 200 µL × 1 M stock)",
        fontsize=12, fontweight="bold", y=1.01
    )

    scenario_linestyles = ["-", "--", ":"]
    scenario_colors_bg  = ["#e8f4f8", "#fff8e8", "#f0f8e8"]

    for ax_idx, metal in enumerate(METALS):
        ax = axes[ax_idx]
        initial = INITIAL_MGL[metal]

        # ── Model curves ──
        for sc_idx, (sc, df) in enumerate(zip(SCENARIOS, models)):
            col = f"{metal}_mgl"
            if col not in df.columns:
                continue
            pct = (df[col] / initial * 100).clip(0, 100)
            ax.plot(df["equiv"], pct,
                    color=METAL_COLORS[metal],
                    linestyle=scenario_linestyles[sc_idx],
                    lw=METAL_STYLE[metal]["lw"],
                    alpha=0.85,
                    label=sc["label"].replace("\n", " ") if ax_idx == 0 else "_")

        # ── Experimental points ──
        exp_pct = [v / initial * 100 for v in EXP_MGL[metal]]
        ax.scatter(EXP_EQUIV, exp_pct,
                   color="black", marker="*", s=90, zorder=5,
                   label="Experiment (CC-NRCan-078)" if ax_idx == 0 else "_")

        # Formatting
        ax.set_title(metal, fontsize=13, fontweight="bold",
                     color=METAL_COLORS[metal], pad=6)
        ax.set_xlabel("NaOH equivalents", fontsize=10)
        if ax_idx in [0, 3]:
            ax.set_ylabel("Dissolved fraction (%)", fontsize=10)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-3, 108)
        ax.set_xticks([0, 0.3, 0.7, 1.0])
        ax.axvline(x=0.3, color="gray", lw=0.7, linestyle=":", alpha=0.5)
        ax.axvline(x=0.7, color="gray", lw=0.7, linestyle=":", alpha=0.5)
        ax.axvline(x=1.0, color="gray", lw=0.7, linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.2)

    # ── Shared legend ──
    scenario_handles = [
        mlines.Line2D([0], [0], color="dimgray",
                      linestyle=scenario_linestyles[i], lw=1.8,
                      label=SCENARIOS[i]["label"].replace("\n", " "))
        for i in range(len(SCENARIOS))
    ]
    exp_handle = mlines.Line2D([0], [0], color="black", marker="*",
                                linestyle="none", ms=8, label="Experiment (CC-NRCan-078)")
    fig.legend(handles=scenario_handles + [exp_handle],
               loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="experiments/single")
    parser.add_argument("--out",  default="experiments/model_vs_exp.png")
    args = parser.parse_args()
    make_figure(args.base, args.out)


if __name__ == "__main__":
    main()
