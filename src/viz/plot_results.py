"""
Plot stage 3 (NaOH neutralization) results from a phreeqc-modifier-agent experiment.

Usage:
    python src/plot_results.py \
        --experiment experiments/single/EDTA_dose0p20_NaOH4p00 \
        --modifier "EDTA 0.20 mol/kgw" --naoh-dose 4.0
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from pathlib import Path


# ── Metadata ──────────────────────────────────────────────────────────────────

METALS = ["Al", "Fe", "Cu", "Co", "Ni", "Mn"]

# Phase that precipitates per metal (for legend annotation)
METAL_PHASE_LABEL = {
    "Al": "Al(OH)₃(am)",
    "Fe": "Fe(OH)₃(a)",
    "Cu": "Cu(OH)₂(s)",
    "Co": "Co(OH)₂(s)",
    "Ni": "Ni(OH)₂(s)",
    "Mn": "MnOOH",       # Manganite under oxidizing conditions
}

# Primary PHREEQC column per metal
PHASES = ["Al(OH)3(am)", "Fe(OH)3(a)", "Cu(OH)2(s)",
          "Co(OH)2(s)", "Ni(OH)2(s)", "Manganite"]

MN_ALL_PHASES = ["Mn(OH)2(s)", "Manganite", "Hausmannite"]

INPUT_MOL = {"Al": 0.010926, "Fe": 0.005179, "Cu": 0.020901,
             "Co": 0.075241, "Ni": 0.126536, "Mn": 0.060261}

EDTA_COLS = {
    "Cu": "m_CuEdta-2", "Co": "m_CoEdta-2",
    "Ni": "m_NiEdta-2", "Mn": "m_MnEdta-2",
    "Al": "m_AlEdta-",
}

METAL_COLORS = {
    "Al": "#4c72b0",
    "Fe": "#c44e52",
    "Cu": "#dd8452",
    "Co": "#55a868",
    "Ni": "#8172b3",
    "Mn": "#937860",
}

METAL_STYLE = {
    "Al": {"linestyle": "--", "marker": "o", "lw": 2.2},
    "Fe": {"linestyle": "--", "marker": "o", "lw": 2.2},
    "Cu": {"linestyle": "--", "marker": "o", "lw": 2.2},
    "Co": {"linestyle": "-",  "marker": "s", "lw": 1.8},
    "Ni": {"linestyle": "-",  "marker": "^", "lw": 1.8},
    "Mn": {"linestyle": "-",  "marker": "D", "lw": 1.8},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path, sep="\t", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.replace(1e-99, np.nan, inplace=True)
    return df


def add_fractions(df):
    for m in METALS:
        if m in df.columns:
            df[f"frac_dissolved_{m}"] = (df[m] / INPUT_MOL[m]).clip(0, 1)
        if m == "Mn":
            total = sum(df[p] for p in MN_ALL_PHASES if p in df.columns)
            df["frac_precipitated_Mn"] = (total / INPUT_MOL["Mn"]).clip(0, 1)
        else:
            phase = next((p for p in PHASES if p.startswith(m)), None)
            if phase and phase in df.columns:
                df[f"frac_precipitated_{m}"] = (df[phase] / INPUT_MOL[m]).clip(0, 1)
        ecol = EDTA_COLS.get(m)
        if ecol and ecol in df.columns:
            df[f"frac_edta_{m}"] = (df[ecol] / INPUT_MOL[m]).clip(0, 1)
    return df


def onset_pH(df, metal, threshold=0.05):
    """Return pH at which frac_precipitated first exceeds threshold."""
    col = f"frac_precipitated_{metal}"
    if col not in df.columns:
        return None
    mask = df[col] >= threshold
    if not mask.any():
        return None
    return df.loc[mask.idxmax(), "pH"]


def complete_pH(df, metal, threshold=0.95):
    """Return pH at which frac_precipitated first exceeds threshold."""
    col = f"frac_precipitated_{metal}"
    if col not in df.columns:
        return None
    mask = df[col] >= threshold
    if not mask.any():
        return None
    return df.loc[mask.idxmax(), "pH"]


# ── Sub-plots ─────────────────────────────────────────────────────────────────

def _legend_label(m):
    return f"{m}  [{METAL_PHASE_LABEL[m]}]"


def plot_dissolved(df, ax):
    for m in METALS:
        col = f"frac_dissolved_{m}"
        if col in df.columns:
            s = METAL_STYLE[m]
            ax.plot(df["pH"], df[col] * 100,
                    color=METAL_COLORS[m], lw=s["lw"], marker=s["marker"],
                    linestyle=s["linestyle"], ms=4, label=m)
    ax.set_xlabel("pH", fontsize=12)
    ax.set_ylabel("Dissolved fraction (%)", fontsize=12)
    ax.set_title("Dissolved metals vs pH", fontsize=13)
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_precipitated(df, ax):
    for m in METALS:
        col = f"frac_precipitated_{m}"
        if col in df.columns:
            s = METAL_STYLE[m]
            ax.plot(df["pH"], df[col] * 100,
                    color=METAL_COLORS[m], lw=s["lw"], marker=s["marker"],
                    linestyle=s["linestyle"], ms=4,
                    label=_legend_label(m))
    ax.set_xlabel("pH", fontsize=12)
    ax.set_ylabel("Precipitated fraction (%)", fontsize=12)
    ax.set_title("Precipitated phases vs pH", fontsize=13)
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=8.5, ncol=1, loc="center right")
    ax.grid(True, alpha=0.3)


def plot_ionic_strength(df, ax):
    ax.plot(df["pH"], df["mu"], color="black", lw=2)
    ax.set_xlabel("pH", fontsize=12)
    ax.set_ylabel("Ionic strength (mol/kgw)", fontsize=12)
    ax.set_title("Ionic strength vs pH", fontsize=13)
    ax.grid(True, alpha=0.3)


def plot_edta_complexation(df, ax):
    any_plotted = False
    for m in METALS:
        col = f"frac_edta_{m}"
        if col in df.columns and df[col].notna().any():
            s = METAL_STYLE[m]
            ax.plot(df["pH"], df[col] * 100,
                    color=METAL_COLORS[m], lw=s["lw"], marker=s["marker"],
                    linestyle=s["linestyle"], ms=4, label=m)
            any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "No EDTA complexes detected",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("pH", fontsize=12)
    ax.set_ylabel("EDTA-complexed fraction (%)", fontsize=12)
    ax.set_title("Metal-EDTA complexation vs pH", fontsize=13)
    ax.set_ylim(-0.2, None)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_precipitation_order(df, ax):
    """
    Horizontal bar chart showing onset and completion pH for each metal,
    ordered by precipitation onset. Annotates the precipitating phase.
    """
    records = []
    for m in METALS:
        on  = onset_pH(df, m, threshold=0.05)
        off = complete_pH(df, m, threshold=0.95)
        if on is not None:
            records.append({
                "metal":  m,
                "phase":  METAL_PHASE_LABEL[m],
                "onset":  on,
                "complete": off if off is not None else df["pH"].max(),
            })

    records.sort(key=lambda r: r["onset"])

    y_pos = range(len(records))
    bar_height = 0.55

    for i, rec in enumerate(records):
        width = rec["complete"] - rec["onset"]
        ax.barh(i, width, left=rec["onset"], height=bar_height,
                color=METAL_COLORS[rec["metal"]], alpha=0.85,
                edgecolor="black", linewidth=0.7)
        # Onset marker
        ax.plot(rec["onset"], i, marker="|", color="black",
                markersize=10, markeredgewidth=1.5)
        # Phase label inside bar
        mid = rec["onset"] + width / 2
        ax.text(mid, i, rec["phase"],
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white",
                path_effects=[
                    __import__("matplotlib.patheffects", fromlist=["withStroke"])
                    .withStroke(linewidth=2, foreground="black")
                ])

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{r['metal']}" for r in records], fontsize=11
    )
    ax.set_xlabel("pH", fontsize=12)
    ax.set_title("Precipitation onset and completion order", fontsize=13)
    ax.set_xlim(0, df["pH"].max() + 0.5)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Legend: onset marker explanation
    ax.plot([], [], marker="|", color="black", markersize=8,
            label="5% precipitation onset", linestyle="none")
    ax.axvline(x=-1, color="gray", linestyle="--", alpha=0)  # invisible
    ax.legend(fontsize=9, loc="lower right")


# ── Figure builders ───────────────────────────────────────────────────────────

def make_main_figure(df, out_path, modifier_name, naoh_dose):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"NaOH neutralization — NMC532 leachate + {modifier_name} "
        f"(NaOH dose = {naoh_dose} mol/kgw)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plot_dissolved(df, axes[0, 0])
    plot_precipitated(df, axes[0, 1])
    plot_ionic_strength(df, axes[1, 0])
    plot_edta_complexation(df, axes[1, 1])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def make_order_figure(df, out_path, modifier_name, naoh_dose):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"Precipitation order — NMC532 leachate + {modifier_name} "
        f"(NaOH dose = {naoh_dose} mol/kgw)",
        fontsize=12, fontweight="bold"
    )
    plot_precipitation_order(df, ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--modifier",   default="EDTA")
    parser.add_argument("--naoh-dose",  type=float, default=2.0)
    args = parser.parse_args()

    exp = Path(args.experiment)
    neut_csv = exp / "neutralization.csv"
    if not neut_csv.exists():
        raise FileNotFoundError(f"neutralization.csv not found in {exp}")

    df = load(neut_csv)
    df = add_fractions(df)

    make_main_figure(df, exp / "stage3_neutralization.png",
                     modifier_name=args.modifier, naoh_dose=args.naoh_dose)
    make_order_figure(df, exp / "stage3_precipitation_order.png",
                      modifier_name=args.modifier, naoh_dose=args.naoh_dose)


if __name__ == "__main__":
    main()
