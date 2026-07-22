"""
AnalystAgent: loads simulation CSVs, generates comparison figure, computes
the modifier selectivity scorecard, and writes results back to run_config.yaml.
All metals, thresholds, and experimental data come from run_config.yaml.
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from datetime import datetime

import sys; sys.path.insert(0, ".")
from src.agents.base import BaseAgent
from state import SimState

METAL_COLORS = {
    "Al": "#4c72b0", "Fe": "#c44e52", "Cu": "#dd8452",
    "Co": "#55a868", "Ni": "#8172b3", "Mn": "#937860",
}
SCENARIO_STYLE = {
    "open":    {"color": "#1f77b4", "ls": "-"},
    "closed":  {"color": "#2ca02c", "ls": "--"},
    "partial": {"color": "#d62728", "ls": ":"},
}


def _load_csv(path: str, naoh_max: float, metals: list, input_mol: dict) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.replace(1e-99, np.nan, inplace=True)
    df["equiv"] = df["reaction"] / naoh_max
    for m in metals:
        if m in df.columns:
            df[f"pct_{m}"] = (df[m] / input_mol[m] * 100).clip(0, 100)
    return df


def _exp_pct(metal: str, exp_cfg: dict) -> list:
    init = exp_cfg["mg_per_L"][metal][0]
    return [v / init * 100 for v in exp_cfg["mg_per_L"][metal]]


# sit.dat-resolved phase names for each metal
METAL_PHASE_LABEL = {
    "Al": "Gibbsite",         "Fe": "Ferrihydrite(am)", "Cu": "Cu(OH)2(s)",
    "Co": "Co(OH)2(s)",       "Ni": "Ni(OH)2(s)",       "Mn": "MnOOH",
}
PRECIP_PHASES = {
    "Al": "Gibbsite",         "Fe": "Ferrihydrite(am)", "Cu": "Cu(OH)2(s)",
    "Co": "Co(OH)2(s)",       "Ni": "Ni(OH)2(s)",
}
MN_PHASES = ["Mn(OH)2(s)", "Manganite", "Hausmannite"]


def _add_fractions(df: pd.DataFrame, metals: list, input_mol: dict) -> pd.DataFrame:
    for m in metals:
        if m in df.columns:
            df[f"frac_dissolved_{m}"] = (df[m] / input_mol[m]).clip(0, 1)
        if m == "Mn":
            total = sum(df[p] for p in MN_PHASES if p in df.columns)
            df["frac_precipitated_Mn"] = (total / input_mol["Mn"]).clip(0, 1)
        else:
            phase = PRECIP_PHASES.get(m)
            if phase and phase in df.columns:
                df[f"frac_precipitated_{m}"] = (df[phase] / input_mol[m]).clip(0, 1)
    return df


def _onset_pH(df, metal, threshold=0.05):
    col = f"frac_precipitated_{metal}"
    if col not in df.columns:
        return None
    mask = df[col] >= threshold
    return float(df.loc[mask.idxmax(), "pH"]) if mask.any() else None


def _complete_pH(df, metal, threshold=0.95):
    col = f"frac_precipitated_{metal}"
    if col not in df.columns:
        return None
    mask = df[col] >= threshold
    return float(df.loc[mask.idxmax(), "pH"]) if mask.any() else None


def _precip_order_figure(df: pd.DataFrame, metals: list, slug: str,
                         scenario: str, out_path: Path):
    records = []
    for m in metals:
        on  = _onset_pH(df, m)
        off = _complete_pH(df, m)
        if on is not None:
            records.append({
                "metal":    m,
                "phase":    METAL_PHASE_LABEL.get(m, m),
                "onset":    on,
                "complete": off if off is not None else float(df["pH"].max()),
            })
    if not records:
        return
    records.sort(key=lambda r: r["onset"])

    fig, ax = plt.subplots(figsize=(10, max(3, len(records) * 0.8)))
    for i, rec in enumerate(records):
        width = max(rec["complete"] - rec["onset"], 0.05)
        ax.barh(i, width, left=rec["onset"], height=0.55,
                color=METAL_COLORS.get(rec["metal"], "gray"),
                alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.plot(rec["onset"], i, marker="|", color="black",
                markersize=10, markeredgewidth=1.5)
        mid = rec["onset"] + width / 2
        ax.text(mid, i, rec["phase"], ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_yticks(range(len(records)))
    ax.set_yticklabels([r["metal"] for r in records], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("pH", fontsize=12)
    ax.set_xlim(0, float(df["pH"].max()) + 0.5)
    ax.set_title(
        f"Precipitation order — {slug} ({scenario} system)",
        fontsize=12, fontweight="bold"
    )
    ax.plot([], [], marker="|", color="black", markersize=8,
            linestyle="none", label="5% onset")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[AnalystAgent] Saved precipitation order: {out_path}")


def _rmse(df: pd.DataFrame, metals: list, input_mol: dict, exp_cfg: dict) -> dict:
    """RMSE (pp dissolved) between open-system model and experiment at exp equiv points."""
    exp_equiv = exp_cfg["equiv"]
    out = {}
    for m in metals:
        col = f"pct_{m}"
        if col not in df.columns:
            continue
        exp_vals = exp_cfg["mg_per_L"][m]
        init = exp_vals[0]
        if not init:
            continue
        exp_pcts = [v / init * 100 for v in exp_vals]
        model_pcts = [
            float(df.iloc[(df["equiv"] - eq).abs().argmin()][col])
            for eq in exp_equiv
        ]
        out[m] = round(float(np.sqrt(np.mean(
            (np.array(model_pcts) - np.array(exp_pcts)) ** 2
        ))), 2)
    return out


def _species_contribution_figure(df: pd.DataFrame, metals: list, input_mol: dict,
                                  exp_equiv: list, slug: str, out_path: Path):
    """Stacked bar: how much each dissolved complex + precipitate holds each metal."""
    x      = np.arange(len(exp_equiv))
    idx_list = [(df["equiv"] - eq).abs().argmin() for eq in exp_equiv]

    ncols = 3
    nrows = -(-len(metals) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for ax, metal in zip(axes, metals):
        prefix = f"m_{metal}"
        species_cols = [c for c in df.columns if c.startswith(prefix)]
        if not species_cols:
            ax.set_visible(False)
            continue

        cmap   = plt.cm.tab10(np.linspace(0, 0.9, len(species_cols) + 1))
        bottom = np.zeros(len(exp_equiv))

        for i, col in enumerate(species_cols):
            vals = np.array([
                min(float(df.iloc[idx][col]) / input_mol[metal] * 100, 100)
                for idx in idx_list
            ])
            label = col[2:]   # strip "m_"
            ax.bar(x, vals, bottom=bottom, color=cmap[i], label=label, edgecolor="white", linewidth=0.4)
            bottom += vals

        # precipitated = remainder
        precip = np.clip(100 - bottom, 0, 100)
        ax.bar(x, precip, bottom=bottom, color="lightgray", label="precipitated",
               edgecolor="white", linewidth=0.4, hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels([str(eq) for eq in exp_equiv], fontsize=9)
        ax.set_xlabel("NaOH equiv", fontsize=9)
        ax.set_ylabel("% of input", fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_title(metal, color=METAL_COLORS.get(metal, "black"),
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", ncol=1)

    for ax in axes[len(metals):]:
        ax.set_visible(False)

    fig.suptitle(f"Species contributions — {slug} (open system)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[AnalystAgent] Saved species contribution: {out_path}")


def _scorecard(dfs: dict, metals: list, input_mol: dict,
               threshold: float, comparison_equiv: list,
               target_metals: list, impurity_metals: list) -> dict:
    """Score each metal based on its role.

    target_metals:   we want these to remain dissolved  → high pct = good
    impurity_metals: we want these to precipitate       → low pct = good
    """
    df = dfs.get("open")
    if df is None:
        return {}
    scorecard = {}
    for m in metals:
        col = f"pct_{m}"
        if col not in df.columns:
            continue
        scores = {}
        is_target = m in target_metals
        for eq in comparison_equiv:
            idx = (df["equiv"] - eq).abs().argmin()
            val = float(df.iloc[idx][col])
            scores[f"pct_at_{eq}_equiv"] = round(val, 1)
            if is_target:
                scores[f"protected_{eq}"] = val > threshold * 100
            else:
                # impurity: success means it has precipitated (low dissolved pct)
                scores[f"removed_{eq}"] = val < (1 - threshold) * 100
        if is_target:
            scores["score"]  = sum(int(scores[f"protected_{eq}"]) for eq in comparison_equiv)
            scores["role"]   = "target"
        else:
            scores["score"]  = sum(int(scores[f"removed_{eq}"]) for eq in comparison_equiv)
            scores["role"]   = "impurity"
        scores["max_score"] = len(comparison_equiv)
        scorecard[m] = scores
    return scorecard


def _figure(dfs: dict, metals: list, exp_cfg: dict, slug: str, out_path: Path):
    ncols = 3
    nrows = -(-len(metals) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flatten()

    exp_equiv = exp_cfg["equiv"]

    for ax, metal in zip(axes, metals):
        for scenario, df in dfs.items():
            col = f"pct_{metal}"
            if col in df.columns:
                st = SCENARIO_STYLE.get(scenario, {"color": "gray", "ls": "-"})
                ax.plot(df["equiv"], df[col], color=st["color"],
                        lw=1.8, linestyle=st["ls"], label=scenario)

        exp = _exp_pct(metal, exp_cfg)
        ax.scatter(exp_equiv, exp, color="black", marker="*", s=120,
                   zorder=6, label="Experiment")

        ax.set_title(metal, color=METAL_COLORS.get(metal, "black"),
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("NaOH equivalents", fontsize=10)
        ax.set_ylabel("Dissolved fraction (%)", fontsize=10)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-3, 108)
        ax.set_xticks([0, 0.3, 0.7, 1.0])
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for ax in axes[len(metals):]:
        ax.set_visible(False)

    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Model vs experiment: {slug}\n"
        f"NMC532 leachate, modifier 0.20 mol/kgw "
        f"[ref: {exp_cfg.get('source', 'unknown')}]",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[AnalystAgent] Saved figure: {out_path}")


class AnalystAgent(BaseAgent):
    name = "AnalystAgent"

    def run(self, state: SimState) -> SimState:
        if state.get("status") != "simulated":
            state["error"] = "AnalystAgent: expected status 'simulated'"
            return state

        cfg             = state["run_config"]
        leachate_cfg    = cfg["leachate"]
        exp_cfg         = cfg["experiment"]
        analyst_cfg     = cfg["analyst"]
        metals          = leachate_cfg["metals"]
        naoh_max        = leachate_cfg["naoh_max"]
        input_mol       = leachate_cfg["input_mol"]
        threshold        = analyst_cfg["protection_threshold"]
        comparison_equiv = analyst_cfg["comparison_equiv"]
        target_metals    = analyst_cfg.get("target_metals", metals)
        impurity_metals  = analyst_cfg.get("impurity_metals", [])

        slug = Path(state["modifier_yaml_path"]).stem
        dfs  = {s: _load_csv(p, naoh_max, metals, input_mol)
                for s, p in state["sim_paths"].items()
                if Path(p).exists()}

        # Add precipitated fractions for order plots
        for df in dfs.values():
            _add_fractions(df, metals, input_mol)

        scorecard = _scorecard(dfs, metals, input_mol, threshold, comparison_equiv,
                               target_metals, impurity_metals)

        open_df = dfs.get("open")
        if open_df is not None:
            rmse = _rmse(open_df, metals, input_mol, exp_cfg)
            for m, val in rmse.items():
                if m in scorecard:
                    scorecard[m]["rmse"] = val

        fig_path  = Path(f"experiments/{slug}_model_vs_exp.png")
        _figure(dfs, metals, exp_cfg, slug, fig_path)

        if open_df is not None:
            sp_path = Path(f"experiments/{slug}_species_contribution.png")
            _species_contribution_figure(
                open_df, metals, input_mol, exp_cfg["equiv"], slug, sp_path
            )

        # Precipitation order figure — one per scenario
        for scenario, df in dfs.items():
            order_path = Path(f"experiments/single/{slug}_{scenario}/{slug}_{scenario}_precip_order.png")
            _precip_order_figure(df, metals, slug, scenario, order_path)

        print(f"[{self.name}] Scorecard:")
        for m, s in scorecard.items():
            parts = [f"{s[f'pct_at_{eq}_equiv']:.0f}% @ {eq} eq"
                     for eq in comparison_equiv]
            role  = "PROTECT" if s["role"] == "target" else "REMOVE"
            rmse_str = f"  RMSE={s['rmse']:.1f}pp" if "rmse" in s else ""
            print(f"  {m:2s}  {' | '.join(parts)}  [{s['score']}/{s['max_score']}] ({role}){rmse_str}")

        # ── Write scorecard back to the most recent run entry in run_config ───
        cfg_path = Path(state["config_path"])
        runs     = cfg.get("runs", [])
        for run in reversed(runs):
            if run.get("modifier_slug") == slug:
                run["scorecard"]              = scorecard
                run["figure"]                = str(fig_path)
                run["species_contribution"]  = str(sp_path) if open_df is not None else None
                run["precip_order_figs"] = [
                    f"experiments/single/{slug}_{s}/{slug}_{s}_precip_order.png"
                    for s in dfs
                ]
                run["status"]     = "analysed"
                run["completed"]  = datetime.utcnow().isoformat()
                break

        cfg_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        print(f"[{self.name}] Scorecard written back to {cfg_path}")

        state["scorecard"]                 = scorecard
        state["figure_path"]               = str(fig_path)
        state["species_contribution_path"] = str(sp_path) if open_df is not None else None
        state["status"]                    = "analysed"
        state["error"]                     = None
        return state
