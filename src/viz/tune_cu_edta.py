"""
Sweep CuEdta-2 log_k values and compare dissolved Cu vs NaOH equivalents
against CC-NRCan-078 experimental data.

Usage:
    python src/tune_cu_edta.py
"""

import subprocess, shutil, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

LOG_K_VALUES  = [18.80, 19.0, 19.5, 20.0, 20.5, 21.0, 22.0]   # fine sweep around transition
NAOH_MAX      = 0.627                       # mol/kgw = 1.0 equiv
CU_INPUT_MOL  = 0.020901
CU_MW         = 63.546

EXP_EQUIV = [0.0, 0.3, 0.7, 1.0]
EXP_MGL   = [1328.2, 844.0, 865.8, 913.2]
EXP_INIT  = 1328.2

PHREEQC_EXE = shutil.which("phreeqc") or "/Users/vahid/.local/bin/phreeqc"
DATABASE    = "/Users/vahid/.local/share/doc/phreeqc/database/sit.dat"

SCRATCH = Path("experiments/cu_edta_tune")
SCRATCH.mkdir(parents=True, exist_ok=True)

EDTA_PHR_BASE = Path("database/modifiers/EDTA.phr").read_text()

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_edta_phr(log_k_cu):
    """Return EDTA.phr content with CuEdta-2 log_k replaced."""
    return re.sub(
        r"(Cu\+2 \+ Edta-4 = CuEdta-2\s+log_k\s+)[\d.]+",
        rf"\g<1>{log_k_cu}",
        EDTA_PHR_BASE
    )


def run_sim(log_k_cu):
    tag = f"logk{str(log_k_cu).replace('.','p')}"
    out_dir = SCRATCH / tag
    out_dir.mkdir(exist_ok=True)

    # Write patched EDTA.phr
    edta_phr = out_dir / "EDTA.phr"
    edta_phr.write_text(make_edta_phr(log_k_cu))

    # Generate input using generate_input.py with patched modifier db
    import sys; sys.path.insert(0, "src/pipeline")
    import yaml
    from generate_input import load_yaml, assemble_database, render_template

    leachate = load_yaml("config/nmc532_exp_open.yaml")
    modifier = load_yaml("config/modifiers/EDTA.yaml")["modifier"]
    modifier["database"] = str(edta_phr)

    db_header = assemble_database(
        phases_path="database/phases.phr",
        solution_species_path="database/solution_species.phr",
        modifier_db_path=str(edta_phr),
    )
    context = {
        "leachate":           leachate["leachate"],
        "neutralization":     leachate["neutralization"],
        "equilibrium_phases": leachate["equilibrium_phases"],
        "knobs":              leachate["knobs"],
        "modifier":           modifier,
        "o2_reservoir_mol":   leachate.get("o2_reservoir_mol", 10.0),
    }
    rendered = render_template("templates", "neutralization.phr.j2", context)
    inp_path = out_dir / "input.phr"
    inp_path.write_text(db_header + "\n\n" + rendered)

    log_path = out_dir / "phreeqc.log"
    subprocess.run(
        [PHREEQC_EXE, str(inp_path), str(log_path), DATABASE],
        capture_output=True, check=True
    )

    # Move output CSVs
    for fname in ["neutralization.csv", "leaching.csv"]:
        src = Path(fname)
        if src.exists():
            src.rename(out_dir / fname)

    return out_dir / "neutralization.csv"


def load_cu_dissolved(csv_path):
    df = pd.read_csv(csv_path, sep="\t", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.replace(1e-99, np.nan, inplace=True)
    df["equiv"] = df["reaction"] / NAOH_MAX
    df["Cu_pct"] = (df["Cu"] / CU_INPUT_MOL * 100).clip(0, 100)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}
    for lk in LOG_K_VALUES:
        print(f"Running log_k = {lk} ...")
        csv = run_sim(lk)
        results[lk] = load_cu_dissolved(csv)
        print(f"  Done. Cu range: {results[lk]['Cu_pct'].min():.1f}% – {results[lk]['Cu_pct'].max():.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(LOG_K_VALUES) - 1)) for i in range(len(LOG_K_VALUES))]

    fig, ax = plt.subplots(figsize=(8, 5))

    for (lk, df), col in zip(results.items(), colors):
        lw = 2.4 if lk == 18.80 else 1.6
        ls = "--" if lk == 18.80 else "-"
        ax.plot(df["equiv"], df["Cu_pct"], color=col, lw=lw, linestyle=ls,
                label=f"log K = {lk}")

    # Experimental points
    exp_pct = [v / EXP_INIT * 100 for v in EXP_MGL]
    ax.scatter(EXP_EQUIV, exp_pct, color="black", marker="*", s=120, zorder=6,
               label="Experiment (CC-NRCan-078)")

    ax.axhline(y=exp_pct[1], color="gray", lw=0.8, linestyle=":", alpha=0.6)
    ax.text(1.02, exp_pct[1], f"{exp_pct[1]:.0f}%", va="center", fontsize=9, color="gray")

    ax.set_xlabel("NaOH equivalents", fontsize=12)
    ax.set_ylabel("Cu dissolved fraction (%)", fontsize=12)
    ax.set_title("CuEdta²⁻ log K sensitivity — dissolved Cu vs NaOH equivalents\n"
                 "Open system (atmospheric O₂), EDTA 0.20 mol/kgw (Na₄EDTA)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-3, 108)
    ax.set_xticks([0, 0.3, 0.7, 1.0])
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10, loc="center right")

    out = SCRATCH / "cu_edta_sensitivity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
