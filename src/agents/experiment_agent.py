"""
ExperimentAgent: reads experimental HTE data from the master xlsx file and
injects the correct modifier's measurements into pipeline state.

Replaces the hardcoded 'experiment:' block in run_config.yaml with data
looked up by modifier name at runtime.

xlsx structure (per sheet):
  row 5  col 1 : modifier name
  rows starting at col 2 = "Concentration of organic modifier (M)"
      → one sub-table per modifier dose tested
      → col 4 = NaOH Equivs, cols 5+ = metals (Al, Co, Cu, Fe, Li, Mn, Ni)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

import sys; sys.path.insert(0, ".")
from src.agents.base import BaseAgent
from state import SimState

# Metals the pipeline cares about (Li excluded — not tracked in PHREEQC)
_PIPELINE_METALS = {"Al", "Fe", "Cu", "Co", "Ni", "Mn"}

# Token-score threshold for fuzzy modifier name matching
_MATCH_THRESHOLD = 2


def _normalize(name: str) -> list[str]:
    """Split name into lowercase tokens for fuzzy matching."""
    return [t.lower() for t in re.split(r"[\s\-(),&]+", name) if len(t) > 2]


def _match_sheet(xl: pd.ExcelFile, modifier_name: str) -> tuple[str | None, str | None]:
    """Return (sheet_name, matched_modifier_name) for the best-matching sheet."""
    query_tokens = _normalize(modifier_name)
    best_sheet, best_label, best_score = None, None, 0

    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet, header=None, nrows=8)
        # Row 5, col 1 holds the modifier salt name
        try:
            label = str(df.iloc[5, 1]).strip()
        except IndexError:
            continue
        if not label or label.lower() in ("nan", "no midifier", "no modifier"):
            continue
        score = sum(1 for t in query_tokens if t in label.lower())
        if score > best_score:
            best_score, best_sheet, best_label = score, sheet, label

    if best_score >= _MATCH_THRESHOLD:
        return best_sheet, best_label
    return None, None


def _parse_subtables(df: pd.DataFrame) -> list[dict]:
    """
    Parse all sub-tables from a sheet.

    Each sub-table begins at a row where col 2 == "Concentration of organic modifier (M)".
    Returns list of dicts: {dose_M, equiv, mg_per_L: {metal: [values]}}
    """
    subtables = []
    header_rows = df[df.iloc[:, 2].astype(str).str.contains(
        "Concentration of organic modifier", na=False
    )].index.tolist()

    for h in header_rows:
        # col 2 of the next row = modifier dose
        try:
            dose = float(df.iloc[h + 1, 2])
        except (ValueError, TypeError, IndexError):
            dose = None

        # Metal column names are in the same header row, cols 5 onward
        metal_cols = {}
        for col_idx in range(5, df.shape[1]):
            metal_name = str(df.iloc[h, col_idx]).strip()
            if metal_name and metal_name != "nan" and metal_name in _PIPELINE_METALS:
                metal_cols[col_idx] = metal_name

        if not metal_cols:
            continue

        # Data rows: from h+1 until next blank NaOH Equivs block
        equiv_list = []
        mg_data = {m: [] for m in metal_cols.values()}

        row = h + 1
        while row < len(df):
            equiv_val = df.iloc[row, 4]
            if pd.isna(equiv_val):
                break
            try:
                equiv = float(equiv_val)
            except (ValueError, TypeError):
                break
            equiv_list.append(round(equiv, 3))
            for col_idx, metal in metal_cols.items():
                try:
                    mg_data[metal].append(float(df.iloc[row, col_idx]))
                except (ValueError, TypeError):
                    mg_data[metal].append(np.nan)
            row += 1

        if equiv_list:
            subtables.append({
                "dose_M":    dose,
                "equiv":     equiv_list,
                "mg_per_L":  mg_data,
            })

    return subtables


def _select_subtable(subtables: list[dict], target_dose: float) -> dict | None:
    """Pick the sub-table with modifier dose closest to target_dose."""
    if not subtables:
        return None
    return min(
        (s for s in subtables if s["dose_M"] is not None),
        key=lambda s: abs(s["dose_M"] - target_dose),
        default=subtables[0],
    )


class ExperimentAgent(BaseAgent):
    name = "ExperimentAgent"

    def run(self, state: SimState) -> SimState:
        cfg          = state["run_config"]
        xlsx_path    = Path(cfg.get("experiment_xlsx", ""))
        modifier     = state["modifier_name"]
        target_dose  = cfg.get("modifier", {}).get("dose_mol", 0.20)

        if not xlsx_path.exists():
            print(f"[{self.name}] xlsx not found at '{xlsx_path}' — using run_config experiment block.")
            return state

        print(f"[{self.name}] Reading: {xlsx_path.name}")
        xl = pd.ExcelFile(xlsx_path)

        sheet, matched_label = _match_sheet(xl, modifier)
        if sheet is None:
            print(f"[{self.name}] No sheet matched '{modifier}' — using run_config experiment block.")
            return state

        print(f"[{self.name}] Matched sheet '{sheet}': modifier = '{matched_label}'")

        df        = pd.read_excel(xl, sheet_name=sheet, header=None)
        subtables = _parse_subtables(df)

        if not subtables:
            print(f"[{self.name}] No data parsed from sheet '{sheet}'.")
            return state

        best = _select_subtable(subtables, target_dose)
        print(f"[{self.name}] Selected dose {best['dose_M']:.4f} M "
              f"(target {target_dose} M) — {len(best['equiv'])} equiv points")

        # Build experiment block in the same format AnalystAgent expects
        experiment = {
            "source":    sheet,
            "equiv":     best["equiv"],
            "mg_per_L":  {m: v for m, v in best["mg_per_L"].items()},
        }

        # Inject into run_config so AnalystAgent picks it up transparently
        cfg["experiment"] = experiment
        state["run_config"] = cfg

        print(f"[{self.name}] Experiment loaded: {list(experiment['mg_per_L'].keys())} "
              f"at equiv {experiment['equiv']}")
        return state
