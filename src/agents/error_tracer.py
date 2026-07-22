"""
ErrorTracerAgent: diagnoses PHREEQC errors, applies targeted fixes to modifier
.phr files, and signals SimRunnerAgent to retry. Sits between SimRunnerAgent
and the error halt in the Conductor.

Supported error patterns:
  - Missing SOLUTION_MASTER_SPECIES  → adds master species block to .phr
  - Undefined reaction for species   → adds identity reaction to .phr
  - Species element not tabulated    → same fix as missing master species
"""

import re
from collections import Counter
from pathlib import Path

import sys; sys.path.insert(0, ".")
from src.agents.base import BaseAgent
from state import SimState

# Maps PHREEQC error patterns to fix strategies
ERROR_PATTERNS = [
    (
        re.compile(r"Elements in species have not been tabulated,\s+(\S+)"),
        "missing_master_species",
    ),
    (
        re.compile(r"Reaction for species has not been defined,\s+(\S+)"),
        "missing_identity_reaction",
    ),
]

_METAL_CHARGES = {"Cu": 2, "Co": 2, "Ni": 2, "Mn": 2, "Al": 3, "Fe": 3}
_CHARGE_IMBALANCE = re.compile(
    r"Equation is not charge balanced, right - left =\s+([-\d.]+)\s+moles charge\s*\n"
    r"ERROR: Equation for species (\S+) does not balance",
    re.MULTILINE,
)


def _phreeqc_charge(name: str) -> int:
    m = re.search(r"([+-]\d*)$", name)
    if not m:
        return 0
    s = m.group(1)
    return (1 if s == "+" else -1) if s in ("+", "-") else int(s)


def _fix_charge_balance(phr_text: str, error: str) -> tuple[str, list[str]]:
    """Fix charge-imbalanced SOLUTION_SPECIES reactions.

    Strategy:
      1. Parse each failing species and its imbalance from the error log.
      2. Infer the correct ligand charge from ML complexes (majority vote).
      3. Rewrite the master species name in SOLUTION_MASTER_SPECIES.
      4. Rewrite every reaction and product name in SOLUTION_SPECIES.
    """
    pairs = _CHARGE_IMBALANCE.findall(error)
    if not pairs:
        return phr_text, []

    # Extract current master species from SOLUTION_MASTER_SPECIES block
    ms_match = re.search(
        r"^SOLUTION_MASTER_SPECIES.*?^(\S+)\s+(\S+)\s+\S+\s+\S+",
        phr_text, re.MULTILINE | re.DOTALL
    )
    if not ms_match:
        return phr_text, []
    element     = ms_match.group(1)          # e.g. "Asp"
    current_ms  = ms_match.group(2)          # e.g. "Asp-1"
    current_Lc  = _phreeqc_charge(current_ms)

    # Infer correct Lc from ML reactions (those with exactly one L in left side)
    # For "M+mc + L{Lc} = product", correct_Lc = prod_charge - mc
    inferred = []
    for _imbalance, species_name in pairs:
        prod_charge = _phreeqc_charge(species_name)
        # Find metal charge from the reaction line in phr_text
        rxn_match = re.search(
            rf"(\S+)\+(\d)\s*\+\s*{re.escape(current_ms)}\s*=\s*{re.escape(species_name)}",
            phr_text
        )
        if rxn_match:
            mc = int(rxn_match.group(2))
            inferred.append(prod_charge - mc)

    if not inferred:
        return phr_text, []

    correct_Lc  = Counter(inferred).most_common(1)[0][0]
    if correct_Lc == current_Lc:
        return phr_text, []

    correct_ms = f"{element}{correct_Lc:+d}" if correct_Lc != 0 else element
    fixes = [f"Corrected master species charge: {current_ms} → {correct_ms}"]

    # Replace master species name everywhere in the file
    phr_text = phr_text.replace(current_ms, correct_ms)

    # Fix each product charge so reactions are charge-balanced
    for _imbalance, old_name in pairs:
        rxn_match = re.search(
            rf"(\S+)\+(\d)\s*\+\s*(?:2)?{re.escape(correct_ms)}\s*=\s*{re.escape(old_name)}",
            phr_text
        )
        if not rxn_match:
            continue
        mc = int(rxn_match.group(2))
        # Count how many L appear on the left (1 or 2)
        n_ligand = 2 if re.search(
            rf"2{re.escape(correct_ms)}", phr_text[:phr_text.index(old_name)]
        ) else 1
        expected_charge = mc + n_ligand * correct_Lc
        base    = re.sub(r"[+-]\d*$", "", old_name)
        new_name = base if expected_charge == 0 else f"{base}{expected_charge:+d}"
        if new_name != old_name:
            phr_text = phr_text.replace(old_name, new_name, 1)
            fixes.append(f"Product name: {old_name} → {new_name}")

    return phr_text, fixes


def _parse_species(species_str: str) -> tuple[str, int]:
    """Parse 'Asp-2' → ('Asp', -2),  'HAsp-1' → ('HAsp', -1), 'AlAsp+1' → ('AlAsp', 1)."""
    m = re.match(r"^([A-Za-z]+)([+-]\d+)?$", species_str)
    if not m:
        return species_str, 0
    name   = m.group(1)
    charge = int(m.group(2)) if m.group(2) else 0
    return name, charge


def _element_from_species(species_str: str) -> str:
    """Extract element name: 'Asp-2' → 'Asp'."""
    return re.match(r"^([A-Za-z]+)", species_str).group(1)


class ErrorTracerAgent(BaseAgent):
    name = "ErrorTracerAgent"

    def run(self, state: SimState) -> SimState:
        error = state.get("error", "")
        if not error or "PHREEQC failed" not in error:
            return state   # not a PHREEQC error — pass through

        print(f"[{self.name}] Analyzing PHREEQC error...")

        # Find the failing scenario's log file
        phr_path = Path(state["modifier_phr_path"])
        phr_text = phr_path.read_text()

        fixes_applied = []

        # ── Charge balance fix (highest priority — prevents all downstream errors) ──
        if "not charge balanced" in error or "does not balance" in error:
            phr_text, charge_fixes = _fix_charge_balance(phr_text, error)
            if charge_fixes:
                fixes_applied.extend(charge_fixes)
                for f in charge_fixes:
                    print(f"[{self.name}] Fix: {f}")

        for pattern, fix_type in ERROR_PATTERNS:
            matches = pattern.findall(error)
            for species_str in matches:
                element = _element_from_species(species_str)
                _, charge = _parse_species(species_str)
                charge_str = f"{charge:+d}" if charge != 0 else ""
                master = f"{element}{charge_str}"

                if fix_type == "missing_master_species":
                    if "SOLUTION_MASTER_SPECIES" not in phr_text:
                        # Estimate MW from modifier yaml formula field
                        modifier_yaml_path = Path(state["modifier_yaml_path"])
                        import yaml
                        mod = yaml.safe_load(modifier_yaml_path.read_text())["modifier"]
                        formula = mod.get("formula", "")
                        mw = _estimate_mw(formula)

                        master_block = (
                            f"\nSOLUTION_MASTER_SPECIES\n"
                            f"# Element  MasterSpecies  Alkalinity  Formula  GramFormWt\n"
                            f"{element}    {master}    0    {formula}    {mw}\n"
                        )
                        phr_text = master_block + phr_text
                        fixes_applied.append(f"Added SOLUTION_MASTER_SPECIES for {master}")
                        print(f"[{self.name}] Fix: added master species block for {master}")

                if fix_type in ("missing_master_species", "missing_identity_reaction"):
                    identity = f"{master} = {master}\n    log_k 0\n"
                    if identity not in phr_text and "SOLUTION_SPECIES" in phr_text:
                        phr_text = phr_text.replace(
                            "SOLUTION_SPECIES\n",
                            f"SOLUTION_SPECIES\n\n# Identity reaction (required by PHREEQC)\n{identity}\n",
                            1,
                        )
                        fixes_applied.append(f"Added identity reaction for {master}")
                        print(f"[{self.name}] Fix: added identity reaction for {master}")

        if fixes_applied:
            phr_path.write_text(phr_text)
            print(f"[{self.name}] Rewrote {phr_path} with {len(fixes_applied)} fix(es)")
            # Clear error and reset status so SimRunnerAgent retries
            state["error"]  = None
            state["status"] = "scouted"
            state["_error_tracer_fixes"] = fixes_applied
        else:
            print(f"[{self.name}] No automatic fix found for error:\n  {error}")

        return state


def _estimate_mw(formula: str) -> float:
    """Estimate molar mass from molecular formula string."""
    atomic_mass = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "S": 32.06}
    mw = 0.0
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        mw += atomic_mass.get(element, 0) * int(count if count else 1)
    return round(mw, 3)
