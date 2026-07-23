"""
ScoutAgent: given a modifier name, uses Claude to look up thermodynamic constants
(pKa, metal-ligand log K), generates modifier.yaml + modifier.phr, validates
against any previously tuned values in run_config.yaml, and writes run history.
"""

import json
import re
import yaml
from collections import Counter
from pathlib import Path
from datetime import datetime

import sys; sys.path.insert(0, ".")
from anthropic import Anthropic
from src.agents.base import BaseAgent
from state import SimState

_METAL_CHARGES = {"Cu": 2, "Co": 2, "Ni": 2, "Mn": 2, "Al": 3, "Fe": 3}


def _phreeqc_charge(name: str) -> int:
    """Parse charge from PHREEQC species name: 'CuAsp'→0, 'AlAsp+1'→1, 'CoAsp2-2'→-2."""
    m = re.search(r"([+-]\d*)$", name)
    if not m:
        return 0
    s = m.group(1)
    return (1 if s == "+" else -1) if s in ("+", "-") else int(s)


def _validate_charges(data: dict, agent_name: str = "ScoutAgent") -> dict:
    """Validate LLM-returned charge and complex phreeqc_names; auto-correct inconsistencies.

    Strategy:
      1. Infer correct ligand charge from ML complexes (majority vote).
      2. Recompute and fix every complex phreeqc_name to match.
    """
    complexes = data.get("complexes", [])
    declared  = int(data.get("charge", 0))

    # Step 1 — infer Lc from ML complexes (exclude Fe which is always suppressed)
    inferred = [
        _phreeqc_charge(c["phreeqc_name"]) - _METAL_CHARGES.get(c["metal"], 2)
        for c in complexes
        if c["type"] == "ML" and c.get("log_k") is not None and c["metal"] != "Fe"
    ]
    if not inferred:
        return data

    correct_Lc = Counter(inferred).most_common(1)[0][0]
    if correct_Lc != declared:
        print(f"[{agent_name}] Charge auto-correct: LLM returned {declared}, "
              f"inferred {correct_Lc} from {len(inferred)} ML complex names — fixing.")
        data = dict(data)
        data["charge"] = correct_Lc

    # Step 2 — fix every complex phreeqc_name to be charge-consistent
    _n_ligand = {"ML": 1, "ML2": 2, "MOHL": 1, "MOHL2": 2, "MOH2L": 1}
    _n_oh     = {"ML": 0, "ML2": 0, "MOHL": 1, "MOHL2": 1,  "MOH2L": 2}
    fixed = []
    for c in complexes:
        c  = dict(c)
        mc = _METAL_CHARGES.get(c["metal"], 2)
        nL = _n_ligand.get(c["type"], 1)
        nOH= _n_oh.get(c["type"], 0)
        expected_charge = mc + nL * correct_Lc - nOH
        base = re.sub(r"[+-]\d*$", "", c["phreeqc_name"])
        correct_name = base if expected_charge == 0 else f"{base}{expected_charge:+d}"
        if correct_name != c["phreeqc_name"]:
            print(f"[{agent_name}] Name fix: {c['phreeqc_name']} → {correct_name} "
                  f"({c['metal']} {c['type']})")
            c["phreeqc_name"] = correct_name
        fixed.append(c)
    data["complexes"] = fixed
    return data

def _repair_truncated_json(raw: str) -> dict | None:
    """Try to salvage a truncated JSON response by trimming the incomplete tail.

    Walks back from the end of the string to find the last complete object
    boundary ('}'), closes the complexes array and top-level object, then
    attempts a parse. Returns the parsed dict or None if repair fails.
    """
    # Find the last complete closing brace before the truncation point
    idx = raw.rfind("}")
    if idx == -1:
        return None
    candidate = raw[: idx + 1]
    # Count how many unclosed '[' and '{' remain and close them
    open_brackets = candidate.count("[") - candidate.count("]")
    open_braces   = candidate.count("{") - candidate.count("}")
    candidate += "]" * open_brackets + "}" * open_braces
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


SYSTEM_PROMPT = """\
You are a thermodynamic chemistry expert specializing in metal-ligand complexation.
Given an organic modifier name, return ONLY a JSON object with these fields:

  - "slug": short snake_case identifier (e.g. "l_aspartate")
  - "phreeqc_name": PHREEQC master species symbol (e.g. "Asp", "Edta", "Cit")
  - "formula": neutral molecular formula of the free acid form
  - "charge": integer charge of the fully deprotonated master species (negative)
  - "na_stoich": number of Na+ per formula unit in the sodium salt form (e.g. 1 for monosodium)
  - "denticity": number of donor atoms
  - "pka": list of pKa values at 25C, I=0, ascending order (for the free acid)
  - "complexes": list of all known metal complexes. Each entry:
      {
        "metal":        one of Cu, Co, Ni, Mn, Al, Fe,
        "type":         one of "ML" | "ML2" | "MOHL" | "MOHL2" | "MOH2L",
        "phreeqc_name": PHREEQC species name including charge suffix (e.g. "CoAsp", "CoAsp2-2", "CoOHAsp-"),
        "log_k":        cumulative log_beta at 25C, I=0 (null if unknown),
        "source":       brief citation
      }
      Types:
        ML    = M + L         (1:1 complex)
        ML2   = M + 2L        (1:2 bis-complex)
        MOHL  = M + L + OH-   (ternary hydroxo, written as M + L - H+ + H2O in PHREEQC)
        MOHL2 = M + 2L + OH-  (ternary hydroxo bis)
        MOH2L = M + L + 2OH-  (di-hydroxo ternary, written as M + L - 2H+ + 2H2O)
      Include all types for which log_k data exist in NIST or peer-reviewed literature.
      Omit a complex if no literature value is available — do not guess.
  - "source": primary reference for the dataset

Return only the JSON. No explanation, no markdown fences.
"""


class ScoutAgent(BaseAgent):
    name = "ScoutAgent"

    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.client = Anthropic()
        self.model  = model

    def run(self, state: SimState) -> SimState:
        cfg     = state["run_config"]
        metals  = cfg["leachate"]["metals"]
        s_cfg   = cfg.get("scout", {})
        threshold    = s_cfg.get("flag_discrepancy_threshold", 0.5)
        auto_correct = s_cfg.get("auto_correct", False)

        print(f"[{self.name}] Scouting: {state['modifier_name']}")

        # ── 1. Query Claude (LLM) for structure, pKa, and any gaps left by DataAgent ──
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["modifier_name"]}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present (```json ... ``` or ``` ... ```)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            repaired = _repair_truncated_json(raw)
            if repaired is not None:
                print(f"[{self.name}] JSON truncated — repaired by dropping incomplete tail")
                data = repaired
            else:
                state["error"] = f"ScoutAgent: JSON parse failed — {e}\nRaw: {raw}"
                return state

        print(f"[{self.name}] LLM retrieved: {json.dumps(data, indent=2)}")
        data = _validate_charges(data, self.name)

        # ── 1b. Override LLM log_k values with authoritative DB values from DataAgent ──
        db_constants = state.get("db_constants")
        if db_constants:
            _type_map = {"ML": "ML", "ML2": "ML2", "MOHL": "MOHL", "MHL": "MHL"}
            n_overrides = 0
            for cplx in data.get("complexes", []):
                metal = cplx["metal"]
                ctype = cplx.get("type")
                db_entry = db_constants.get(metal, {}).get(_type_map.get(ctype, ""))
                if db_entry and db_entry["log_k"] is not None:
                    old = cplx.get("log_k")
                    cplx["log_k"] = db_entry["log_k"]
                    cplx["source"] = "NIST SRD 46 v8.0 (local DB)"
                    if db_entry["uncertain"]:
                        cplx["source"] += " [uncertain]"
                    if old != db_entry["log_k"]:
                        print(f"[{self.name}] DB override {metal} {ctype}: "
                              f"{old} → {db_entry['log_k']}"
                              + (" [uncertain]" if db_entry["uncertain"] else ""))
                        n_overrides += 1
            print(f"[{self.name}] DB overrides applied: {n_overrides}")

        # ── 2. Validate against any existing tuned values in run_config ────────
        flags     = []
        complexes = data.get("complexes", [])
        # Build flat log_k dict from complexes list for backward-compat validation
        log_k  = {c["metal"]: c["log_k"] for c in complexes
                  if c["type"] == "ML" and c.get("log_k") is not None}
        log_k2 = {c["metal"]: c["log_k"] for c in complexes
                  if c["type"] == "ML2" and c.get("log_k") is not None}
        runs   = cfg.get("runs", [])

        # Find most recent run for this modifier if it exists
        prior = next((r for r in reversed(runs)
                      if r.get("modifier_slug") == data["slug"]), None)

        if prior:
            for metal in metals:
                lit_val   = log_k.get(metal)
                tuned_val = prior.get("log_k_tuned", {}).get(metal)
                if lit_val is not None and tuned_val is not None:
                    diff = abs(lit_val - tuned_val)
                    if diff > threshold:
                        flags.append({
                            "metal":     metal,
                            "literature": lit_val,
                            "tuned":      tuned_val,
                            "delta":      round(diff, 3),
                            "action":     "corrected" if auto_correct else "flagged",
                        })
                        print(f"[{self.name}] FLAG {metal}: literature={lit_val}, "
                              f"tuned={tuned_val}, delta={diff:.3f}")
                        if auto_correct:
                            log_k[metal] = lit_val

        # ── 3. Write modifier.yaml / .phr (skip if files exist and not force-scouting) ──
        slug         = data["slug"]
        phreeqc_name = data["phreeqc_name"]
        charge       = int(data["charge"])
        pkas         = data.get("pka", [])
        yaml_path = Path(f"config/modifiers/{slug}.yaml")
        phr_path  = Path(f"database/modifiers/{slug}.phr")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        phr_path.parent.mkdir(parents=True, exist_ok=True)

        force_scout = state.get("force_scout", False)
        files_exist = yaml_path.exists() and phr_path.exists()

        if files_exist and not force_scout:
            print(f"[{self.name}] Existing files found — preserving tuned .phr and .yaml "
                  f"(pass force_scout=True to regenerate from LLM values).")
            state["modifier_yaml_path"] = str(yaml_path)
            state["modifier_phr_path"]  = str(phr_path)
            state["descriptor"]         = {}
            state["scout_flags"]        = flags
            state["status"]             = "scouted"
            state["error"]              = None
            # Still write the run entry so history is tracked
            run_entry = {
                "modifier_slug": slug,
                "modifier_name": state["modifier_name"],
                "timestamp":     datetime.utcnow().isoformat(),
                "status":        "scouted",
                "descriptor":    {},
                "log_k_tuned":   {m: log_k.get(m) for m in metals},
                "scout_flags":   flags,
                "scorecard":     None,
            }
            cfg["runs"].append(run_entry)
            Path(state["config_path"]).write_text(
                yaml.dump(cfg, default_flow_style=False, sort_keys=False)
            )
            return state

        # na_counter_ions from explicit salt stoichiometry (not ligand charge)
        na_stoich    = int(data.get("na_stoich", abs(charge)))
        na_counter   = round(na_stoich * 0.20, 4)
        phreeqc_elem = data["phreeqc_name"]

        # Build selected_molalities: master + protonated forms + all complex names
        protonated = " ".join(
            f"H{i}{phreeqc_elem}{charge+i:+d}" if (charge+i) != 0
            else f"H{i}{phreeqc_elem}"
            for i in range(1, len(pkas)+1)
        )
        complex_names = " ".join(
            c["phreeqc_name"] for c in complexes
            if c.get("log_k") is not None and c["metal"] != "Fe"
        )
        selected_mol = " ".join(
            filter(None, [f"{phreeqc_elem}{charge}", protonated, complex_names])
        ).strip()

        modifier_yaml = {
            "modifier": {
                "name":                slug,
                "full_name":           state["modifier_name"],
                "formula":             data["formula"],
                "database":            str(phr_path),
                "dose_mol":            0.20,
                "phreeqc_element":     phreeqc_elem,
                "na_counter_ions":     na_counter,
                "selected_molalities": selected_mol,
            }
        }
        yaml_path.write_text(yaml.dump(modifier_yaml, default_flow_style=False))
        print(f"[{self.name}] Written: {yaml_path}")

        # ── 4. Write modifier.phr ──────────────────────────────────────────────

        # Compute molar mass of master species (fully deprotonated form)
        # MW(master) = MW(neutral formula) - n_removed_H * 1.008
        # n_removed_H = len(pkas) protons removed from fully protonated cation
        # fully protonated carries charge +1, master species carries `charge`
        # so n_removed = 1 - charge (e.g. charge=-2 → removed 3 H from +1 form)
        n_removed = 1 - charge
        mw_neutral = sum(
            {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "S": 32.06}
            .get(c, 0) * int(n if n else 1)
            for c, n in __import__("re").findall(r"([A-Z][a-z]?)(\d*)", data["formula"])
        )
        mw_master = round(mw_neutral - n_removed * 1.008, 3)

        lines = [
            f"# Thermodynamic data for {state['modifier_name']}",
            f"# Source: {data.get('source', 'LLM-assisted lookup — verify before publication')}",
            "",
            "SOLUTION_MASTER_SPECIES",
            f"# Element  MasterSpecies  Alkalinity  Formula  GramFormWt",
            f"{phreeqc_name}    {phreeqc_name}{charge}    0    {data['formula']}    {mw_master}",
            "",
            "SOLUTION_SPECIES",
            "",
            "# Identity reaction (required by PHREEQC for master species)",
            f"{phreeqc_name}{charge} = {phreeqc_name}{charge}",
            f"    log_k 0",
            "",
            "# Protonation steps",
        ]

        for i, pka in enumerate(pkas, 1):
            product_charge = charge + i
            h_prefix  = f"H{i}" if i > 1 else "H"
            sign      = f"{product_charge:+d}" if product_charge != 0 else ""
            product   = f"{h_prefix}{phreeqc_name}{sign}"
            lines    += [
                f"{phreeqc_name}{charge} + {i}H+ = {product}",
                f"    log_k {sum(pkas[:i]):.2f}",
                "",
            ]

        # Fe is always suppressed: Fe(OH)3 precipitation is kinetically irreversible
        # on HTE timescales regardless of modifier. Universal framework assumption.
        SUPPRESSED_METALS = {"Fe"}

        # PHREEQC reaction templates keyed by complex type
        # L = phreeqc_name+charge, M = metal+mc
        def _reaction(ctype, metal, mc, L, Lc, name):
            M = f"{metal}+{mc}"
            if ctype == "ML":
                return f"{M} + {L}{Lc} = {name}"
            if ctype == "ML2":
                return f"{M} + 2{L}{Lc} = {name}"
            if ctype == "MOHL":
                return f"{M} + {L}{Lc} - H+ + H2O = {name}"
            if ctype == "MOHL2":
                return f"{M} + 2{L}{Lc} - H+ + H2O = {name}"
            if ctype == "MOH2L":
                return f"{M} + {L}{Lc} - 2H+ + 2H2O = {name}"
            return None

        metal_charge_map = {"Cu": 2, "Co": 2, "Ni": 2, "Mn": 2, "Al": 3, "Fe": 3}
        lines.append("# Metal complexes (all types)")
        for cplx in complexes:
            metal = cplx["metal"]
            ctype = cplx["type"]
            cname = cplx["phreeqc_name"]
            lk    = cplx.get("log_k")
            if lk is None:
                continue
            mc  = metal_charge_map.get(metal, 2)
            rxn = _reaction(ctype, metal, mc, phreeqc_name, charge, cname)
            if rxn is None:
                continue
            if metal in SUPPRESSED_METALS:
                lines += [
                    f"# {metal} suppressed: Fe(OH)3 kinetically irreversible on HTE timescales.",
                    f"{rxn}",
                    f"    log_k -999",
                    "",
                ]
            else:
                lines += [
                    f"{rxn}",
                    f"    log_k {lk:.2f}",
                    "",
                ]

        phr_path.write_text("\n".join(lines))
        print(f"[{self.name}] Written: {phr_path}")

        # ── 5. Build descriptor vector for BO ─────────────────────────────────
        descriptor = {
            "denticity": data.get("denticity"),
            "charge":    charge,
            "n_pka":     len(pkas),
            **{f"log_k_{m}": log_k.get(m) for m in metals},
            **{f"log_k2_{m}": log_k2.get(m) for m in metals},
        }

        # ── 6. Write back run entry to run_config.yaml ────────────────────────
        run_entry = {
            "modifier_slug":  slug,
            "modifier_name":  state["modifier_name"],
            "timestamp":      datetime.utcnow().isoformat(),
            "status":         "scouted",
            "descriptor":     descriptor,
            "log_k_tuned":    {m: log_k.get(m) for m in metals},
            "scout_flags":    flags,
            "scorecard":      None,   # filled by AnalystAgent
        }
        cfg["runs"].append(run_entry)
        config_path = Path(state["config_path"])
        config_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        print(f"[{self.name}] Run entry written to {config_path}")

        state["modifier_yaml_path"] = str(yaml_path)
        state["modifier_phr_path"]  = str(phr_path)
        state["descriptor"]         = descriptor
        state["scout_flags"]        = flags
        state["status"]             = "scouted"
        state["error"]              = None
        return state
