"""
DataAgent: queries NIST_SRD_46_ported.db for metal-ligand formation constants.

Returns a constants dict in the same schema ScoutAgent expects from the LLM,
so ScoutAgent can skip its LLM call when full DB coverage is available.
Missing entries are returned with log_k=null so ScoutAgent can fill gaps via LLM.
"""

import re
import sqlite3
from pathlib import Path

import sys; sys.path.insert(0, ".")
from src.agents.base import BaseAgent
from state import SimState

DB_PATH = Path("database/modifiers/NIST_SRD_46_ported.db")

# Map internal metal symbols → HTML names used in the DB
_METAL_TO_DB = {
    "Cu": "Cu<sup>2+</sup>",
    "Co": "Co<sup>2+</sup>",
    "Ni": "Ni<sup>2+</sup>",
    "Mn": "Mn<sup>2+</sup>",
    "Al": "Al<sup>3+</sup>",
    "Fe": "Fe<sup>3+</sup>",
}

# Map beta_definition HTML strings → our type codes
_BD_TO_TYPE = {
    "[ML]/[M][L]":                              "ML",
    "[ML<sub>2</sub>]/[M][L]<sup>2</sup>":      "ML2",
    "[ML<sub>3</sub>]/[M][L]<sup>3</sup>":      "ML3",
    "[MHL]/[ML][H]":                            "MHL",   # protonated
    "[M(OH)L]/[M][OH][L]":                      "MOHL",
    "[M(OH)L]/[ML][OH]":                        "MOHL",
}

_METAL_CHARGES = {"Cu": 2, "Co": 2, "Ni": 2, "Mn": 2, "Al": 3, "Fe": 3}


def _strip_html(text: str) -> str:
    """Remove HTML tags from DB strings."""
    return re.sub(r"<[^>]+>", "", text)


def _parse_constant(raw: str):
    """Parse DB constant string; returns (value, uncertain) tuple.

    NIST flags uncertain values with parentheses, e.g. '(3.7)'.
    Returns (float, True) for uncertain, (float, False) for certain, (None, False) if unparseable.
    """
    if not raw or raw.strip() in ("*", "", "0"):
        return None, False
    s = raw.strip()
    uncertain = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        return float(s), uncertain
    except ValueError:
        return None, False


def _find_ligand(conn: sqlite3.Connection, modifier_name: str) -> tuple[int | None, str | None]:
    """Resolve modifier name to (ligandenID, matched_name) using tiered matching."""
    cur = conn.cursor()

    # 1. Exact match on name_ligand
    cur.execute("SELECT ligandenID, name_ligand FROM liganden WHERE name_ligand = ?",
                (modifier_name,))
    row = cur.fetchone()
    if row:
        return row[0], row[1]

    # 2. Parenthetical alias match — NIST often stores "Full Name (Common Name)"
    #    Try matching the portion inside parentheses against the full modifier name
    cur.execute("SELECT ligandenID, name_ligand FROM liganden WHERE name_ligand LIKE ?",
                (f"%{modifier_name}%",))
    row = cur.fetchone()
    if row:
        return row[0], row[1]

    # 3. Token-based: score by how many words from modifier_name appear in name_ligand
    words = [w.lower() for w in re.split(r"[\s\-(),]+", modifier_name) if len(w) > 3]
    if not words:
        return None, None
    cur.execute("SELECT ligandenID, name_ligand FROM liganden")
    best_id, best_name, best_score = None, None, 0
    for lid, lname in cur.fetchall():
        lname_lower = lname.lower()
        score = sum(1 for w in words if w in lname_lower)
        if score > best_score:
            best_score, best_id, best_name = score, lid, lname
    if best_score >= max(2, len(words) // 2):
        return best_id, best_name
    return None, None


def _query_constants(conn: sqlite3.Connection, ligand_id: int,
                     metals: list[str]) -> dict:
    """
    Query all ML/ML2/MHL/MOHL constants at 25C for the given metals.

    Returns dict keyed by metal symbol:
      { "ML": (log_k, uncertain), "ML2": ..., "MHL": ..., "MOHL": ... }
    """
    db_metals = [_METAL_TO_DB[m] for m in metals if m in _METAL_TO_DB]
    placeholders = ",".join("?" * len(db_metals))

    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            m.name_metal,
            bd.name_beta_definition,
            v.ionicstrength,
            v.constant
        FROM verkn_ligand_metal v
        JOIN metal m        ON m.metalID        = v.metalNr
        JOIN beta_definition bd ON bd.beta_definitionID = v.beta_definitionNr
        JOIN constanttyp ct ON ct.constanttypID = v.constanttypNr
        WHERE v.ligandenNr = ?
          AND m.name_metal IN ({placeholders})
          AND ct.name_constanttyp = 'K'
          AND v.temperature = '25'
        ORDER BY m.name_metal, bd.name_beta_definition, CAST(v.ionicstrength AS REAL)
    """, [ligand_id, *db_metals])

    rows = cur.fetchall()

    # Reverse-map DB metal name → symbol
    _db_to_symbol = {v: k for k, v in _METAL_TO_DB.items()}

    # Collect candidates: {symbol: {type: [(ionic_strength, value, uncertain)]}}
    from collections import defaultdict
    candidates = defaultdict(lambda: defaultdict(list))

    for db_metal, db_bd, ionic_str, raw_const in rows:
        symbol = _db_to_symbol.get(db_metal)
        if not symbol:
            continue
        ctype = _BD_TO_TYPE.get(db_bd)
        if not ctype:
            continue
        val, uncertain = _parse_constant(raw_const)
        if val is None:
            continue
        try:
            I = float(ionic_str)
        except (TypeError, ValueError):
            I = 999.0
        candidates[symbol][ctype].append((I, val, uncertain))

    # Select canonical value: prefer I closest to 0, then lowest I
    result = {}
    for symbol, types in candidates.items():
        result[symbol] = {}
        for ctype, entries in types.items():
            entries.sort(key=lambda x: x[0])
            I, val, uncertain = entries[0]
            result[symbol][ctype] = {
                "log_k":      val,
                "uncertain":  uncertain,
                "ionic_str":  I,
                "confidence": "db_uncertain" if uncertain else "db_exact",
            }

    return result


def _build_phreeqc_name(metal: str, ctype: str, phreeqc_elem: str,
                         ligand_charge: int) -> str:
    """Construct PHREEQC species name from metal, complex type, and ligand charge."""
    mc = _METAL_CHARGES.get(metal, 2)
    n_L  = {"ML": 1, "ML2": 2, "ML3": 3, "MOHL": 1, "MHL": 1}.get(ctype, 1)
    n_OH = {"MOHL": 1}.get(ctype, 0)
    n_H  = {"MHL": -1}.get(ctype, 0)   # protonated: one H added back

    net = mc + n_L * ligand_charge - n_OH + n_H
    suffix = "" if net == 0 else f"{net:+d}"

    n_str = "" if n_L == 1 else str(n_L)
    oh_str = "OH" if n_OH == 1 else ""
    h_str  = "H"  if n_H == -1 else ""

    base = f"{metal}{oh_str}{h_str}{phreeqc_elem}{n_str}"
    return f"{base}{suffix}"


class DataAgent(BaseAgent):
    name = "DataAgent"

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def run(self, state: SimState) -> SimState:
        if not self.db_path.exists():
            print(f"[{self.name}] DB not found at {self.db_path} — skipping.")
            state["db_constants"] = None
            return state

        modifier_name = state["modifier_name"]
        metals        = state["run_config"]["leachate"]["metals"]

        print(f"[{self.name}] Querying DB for: {modifier_name}")

        conn = sqlite3.connect(self.db_path)
        try:
            ligand_id, matched_name = _find_ligand(conn, modifier_name)

            if ligand_id is None:
                print(f"[{self.name}] No DB match found — LLM fallback required.")
                state["db_constants"] = None
                return state

            print(f"[{self.name}] Matched ligand: '{matched_name}' (id={ligand_id})")
            constants = _query_constants(conn, ligand_id, metals)
        finally:
            conn.close()

        # Report coverage
        covered   = [m for m in metals if constants.get(m)]
        missing   = [m for m in metals if not constants.get(m)]
        print(f"[{self.name}] DB coverage: {covered} | missing: {missing}")

        for metal, types in constants.items():
            for ctype, entry in types.items():
                flag = " [UNCERTAIN]" if entry["uncertain"] else ""
                print(f"[{self.name}]   {metal} {ctype}: log_k={entry['log_k']:.2f} "
                      f"(I={entry['ionic_str']}){flag}")

        state["db_constants"]    = constants
        state["db_ligand_match"] = matched_name
        return state
