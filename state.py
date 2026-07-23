"""
Shared state definition for the MDPSA LangGraph pipeline.
All agents read from and write to SimState.
"""

from typing import TypedDict, Optional


class SimState(TypedDict):
    # ── Runtime input ──────────────────────────────────────────────────────────
    modifier_name:  str             # e.g. "L-aspartic acid monosodium salt"
    config_path:    str             # path to run_config.yaml
    run_config:     dict            # loaded run_config.yaml contents (read by all agents)
    force_scout:    Optional[bool]  # if True, regenerate .phr/.yaml even if files exist

    # ── DataAgent outputs ──────────────────────────────────────────────────────
    db_constants:    Optional[dict]  # {metal: {type: {log_k, uncertain, confidence}}}
    db_ligand_match: Optional[str]   # matched ligand name in NIST DB

    # ── ScoutAgent outputs ─────────────────────────────────────────────────────
    modifier_yaml_path:  Optional[str]   # config/modifiers/<slug>.yaml
    modifier_phr_path:   Optional[str]   # database/modifiers/<slug>.phr
    descriptor:          Optional[dict]  # thermodynamic feature vector for BO
    scout_flags:         Optional[list]  # discrepancies flagged vs. literature

    # ── SimRunnerAgent outputs ─────────────────────────────────────────────────
    sim_paths: Optional[dict]       # {"open": path, "closed": path, "partial": path}

    # ── AnalystAgent outputs ───────────────────────────────────────────────────
    scorecard:                    Optional[dict]  # per-metal protection scores
    figure_path:                  Optional[str]   # path to model_vs_exp figure
    species_contribution_path:    Optional[str]   # path to species contribution figure

    # ── BeamerAgent outputs ────────────────────────────────────────────────────
    beamer_pdf:   Optional[str]   # path to compiled PDF presentation

    # ── BOAgent outputs ────────────────────────────────────────────────────────
    next_modifier:  Optional[str]   # next modifier name to evaluate (BO loop)
    bo_suggestion:  Optional[dict]  # proposed config updates from BO

    # ── Conductor control ──────────────────────────────────────────────────────
    status:   str                   # "pending"|"scouted"|"simulated"|"analysed"|"done"
    error:    Optional[str]
    messages: list

    # ── ErrorTracerAgent ───────────────────────────────────────────────────────
    _sim_retries:          Optional[int]    # retry counter for SimRunner
    _error_tracer_fixes:   Optional[list]   # list of fixes applied
