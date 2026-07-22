"""
SimRunnerAgent: generates PHREEQC inputs and runs all scenarios defined
in run_config.yaml. Scenarios, paths, and environment are fully config-driven.
"""

import os
import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, ".")
sys.path.insert(0, "src/pipeline")

from src.agents.base import BaseAgent
from state import SimState


def _resolve_phases(phases: list, run_config: dict) -> list:
    """Translate generic phase names to database-specific names using phase_aliases."""
    db_name   = os.path.basename(run_config.get("environment", {}).get("database", ""))
    alias_map = run_config.get("phase_aliases", {}).get(db_name, {})
    resolved  = []
    for p in phases:
        mapped = alias_map.get(p, p)
        if mapped != p:
            print(f"[SimRunnerAgent] Phase alias: {p!r} → {mapped!r}")
        resolved.append(mapped)
    return resolved


class SimRunnerAgent(BaseAgent):
    name = "SimRunnerAgent"

    def run(self, state: SimState) -> SimState:
        if state.get("status") != "scouted":
            state["error"] = "SimRunnerAgent: expected status 'scouted'"
            return state

        cfg          = state["run_config"]
        env_cfg      = cfg["environment"]
        scenarios    = cfg["scenarios"]
        phreeqc_exe  = env_cfg["phreeqc_exe"]
        database     = env_cfg["database"]

        from generate_input import load_yaml, assemble_database
        from run_phreeqc import run as phreeqc_run

        modifier = load_yaml(state["modifier_yaml_path"])["modifier"]
        sim_paths = {}

        jinja_env = Environment(
            loader=FileSystemLoader("templates"),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        for scenario, leachate_yaml in scenarios.items():
            print(f"[{self.name}] Running scenario: {scenario}")
            leachate = load_yaml(leachate_yaml)

            db_header = assemble_database(
                phases_path="database/phases.phr",
                solution_species_path="database/solution_species.phr",
                modifier_db_path=modifier["database"],
            )

            slug    = modifier["name"]
            out_dir = Path(f"experiments/single/{slug}_{scenario}")
            out_dir.mkdir(parents=True, exist_ok=True)

            resolved_phases = _resolve_phases(
                leachate["equilibrium_phases"], cfg
            )
            context = {
                "leachate":           leachate["leachate"],
                "neutralization":     leachate["neutralization"],
                "equilibrium_phases": resolved_phases,
                "knobs":              leachate["knobs"],
                "modifier":           modifier,
                "o2_reservoir_mol":   leachate.get("o2_reservoir_mol", 10.0),
            }
            rendered  = jinja_env.get_template("neutralization.phr.j2").render(**context)
            inp_path  = out_dir / "input.phr"
            inp_path.write_text(db_header + "\n\n" + rendered)

            try:
                phreeqc_run(inp_path, phreeqc_exe=phreeqc_exe, database=database)
                sim_paths[scenario] = str(out_dir / "neutralization.csv")
            except RuntimeError as e:
                state["error"] = f"SimRunnerAgent: PHREEQC failed on '{scenario}' — {e}"
                return state

        state["sim_paths"] = sim_paths
        state["status"]    = "simulated"
        state["error"]     = None
        return state
