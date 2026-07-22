"""
BOAgent: Phase 2 Bayesian Optimization agent.

Reads scorecard history and descriptor vectors from run_config.yaml,
fits a GP surrogate, proposes the next modifier to evaluate, and writes
the proposal back to run_config.yaml before looping to ScoutAgent.

Phase 2 dependencies (uncomment in requirements.txt when ready):
    botorch
    gpytorch
"""

import yaml
from pathlib import Path
from datetime import datetime

import sys; sys.path.insert(0, ".")
from src.agents.base import BaseAgent
from state import SimState


class BOAgent(BaseAgent):
    name = "BOAgent"

    def run(self, state: SimState) -> SimState:
        cfg      = state["run_config"]
        runs     = cfg.get("runs", [])
        cfg_path = Path(state["config_path"])

        # Phase 2: replace stub body below with BoTorch GP + acquisition function
        # ── GP inputs ─────────────────────────────────────────────────────────
        # X = np.array([r["descriptor"].values() for r in runs if r.get("scorecard")])
        # y = np.array([sum(s["score"] for s in r["scorecard"].values()) for r in runs ...])
        # ── Fit GP + optimize acquisition ─────────────────────────────────────
        # model = SingleTaskGP(X_tensor, y_tensor)
        # acqf  = ExpectedImprovement(model, best_f=y.max())
        # next_x, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
        # next_modifier_name = descriptor_to_name(next_x)   # inverse lookup or LLM call
        # ──────────────────────────────────────────────────────────────────────

        n_runs = len([r for r in runs if r.get("scorecard")])
        print(f"[{self.name}] {n_runs} completed run(s) in history. BO not yet active.")

        # Write BO attempt to run_config for traceability
        bo_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_runs":    n_runs,
            "status":    "stub — Phase 2 not implemented",
        }
        cfg.setdefault("bo_log", []).append(bo_log)
        cfg_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))

        state["next_modifier"] = None   # set to modifier name in Phase 2 to loop
        state["bo_suggestion"] = None
        state["status"]        = "done"
        return state
