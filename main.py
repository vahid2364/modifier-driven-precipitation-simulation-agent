"""
Entry point for the modifier-driven precipitation simulation agent.

Usage:
    python main.py --modifier "L-aspartic acid monosodium salt monohydrate"
    python main.py --modifier "sodium citrate" --config config/run_config.yaml
"""

import argparse
import yaml
from conductor import build_conductor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modifier", required=True,
        help="Full modifier name (e.g. 'L-aspartic acid monosodium salt monohydrate')"
    )
    parser.add_argument(
        "--config", default="config/run_config.yaml",
        help="Path to run_config.yaml (default: config/run_config.yaml)"
    )
    parser.add_argument(
        "--force-scout", action="store_true",
        help="Regenerate modifier .phr and .yaml from LLM even if files already exist"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        run_config = yaml.safe_load(f)

    initial_state = {
        "modifier_name":      args.modifier,
        "config_path":        args.config,
        "force_scout":        args.force_scout,
        "run_config":         run_config,
        "modifier_yaml_path": None,
        "modifier_phr_path":  None,
        "descriptor":         None,
        "scout_flags":        None,
        "sim_paths":          None,
        "scorecard":                 None,
        "figure_path":               None,
        "species_contribution_path": None,
        "beamer_pdf":                None,
        "next_modifier":      None,
        "bo_suggestion":      None,
        "status":             "pending",
        "error":              None,
        "messages":           [],
        "_sim_retries":       0,
        "_error_tracer_fixes": None,
    }

    conductor = build_conductor()

    print(f"\n{'='*60}")
    print(f"  MDPSA")
    print(f"  Modifier : {args.modifier}")
    print(f"  Config   : {args.config}")
    print(f"{'='*60}\n")

    final_state = conductor.invoke(initial_state)

    if final_state.get("error"):
        print(f"\nPipeline failed: {final_state['error']}")
        return

    # ── Print flags from ScoutAgent ───────────────────────────────────────────
    flags = final_state.get("scout_flags") or []
    if flags:
        print(f"\n{'='*60}")
        print("  SCOUT FLAGS (literature vs tuned log K discrepancies)")
        print(f"{'='*60}")
        for f in flags:
            print(f"  {f['metal']:2s}  literature={f['literature']}  "
                  f"tuned={f['tuned']}  delta={f['delta']}  [{f['action']}]")

    # ── Print scorecard ───────────────────────────────────────────────────────
    scorecard = final_state.get("scorecard", {})
    cfg       = final_state["run_config"]
    equiv_pts = cfg["analyst"]["comparison_equiv"]

    print(f"\n{'='*60}")
    print("  SELECTIVITY SCORECARD")
    print(f"{'='*60}")
    header = "  Metal  " + "  ".join(f"@ {eq} eq" for eq in equiv_pts) + "  Result"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metal, s in scorecard.items():
        vals = "  ".join(f"{s[f'pct_at_{eq}_equiv']:5.1f}%" for eq in equiv_pts)
        if s.get("role") == "target":
            result = ("PROTECTED" if s["score"] == s["max_score"]
                      else "PARTIAL"   if s["score"] > 0
                      else "LOST")
        else:
            result = ("REMOVED"   if s["score"] == s["max_score"]
                      else "PARTIAL"   if s["score"] > 0
                      else "STAYS DISSOLVED")
        print(f"  {metal:2s}     {vals}  [{result}]")

    print(f"\n  Figure : {final_state.get('figure_path')}")
    print(f"  Config : {args.config}  (updated with run history)")
    print(f"  Status : {final_state.get('status')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
