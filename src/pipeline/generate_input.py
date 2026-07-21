"""
Generate a self-contained PHREEQC input file for a single modifier simulation.

Usage:
    python src/generate_input.py \
        --leachate config/nmc532_baseline.yaml \
        --modifier config/modifiers/EDTA.yaml \
        --naoh-dose 0.5 \
        --modifier-dose 0.05 \
        --output experiments/single/EDTA_dose0p05_NaOH0p5/input.phr
"""

import argparse
import yaml
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_phr(path):
    with open(path) as f:
        return f.read()


def extract_block(text, keyword):
    """Extract content lines belonging to a keyword block (e.g. SOLUTION_SPECIES)."""
    lines = text.splitlines()
    inside = False
    collected = []
    top_keywords = {"PHASES", "SOLUTION_MASTER_SPECIES", "SOLUTION_SPECIES",
                    "KNOBS", "SOLUTION", "REACTION", "EQUILIBRIUM_PHASES",
                    "SELECTED_OUTPUT", "SAVE", "USE", "END", "TITLE"}
    for line in lines:
        stripped = line.strip()
        token = stripped.split()[0].upper() if stripped else ""
        if token == keyword:
            inside = True
            continue
        if inside and token in top_keywords and token != keyword:
            inside = False
        if inside:
            collected.append(line)
    return "\n".join(collected).strip()


def assemble_database(phases_path, solution_species_path, modifier_db_path):
    """Merge PHASES, SOLUTION_MASTER_SPECIES, and all SOLUTION_SPECIES blocks."""
    phases_text   = load_phr(phases_path)
    sol_sp_text   = load_phr(solution_species_path)
    modifier_text = load_phr(modifier_db_path)

    # Collect modifier SOLUTION_MASTER_SPECIES (may be empty for some modifiers)
    master_species = extract_block(modifier_text, "SOLUTION_MASTER_SPECIES")

    # Collect all SOLUTION_SPECIES content into one block
    all_species = "\n\n".join([
        extract_block(sol_sp_text,   "SOLUTION_SPECIES"),
        extract_block(modifier_text, "SOLUTION_SPECIES"),
    ])

    parts = [phases_text.strip()]
    if master_species.strip():
        parts.append("SOLUTION_MASTER_SPECIES\n\n" + master_species)
    parts.append("SOLUTION_SPECIES\n\n" + all_species)

    return "\n\n".join(parts)


def render_template(template_dir, template_file, context):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template(template_file)
    return tmpl.render(**context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leachate",       default="config/nmc532_baseline.yaml")
    parser.add_argument("--modifier",       default="config/modifiers/EDTA.yaml")
    parser.add_argument("--naoh-dose",      type=float, default=None)
    parser.add_argument("--modifier-dose",  type=float, default=None)
    parser.add_argument("--output",         default=None)
    args = parser.parse_args()

    leachate = load_yaml(args.leachate)
    modifier = load_yaml(args.modifier)["modifier"]

    # Override doses if provided
    if args.naoh_dose is not None:
        leachate["neutralization"]["naoh_dose_mol"] = args.naoh_dose
    if args.modifier_dose is not None:
        modifier["dose_mol"] = args.modifier_dose

    # Assemble database header
    db_header = assemble_database(
        phases_path="database/phases.phr",
        solution_species_path="database/solution_species.phr",
        modifier_db_path=modifier["database"],
    )

    # Render template
    context = {
        "leachate":           leachate["leachate"],
        "neutralization":     leachate["neutralization"],
        "equilibrium_phases": leachate["equilibrium_phases"],
        "knobs":              leachate["knobs"],
        "modifier":           modifier,
        "o2_reservoir_mol":   leachate.get("o2_reservoir_mol", 10.0),
    }
    rendered = render_template("templates", "neutralization.phr.j2", context)

    # Final input = database header + rendered body
    final_input = db_header + "\n\n" + rendered

    # Determine output path
    if args.output is None:
        dose_str = f"{modifier['dose_mol']:.2f}".replace(".", "p")
        naoh_str = f"{leachate['neutralization']['naoh_dose_mol']:.2f}".replace(".", "p")
        out_dir = Path(f"experiments/single/{modifier['name']}_dose{dose_str}_NaOH{naoh_str}")
    else:
        out_dir = Path(args.output).parent

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "input.phr"

    with open(out_path, "w") as f:
        f.write(final_input)

    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
