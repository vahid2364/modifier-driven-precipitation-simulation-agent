"""
Run a PHREEQC simulation from a generated input file.

Usage:
    python src/run_phreeqc.py \
        --input experiments/single/EDTA_dose0p05_NaOH0p50/input.phr

    # Or generate + run in one call:
    python src/run_phreeqc.py --generate \
        --leachate config/nmc532_baseline.yaml \
        --modifier config/modifiers/EDTA.yaml \
        --naoh-dose 0.5 --modifier-dose 0.05
"""

import argparse
import subprocess
import shutil
import sys
from pathlib import Path


PHREEQC_CANDIDATES = [
    "phreeqc",
    "/usr/local/bin/phreeqc",
    "/opt/homebrew/bin/phreeqc",
    "/usr/bin/phreeqc",
    "/Users/vahid/.local/bin/phreeqc",
]

DATABASE_CANDIDATES = [
    "/Users/vahid/.local/share/doc/phreeqc/database/sit.dat",
    "/usr/local/share/doc/phreeqc/database/sit.dat",
    "/opt/homebrew/share/doc/phreeqc/database/sit.dat",
    "/usr/share/phreeqc/database/sit.dat",
]


def find_phreeqc():
    """Return path to phreeqc executable or raise."""
    for candidate in PHREEQC_CANDIDATES:
        if shutil.which(candidate):
            return candidate
    raise FileNotFoundError(
        "phreeqc executable not found. Install PHREEQC and ensure it is on PATH."
    )


def find_database():
    """Return path to phreeqc.dat base database or raise."""
    for candidate in DATABASE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "phreeqc.dat not found. Pass --database /path/to/phreeqc.dat explicitly."
    )


def run(input_path: Path, phreeqc_exe: str = None, database: str = None) -> Path:
    """
    Run PHREEQC on input_path.
    Output files are written to the same directory as input_path.
    Returns path to selected_output.csv.
    """
    input_path = Path(input_path).resolve()
    out_dir    = input_path.parent
    log_path   = out_dir / "phreeqc.log"
    sel_out    = out_dir / "selected_output.csv"

    if phreeqc_exe is None:
        phreeqc_exe = find_phreeqc()
    if database is None:
        database = find_database()

    # PHREEQC CLI: phreeqc <input> <output.log> <database>
    cmd = [phreeqc_exe, str(input_path), str(log_path), database]

    print(f"Running : {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("--- PHREEQC stdout ---")
        print(result.stdout[-3000:])
        print("--- PHREEQC stderr ---")
        print(result.stderr[-3000:])
        raise RuntimeError(f"PHREEQC exited with code {result.returncode}")

    # PHREEQC writes output files to the working directory; move them to
    # the experiment folder.
    for fname in ["selected_output.csv", "leaching.csv", "neutralization.csv"]:
        src = Path(fname)
        if src.exists():
            dst = out_dir / fname
            src.rename(dst)

    print(f"Log     : {log_path}")
    print(f"Output  : {sel_out}")
    return sel_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          default=None,
                        help="Path to .phr input file")
    parser.add_argument("--generate",       action="store_true",
                        help="Generate input first, then run")
    parser.add_argument("--leachate",       default="config/nmc532_baseline.yaml")
    parser.add_argument("--modifier",       default="config/modifiers/EDTA.yaml")
    parser.add_argument("--naoh-dose",      type=float, default=0.5)
    parser.add_argument("--modifier-dose",  type=float, default=0.05)
    parser.add_argument("--phreeqc",        default=None,
                        help="Path to phreeqc executable (optional)")
    parser.add_argument("--database",       default=None,
                        help="Path to phreeqc.dat base database (optional)")
    args = parser.parse_args()

    if args.generate or args.input is None:
        # Import and call generate_input inline
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_input import main as gen_main
        import sys as _sys
        _sys.argv = [
            "generate_input.py",
            "--leachate",      args.leachate,
            "--modifier",      args.modifier,
            "--naoh-dose",     str(args.naoh_dose),
            "--modifier-dose", str(args.modifier_dose),
        ]
        gen_main()

        # Reconstruct the expected output path
        import yaml
        mod = yaml.safe_load(open(args.modifier))["modifier"]
        dose_str = f"{args.modifier_dose:.2f}".replace(".", "p")
        naoh_str = f"{args.naoh_dose:.2f}".replace(".", "p")
        input_path = Path(
            f"experiments/single/{mod['name']}_dose{dose_str}_NaOH{naoh_str}/input.phr"
        )
    else:
        input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    run(input_path, phreeqc_exe=args.phreeqc, database=args.database)


if __name__ == "__main__":
    main()
