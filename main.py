#!/usr/bin/env python3
"""
Industrial Fault Detection Using Image Processing
=================================================
Master Execution Pipeline (Phases 1, 2, 3)
-----------------------------------------
Executes:
  --phase 1 : Data setup, synthetic generation, preprocessing, stratified splitting
  --phase 2 : Handcrafted feature extraction, SVM baseline, Deep CNN training
  --phase 3 : Held-out evaluation, diagnostic curves, anomaly highlighting, batch inference
  --all     : Runs all three phases sequentially
"""

import sys
import argparse
import subprocess


def run_phase(phase_num: int):
    script_name = f"run_phase{phase_num}.py"
    print(f"\n{'='*80}")
    print(f"   STARTING PHASE {phase_num}: {script_name.upper()}")
    print(f"{'='*80}\n")
    ret = subprocess.call([sys.executable, script_name])
    if ret != 0:
        print(f"\n[ERROR] Phase {phase_num} failed with exit code {ret}", file=sys.stderr)
        sys.exit(ret)
    print(f"\n[COMPLETED] Phase {phase_num} finished successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Master execution script for Industrial Fault Detection project."
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Specify an individual phase to run (1, 2, or 3).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all phases (1, 2, and 3) end-to-end.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the complete automated pytest suite.",
    )

    args = parser.parse_args()

    if args.test:
        print("\n[INFO] Running test suite via pytest...")
        ret = subprocess.call([sys.executable, "-m", "pytest", "-v", "tests/"])
        sys.exit(ret)

    if args.all:
        for p in [1, 2, 3]:
            run_phase(p)
        print(f"\n{'='*80}")
        print("   ALL PHASES (1, 2, 3) COMPLETED SUCCESSFULLY END-TO-END!")
        print(f"{'='*80}\n")
    elif args.phase is not None:
        run_phase(args.phase)
    else:
        # Default behavior: run all
        print("[INFO] No phase specified. Running all phases end-to-end...")
        for p in [1, 2, 3]:
            run_phase(p)
        print(f"\n{'='*80}")
        print("   ALL PHASES (1, 2, 3) COMPLETED SUCCESSFULLY END-TO-END!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
