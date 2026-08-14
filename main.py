from pathlib import Path
import subprocess
import sys


# ============================================
# DRIFT-SENSE - MAIN PIPELINE
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"


def run_step(script_name):
    script_path = SRC / script_name

    print()
    print("=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print()
        print(f"ERROR: {script_name} failed.")
        sys.exit(result.returncode)


def main():

    print("=" * 60)
    print("DRIFT-SENSE")
    print("Robust Multi-scale and Rotation-aware Detection")
    print("=" * 60)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    # ----------------------------------------
    # Step 1: Generate dataset
    # ----------------------------------------

    run_step("generate_dataset.py")

    # ----------------------------------------
    # Step 2: Run detection
    # ----------------------------------------

    run_step("run_detection.py")

    # ----------------------------------------
    # Step 3: Save detection result
    # ----------------------------------------

    run_step("save_detection.py")

    # ----------------------------------------
    # Step 4: Robustness evaluation
    # ----------------------------------------

    run_step("robustness_test.py")

    # ----------------------------------------
    # Step 5: Generate graphs
    # ----------------------------------------

    run_step("generate_graphs.py")

    # ----------------------------------------
    # Step 6: Method comparison
    # ----------------------------------------

    run_step("method_comparison.py")

    # ----------------------------------------
    # Step 7: Final summary
    # ----------------------------------------

    run_step("final_summary.py")

    print()
    print("=" * 60)
    print("DRIFT-SENSE PIPELINE COMPLETE")
    print("=" * 60)

    print()
    print("Results:")
    print(PROJECT_ROOT / "results")

    print()
    print("Figures:")
    print(PROJECT_ROOT / "figures")


if __name__ == "__main__":
    main()