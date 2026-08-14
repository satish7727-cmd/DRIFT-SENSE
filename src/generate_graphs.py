import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================
# STEP 45 - ROBUSTNESS GRAPHS
# ============================================

print("========================================")
print("STEP 45 - ROBUSTNESS GRAPHS")
print("========================================")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------
# Load results
# --------------------------------------------

csv_path = (
    RESULTS_DIR
    / "robustness_results_corrected.csv"
)

df = pd.read_csv(csv_path)


# ============================================
# GRAPH 1 - MATCHING SCORE
# ============================================

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    df["Condition"],
    df["Score"]
)

plt.xlabel(
    "Test Condition"
)

plt.ylabel(
    "Matching Score"
)

plt.title(
    "DRIFT-SENSE - Robustness Matching Score"
)

plt.ylim(
    0,
    1.05
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

score_path = (
    FIGURES_DIR
    / "robustness_matching_score.png"
)

plt.savefig(
    score_path,
    dpi=300
)

plt.close()


# ============================================
# GRAPH 2 - LOCALIZATION ERROR
# ============================================

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    df["Condition"],
    df["Localization Error"]
)

plt.xlabel(
    "Test Condition"
)

plt.ylabel(
    "Localization Error (pixels)"
)

plt.title(
    "DRIFT-SENSE - Localization Error"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

error_path = (
    FIGURES_DIR
    / "robustness_localization_error.png"
)

plt.savefig(
    error_path,
    dpi=300
)

plt.close()


# ============================================
# GRAPH 3 - DETECTION TIME
# ============================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    range(
        1,
        len(df) + 1
    ),
    df["Detection Time (ms)"],
    marker="o"
)

plt.xlabel(
    "Test Case"
)

plt.ylabel(
    "Detection Time (ms)"
)

plt.title(
    "DRIFT-SENSE - Detection Time"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

time_path = (
    FIGURES_DIR
    / "robustness_detection_time.png"
)

plt.savefig(
    time_path,
    dpi=300
)

plt.close()


# ============================================
# GRAPH 4 - SCALE ERROR
# ============================================

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    df["Condition"],
    df["Scale Error"]
)

plt.xlabel(
    "Test Condition"
)

plt.ylabel(
    "Scale Error"
)

plt.title(
    "DRIFT-SENSE - Scale Estimation Error"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

scale_path = (
    FIGURES_DIR
    / "scale_estimation_error.png"
)

plt.savefig(
    scale_path,
    dpi=300
)

plt.close()


# ============================================
# GRAPH 5 - ROTATION ERROR
# ============================================

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    df["Condition"],
    df["Rotation Error"]
)

plt.xlabel(
    "Test Condition"
)

plt.ylabel(
    "Rotation Error (degrees)"
)

plt.title(
    "DRIFT-SENSE - Rotation Estimation Error"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

rotation_path = (
    FIGURES_DIR
    / "rotation_estimation_error.png"
)

plt.savefig(
    rotation_path,
    dpi=300
)

plt.close()


# ============================================
# PRINT RESULTS
# ============================================

print()
print("Saved figures:")
print("----------------------------------------")
print(score_path)
print(error_path)
print(time_path)
print(scale_path)
print(rotation_path)

print()
print("========================================")
print("STEP 45 COMPLETE")
print("========================================")