import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================
# STEP 46 - FINAL METHOD COMPARISON
# ============================================

print("========================================")
print("STEP 46 - FINAL METHOD COMPARISON")
print("========================================")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# FINAL EXPERIMENTAL VALUES
# ============================================

methods = [
    "Normal Template",
    "Multi-scale + Rotation"
]

matching_scores = [
    0.678829,
    0.971522
]

localization_errors = [
    637.377439,
    0.0
]

success = [
    False,
    True
]


# ============================================
# CREATE DATAFRAME
# ============================================

comparison_df = pd.DataFrame({
    "Method": methods,
    "Matching Score": matching_scores,
    "Localization Error (pixels)": localization_errors,
    "Success": success
})


# ============================================
# SAVE CSV
# ============================================

csv_path = (
    RESULTS_DIR /
    "method_comparison.csv"
)

comparison_df.to_csv(
    csv_path,
    index=False
)


# ============================================
# CALCULATE IMPROVEMENT
# ============================================

score_improvement = (
    matching_scores[1]
    - matching_scores[0]
)

error_reduction = (
    (
        localization_errors[0]
        - localization_errors[1]
    )
    / localization_errors[0]
) * 100


# ============================================
# PRINT RESULTS
# ============================================

print()
print("FINAL METHOD COMPARISON")
print("----------------------------------------")

print(comparison_df.to_string(index=False))

print()
print("PROPOSED METHOD")
print("----------------------------------------")

print(
    f"Matching score: "
    f"{matching_scores[1]:.6f}"
)

print(
    f"Localization error: "
    f"{localization_errors[1]:.3f} pixels"
)

print(
    f"Success: "
    f"{success[1]}"
)

print()
print("IMPROVEMENT OVER NORMAL TEMPLATE")
print("----------------------------------------")

print(
    f"Matching score improvement: "
    f"{score_improvement:.6f}"
)

print(
    f"Localization error reduction: "
    f"{error_reduction:.1f}%"
)


# ============================================
# GRAPH 1 - MATCHING SCORE
# ============================================

plt.figure(figsize=(9, 6))

plt.bar(
    methods,
    matching_scores
)

plt.ylabel(
    "Matching Score"
)

plt.xlabel(
    "Detection Method"
)

plt.title(
    "DRIFT-SENSE - Method Comparison"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

score_graph = (
    FIGURES_DIR /
    "final_method_matching_score.png"
)

plt.savefig(
    score_graph,
    dpi=300
)

plt.close()


# ============================================
# GRAPH 2 - LOCALIZATION ERROR
# ============================================

plt.figure(figsize=(9, 6))

plt.bar(
    methods,
    localization_errors
)

plt.ylabel(
    "Localization Error (pixels)"
)

plt.xlabel(
    "Detection Method"
)

plt.title(
    "DRIFT-SENSE - Localization Error Comparison"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

error_graph = (
    FIGURES_DIR /
    "final_method_localization_error.png"
)

plt.savefig(
    error_graph,
    dpi=300
)

plt.close()


# ============================================
# FINAL SUMMARY
# ============================================

print()
print("Saved files:")
print("----------------------------------------")
print(csv_path)
print(score_graph)
print(error_graph)

print()
print("========================================")
print("STEP 46 COMPLETE")
print("========================================")

print(
    f"Baseline score: "
    f"{matching_scores[0]:.6f}"
)

print(
    f"Proposed score: "
    f"{matching_scores[1]:.6f}"
)

print(
    f"Score improvement: "
    f"{score_improvement:.6f}"
)

print(
    f"Baseline localization error: "
    f"{localization_errors[0]:.3f} pixels"
)

print(
    f"Proposed localization error: "
    f"{localization_errors[1]:.3f} pixels"
)

print(
    f"Localization error reduction: "
    f"{error_reduction:.1f}%"
)

print(
    f"Final proposed-method success: "
    f"{success[1]}"
)

print("========================================")