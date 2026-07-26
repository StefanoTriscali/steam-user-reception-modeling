# ============================================================
# Export compact datasets for the Streamlit application
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

export_dir = Path(__file__).resolve().parent / "data"
export_dir.mkdir(parents=True, exist_ok=True)

# Columns that are useful for the Game Explorer.
# Only columns that actually exist in df_test are retained.
candidate_columns = [
    "AppID",
    "Name",
    "Developers",
    "Publishers",
    "Release date",
    "release_year",
    "release_month",
    "price_original",
    "owners_mid",
    "Average playtime forever",
    "Median playtime forever",
    "Peak CCU",
    "Genres",
    "Categories",
    "Tags",
    "Positive",
    "Negative",
    "total_reviews",
    "review_score_raw",
]

available_columns = [
    column
    for column in candidate_columns
    if column in df_test.columns
]

app_results = df_test[available_columns].copy()

# Add held-out outcomes and predictions.
app_results["actual_score"] = y_test.reindex(app_results.index).to_numpy()
app_results["predicted_score"] = np.asarray(y_pred_final)

# The error definition is consistent with the notebook:
# actual score minus predicted score.
app_results["error"] = (
    app_results["actual_score"]
    - app_results["predicted_score"]
)

app_results["absolute_error"] = app_results["error"].abs()

# Create readable reception quintiles.
quintile_labels = [
    "Q1 - Lowest reception",
    "Q2",
    "Q3",
    "Q4",
    "Q5 - Highest reception",
]

# Ranking first guarantees five similarly sized groups,
# even in the presence of repeated target values.
app_results["reception_quintile"] = pd.qcut(
    app_results["actual_score"].rank(method="first"),
    q=5,
    labels=quintile_labels,
)

# Rename columns to simpler application-friendly names.
app_results = app_results.rename(
    columns={
        "AppID": "app_id",
        "Name": "game_name",
        "Developers": "developers",
        "Publishers": "publishers",
        "Release date": "release_date",
        "Average playtime forever": "average_playtime",
        "Median playtime forever": "median_playtime",
        "Peak CCU": "peak_ccu",
        "Genres": "genres",
        "Categories": "categories",
        "Tags": "tags",
        "Positive": "positive_reviews",
        "Negative": "negative_reviews",
    }
)

# Sort alphabetically for easier searching in the app.
if "game_name" in app_results.columns:
    app_results = app_results.sort_values(
        "game_name",
        key=lambda series: series.astype(str).str.lower(),
    )

# ------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------

model_metrics_export = final_comparison.copy()

model_metrics_export = model_metrics_export.rename(
    columns={
        "model": "Model",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R2",
    }
)

# ------------------------------------------------------------
# Permutation importance
# ------------------------------------------------------------

permutation_export = (
    perm_final_df[
        ["feature", "importance", "importance_std"]
    ]
    .head(20)
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# Error analysis by quintile
# ------------------------------------------------------------

error_by_quintile_export = (
    app_results
    .groupby("reception_quintile", observed=True)
    .agg(
        game_count=("game_name", "size"),
        mean_actual_score=("actual_score", "mean"),
        mean_predicted_score=("predicted_score", "mean"),
        mean_signed_error=("error", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
    )
    .reset_index()
)

error_by_quintile_export["reception_quintile"] = (
    error_by_quintile_export["reception_quintile"].astype(str)
)

# ------------------------------------------------------------
# Save files
# ------------------------------------------------------------

app_results_path = export_dir / "test_predictions.csv.gz"
metrics_path = export_dir / "model_metrics.csv"
importance_path = export_dir / "permutation_importance.csv"
quintile_path = export_dir / "error_by_quintile.csv"

app_results.to_csv(
    app_results_path,
    index=False,
    compression="gzip",
)

model_metrics_export.to_csv(
    metrics_path,
    index=False,
)

permutation_export.to_csv(
    importance_path,
    index=False,
)

error_by_quintile_export.to_csv(
    quintile_path,
    index=False,
)

print("Export completed.")
print(f"Test observations: {len(app_results):,}")
print(f"Columns: {len(app_results.columns)}")

for path in [
    app_results_path,
    metrics_path,
    importance_path,
    quintile_path,
]:
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"{path.name}: {size_mb:.2f} MB")
