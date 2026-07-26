from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"


def _read_csv(filename: str) -> pd.DataFrame:
    """
    Read a CSV file from app/data and raise a clear error
    if the file cannot be found.
    """
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required application file not found: {path}"
        )

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    df = _read_csv("test_predictions.csv.gz")

    required_columns = {
        "game_name",
        "actual_score",
        "predicted_score",
        "error",
        "absolute_error",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "test_predictions.csv.gz is missing the following columns: "
            + ", ".join(sorted(missing_columns))
        )

    numeric_columns = [
        "actual_score",
        "predicted_score",
        "error",
        "absolute_error",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_model_metrics() -> pd.DataFrame:
    df = _read_csv("model_metrics.csv")

    required_columns = {"Model", "RMSE", "MAE", "R2"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "model_metrics.csv is missing the following columns: "
            + ", ".join(sorted(missing_columns))
        )

    for column in ["RMSE", "MAE", "R2"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_permutation_importance() -> pd.DataFrame:
    return _read_csv("permutation_importance.csv")


@st.cache_data(show_spinner=False)
def load_error_by_quintile() -> pd.DataFrame:
    return _read_csv("error_by_quintile.csv")
