import plotly.express as px
import streamlit as st

from app.utils import load_model_metrics, load_test_predictions


st.title("Steam User Reception Explorer")

st.markdown(
    """
    **Can Steam game metadata and post-release engagement signals
    be used to predict user reception?**
    """
)

st.info(
    """
    This is a post-release reception modeling project, not a pre-release
    success forecasting tool. Estimated ownership, playtime and Peak CCU
    are observed only after release.
    """
)

try:
    predictions = load_test_predictions()
    metrics = load_model_metrics()

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


# ---------------------------------------------------------
# Final model metrics
# ---------------------------------------------------------

final_model_rows = metrics[
    metrics["Model"]
    .astype(str)
    .str.contains("final", case=False, na=False)
]

if final_model_rows.empty:
    final_model = metrics.iloc[-1]
else:
    final_model = final_model_rows.iloc[-1]


st.header("Final model performance")

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Held-out games",
    f"{len(predictions):,}",
)

metric_columns[1].metric(
    "Test RMSE",
    f"{final_model['RMSE']:.3f}",
)

metric_columns[2].metric(
    "Test MAE",
    f"{final_model['MAE']:.3f}",
)

metric_columns[3].metric(
    "Test R²",
    f"{final_model['R2']:.3f}",
)


# ---------------------------------------------------------
# Research question and findings
# ---------------------------------------------------------

st.header("Research question")

st.markdown(
    """
    > Can Steam game metadata and post-release engagement signals
    > be used to predict user reception?
    """
)

st.markdown(
    """
    The results provide a partially positive answer. Structured Steam
    metadata contains meaningful predictive information, but a substantial
    share of reception remains unexplained.

    Key findings:

    - XGBoost outperforms the regularized linear baselines.
    - Genres and Categories improve performance over numerical features.
    - Steam Tags provide a further categorical improvement.
    - The final model tends to overestimate poorly received games and
      underestimate highly received games.
    """
)


# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------

left_column, right_column = st.columns(2)


with left_column:
    st.subheader("Held-out target distribution")

    target_figure = px.histogram(
        predictions,
        x="actual_score",
        nbins=50,
        labels={
            "actual_score": "Bayesian-smoothed review score",
        },
    )

    target_figure.update_layout(
        yaxis_title="Number of games",
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(
        target_figure,
        width="stretch",
        config={"scrollZoom": False},
    )


with right_column:
    st.subheader("Model comparison")

    display_metrics = metrics.copy()

    model_name_map = {
        "Dummy mean": "Dummy mean",
        "XGB numeric only": "XGBoost — numerical",
        "XGB numeric + Genres/Categories": "XGBoost + Genres/Categories",
        "XGB final tuned (numerical + Genres/Categories/Tags)": "Final XGBoost + Tags",
    }

    display_metrics["Display model"] = (
        display_metrics["Model"]
        .replace(model_name_map)
    )

    comparison_figure = px.bar(
        display_metrics,
        x="RMSE",
        y="Display model",
        orientation="h",
        text_auto=".3f",
    )

    comparison_figure.update_layout(
        xaxis_title="Held-out test RMSE",
        yaxis_title="",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    comparison_figure.update_yaxes(
        autorange="reversed"
    )

    st.plotly_chart(
        comparison_figure,
        width="stretch",
        config={"scrollZoom": False},
    )


# ---------------------------------------------------------
# Links
# ---------------------------------------------------------

st.header("Project resources")

st.markdown(
    """
    - [GitHub repository](https://github.com/StefanoTriscali/steam-user-reception-modeling)
    - [Technical report](https://github.com/StefanoTriscali/steam-user-reception-modeling/blob/main/reports/steam_user_reception_technical_report.pdf)
    - [Cleaned dataset on Kaggle](https://www.kaggle.com/datasets/stefanotriscali/steam-database-2026-fixed)
    """
)

st.caption(
    "Developed by Stefano Triscali."
)
