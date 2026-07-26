import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    load_error_by_quintile,
    load_model_metrics,
    load_permutation_importance,
    load_test_predictions,
)


st.title("Model Performance")

st.markdown(
    """
    Explore the held-out performance of the final XGBoost model,
    its most influential predictors and the distribution of prediction
    errors across different levels of user reception.
    """
)

try:
    predictions = load_test_predictions()
    metrics = load_model_metrics()
    importance = load_permutation_importance()
    quintile_errors = load_error_by_quintile()

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


# ---------------------------------------------------------
# Summary diagnostics
# ---------------------------------------------------------

mean_signed_error = predictions["error"].mean()
median_absolute_error = predictions["absolute_error"].median()
within_10 = (predictions["absolute_error"] <= 10).mean() * 100
within_20 = (predictions["absolute_error"] <= 20).mean() * 100

summary_columns = st.columns(4)

summary_columns[0].metric(
    "Mean signed error",
    f"{mean_signed_error:+.2f}",
    help="Observed score minus predicted score.",
)

summary_columns[1].metric(
    "Median absolute error",
    f"{median_absolute_error:.2f}",
)

summary_columns[2].metric(
    "Predictions within 10 points",
    f"{within_10:.1f}%",
)

summary_columns[3].metric(
    "Predictions within 20 points",
    f"{within_20:.1f}%",
)


# ---------------------------------------------------------
# Predicted versus actual
# ---------------------------------------------------------

st.header("Predicted versus actual reception")

scatter = go.Figure()

scatter.add_trace(
    go.Scattergl(
        x=predictions["actual_score"],
        y=predictions["predicted_score"],
        mode="markers",
        marker={
            "size": 5,
            "opacity": 0.25,
        },
        text=predictions["game_name"],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Observed: %{x:.2f}<br>"
            "Predicted: %{y:.2f}"
            "<extra></extra>"
        ),
        name="Games",
    )
)

minimum_score = min(
    predictions["actual_score"].min(),
    predictions["predicted_score"].min(),
)

maximum_score = max(
    predictions["actual_score"].max(),
    predictions["predicted_score"].max(),
)

scatter.add_trace(
    go.Scatter(
        x=[minimum_score, maximum_score],
        y=[minimum_score, maximum_score],
        mode="lines",
        line={
            "dash": "dash",
            "width": 2,
        },
        name="Perfect prediction",
        hoverinfo="skip",
    )
)

scatter.update_layout(
    xaxis_title="Observed Bayesian-smoothed score",
    yaxis_title="Predicted score",
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(
    scatter,
    width="stretch",
    config={"displayModeBar": False},
)

st.caption(
    """
    Predictions are concentrated more strongly around the centre of the
    score distribution than the observed outcomes, indicating regression
    toward the mean.
    """
)


# ---------------------------------------------------------
# Model comparison and permutation importance
# ---------------------------------------------------------

left_column, right_column = st.columns(2)


with left_column:
    st.subheader("Held-out model comparison")

    display_metrics = metrics.copy()

    model_name_map = {
        "Dummy mean": "Dummy mean",
        "XGB numeric only": "XGBoost — numerical",
        "XGB numeric + Genres/Categories":
            "XGBoost + Genres/Categories",
        "XGB final tuned (numerical + Genres/Categories/Tags)":
            "Final XGBoost + Tags",
    }

    display_metrics["Display model"] = (
        display_metrics["Model"]
        .replace(model_name_map)
    )

    model_figure = px.bar(
        display_metrics,
        x="RMSE",
        y="Display model",
        orientation="h",
        text_auto=".3f",
    )

    model_figure.update_yaxes(
        autorange="reversed",
        title="",
    )

    model_figure.update_layout(
        xaxis_title="Held-out test RMSE",
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        model_figure,
        width="stretch",
        config={"displayModeBar": False},
    )


with right_column:
    st.subheader("Permutation importance")

    importance_display = (
        importance
        .nlargest(15, "importance")
        .sort_values("importance", ascending=True)
        .copy()
    )

    importance_figure = px.bar(
        importance_display,
        x="importance",
        y="feature",
        orientation="h",
        error_x=(
            "importance_std"
            if "importance_std" in importance_display.columns
            else None
        ),
    )

    importance_figure.update_layout(
        xaxis_title="Increase in RMSE when permuted",
        yaxis_title="",
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        importance_figure,
        width="stretch",
        config={"displayModeBar": False},
    )


# ---------------------------------------------------------
# Error by reception quintile
# ---------------------------------------------------------

st.header("Error across reception quintiles")

signed_column, absolute_column = st.columns(2)


with signed_column:
    st.subheader("Mean signed error")

    signed_error_figure = px.bar(
        quintile_errors,
        x="reception_quintile",
        y="mean_signed_error",
        text_auto=".2f",
    )

    signed_error_figure.add_hline(
        y=0,
        line_dash="dash",
    )

    signed_error_figure.update_layout(
        xaxis_title="True reception quintile",
        yaxis_title="Observed minus predicted score",
        height=430,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        signed_error_figure,
        width="stretch",
        config={"displayModeBar": False},
    )


with absolute_column:
    st.subheader("Mean absolute error")

    absolute_error_figure = px.bar(
        quintile_errors,
        x="reception_quintile",
        y="mean_absolute_error",
        text_auto=".2f",
    )

    absolute_error_figure.update_layout(
        xaxis_title="True reception quintile",
        yaxis_title="Mean absolute error",
        height=430,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        absolute_error_figure,
        width="stretch",
        config={"displayModeBar": False},
    )


st.info(
    """
    Negative signed errors indicate that the model overestimated reception,
    while positive values indicate underestimation. The two extreme quintiles
    exhibit the largest systematic bias and the highest absolute errors.
    """
)

st.caption(
    """
    Permutation importance measures predictive usefulness, not causal impact.
    Correlated predictors may share information and therefore reduce each
    other's measured importance.
    """
)
