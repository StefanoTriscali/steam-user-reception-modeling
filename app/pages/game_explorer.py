import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils import load_test_predictions


st.title("Game Explorer")

st.markdown(
    """
    Select a game from the held-out test set to compare its observed
    Bayesian-smoothed reception score with the prediction produced by
    the final XGBoost model.
    """
)

st.info(
    """
    The prediction shown here is a held-out test prediction. The game was
    not used to train the final model.
    """
)

try:
    predictions = load_test_predictions()

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

# ---------------------------------------------------------
# Prepare game labels
# ---------------------------------------------------------

predictions = predictions.copy()

predictions["game_name"] = (
    predictions["game_name"]
    .fillna("Unknown game")
    .astype(str)
)

if "developers" in predictions.columns:
    predictions["developers"] = (
        predictions["developers"]
        .fillna("Unknown developer")
        .astype(str)
    )

    predictions["display_name"] = (
        predictions["game_name"]
        + " — "
        + predictions["developers"]
    )
else:
    predictions["display_name"] = predictions["game_name"]


if "app_id" in predictions.columns:
    predictions["display_name"] = (
        predictions["display_name"]
        + " ["
        + predictions["app_id"].astype(str)
        + "]"
    )


predictions = predictions.sort_values(
    "display_name",
    key=lambda series: series.str.lower(),
)

# ---------------------------------------------------------
# Prepare case studies and selection
# ---------------------------------------------------------

display_options = predictions["display_name"].tolist()


def find_display_name(game_name: str) -> str | None:
    matches = predictions.loc[
        predictions["game_name"].eq(game_name),
        "display_name",
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


case_studies = {
    "Battlefront": find_display_name(
        "STAR WARS™: Battlefront Classic Collection"
    ),
    "Command & Conquer 4": find_display_name(
        "Command & Conquer™ 4 Tiberian Twilight"
    ),
    "The Walking Dead": find_display_name(
        "The Walking Dead: The Final Season"
    ),
    "WEBFISHING": find_display_name(
        "WEBFISHING"
    ),
}


def select_case(display_name: str) -> None:
    st.session_state["game_selector"] = display_name


# Remove an invalid selection if the underlying data changes
if (
    "game_selector" in st.session_state
    and st.session_state["game_selector"] not in display_options
):
    del st.session_state["game_selector"]


with st.expander("Try an exceptional prediction case"):
    st.caption(
        "These examples show games whose reception was substantially "
        "overestimated or underestimated by the model."
    )

    col1, col2 = st.columns(2)

    col1.button(
        "Overestimated · Battlefront",
        use_container_width=True,
        disabled=case_studies["Battlefront"] is None,
        on_click=select_case,
        args=(case_studies["Battlefront"],),
    )

    col2.button(
        "Overestimated · Command & Conquer 4",
        use_container_width=True,
        disabled=case_studies["Command & Conquer 4"] is None,
        on_click=select_case,
        args=(case_studies["Command & Conquer 4"],),
    )

    col3, col4 = st.columns(2)

    col3.button(
        "Underestimated · The Walking Dead",
        use_container_width=True,
        disabled=case_studies["The Walking Dead"] is None,
        on_click=select_case,
        args=(case_studies["The Walking Dead"],),
    )

    col4.button(
        "Underestimated · WEBFISHING",
        use_container_width=True,
        disabled=case_studies["WEBFISHING"] is None,
        on_click=select_case,
        args=(case_studies["WEBFISHING"],),
    )


selected_label = st.selectbox(
    "Search for a game",
    options=display_options,
    index=None,
    placeholder="Type a game title...",
    key="game_selector",
)


if selected_label is None:
    st.markdown(
        """
        Choose a game above to inspect its prediction, reception score,
        engagement indicators and Steam metadata.
        """
    )
    st.stop()


selected_game = predictions.loc[
    predictions["display_name"] == selected_label
].iloc[0]


st.divider()

st.header(selected_game["game_name"])

if "developers" in selected_game.index:
    st.caption(f"Developer: {selected_game['developers']}")

if "app_id" in selected_game.index and pd.notna(selected_game["app_id"]):
    app_id = int(float(selected_game["app_id"]))

    st.link_button(
        "Open on Steam",
        f"https://store.steampowered.com/app/{app_id}",
        icon="↗",
    )

# ---------------------------------------------------------
# Main metrics
# ---------------------------------------------------------

actual_score = float(selected_game["actual_score"])
predicted_score = float(selected_game["predicted_score"])
signed_error = float(selected_game["error"])
absolute_error = float(selected_game["absolute_error"])


metric_columns = st.columns(4)

metric_columns[0].metric(
    "Observed reception",
    f"{actual_score:.2f}",
)

metric_columns[1].metric(
    "Predicted reception",
    f"{predicted_score:.2f}",
)

metric_columns[2].metric(
    "Signed error",
    f"{signed_error:+.2f}",
    help="Observed score minus predicted score.",
)

metric_columns[3].metric(
    "Absolute error",
    f"{absolute_error:.2f}",
)


if signed_error < 0:
    error_interpretation = (
        "The model overestimated this game's reception."
    )
elif signed_error > 0:
    error_interpretation = (
        "The model underestimated this game's reception."
    )
else:
    error_interpretation = (
        "The prediction matches the observed score."
    )

st.caption(error_interpretation)


# ---------------------------------------------------------
# Actual versus predicted chart
# ---------------------------------------------------------

chart_column, interpretation_column = st.columns([1.4, 1])

with chart_column:
    comparison_figure = go.Figure()

    comparison_figure.add_bar(
        x=["Observed", "Predicted"],
        y=[actual_score, predicted_score],
        text=[f"{actual_score:.2f}", f"{predicted_score:.2f}"],
        textposition="outside",
        marker_color=["#1f77b4", "#ff7f0e"],
    )

    comparison_figure.update_layout(
        title="Observed versus predicted reception",
        yaxis_title="Reception score",
        yaxis_range=[
            min(-10, actual_score - 15, predicted_score - 15),
            max(10, actual_score + 15, predicted_score + 15),
        ],
        showlegend=False,
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        comparison_figure,
        width="stretch",
        config={"displayModeBar": False},
    )

with interpretation_column:
    st.subheader("Prediction interpretation")

    if signed_error < 0:
        st.warning(
            f"The model overestimated reception by "
            f"{absolute_error:.2f} points."
        )
    elif signed_error > 0:
        st.info(
            f"The model underestimated reception by "
            f"{absolute_error:.2f} points."
        )
    else:
        st.success("The predicted and observed scores are identical.")

    percentile = (
        predictions["actual_score"]
        .rank(pct=True)
        .loc[selected_game.name]
        * 100
    )

    st.metric(
        "Reception percentile",
        f"{percentile:.0f}th",
        help=(
            f"This game's observed reception is higher than "
            f"approximately {percentile:.1f}% of held-out games."
        ),
    )

    if "reception_quintile" in selected_game.index:
        st.markdown(
            f"**Reception group:** "
            f"{selected_game['reception_quintile']}"
        )

# ---------------------------------------------------------
# Game information
# ---------------------------------------------------------

st.subheader("Game information")

information_columns = st.columns(3)


def format_number(value, decimals=0):
    if pd.isna(value):
        return "Not available"

    return f"{value:,.{decimals}f}"


with information_columns[0]:
    if "release_date" in selected_game.index:
        st.markdown(
            f"**Release date:** {selected_game['release_date']}"
        )

    if "release_year" in selected_game.index:
        st.markdown(
            f"**Release year:** "
            f"{int(selected_game['release_year'])}"
        )

    if "price_original" in selected_game.index:
        st.markdown(
            f"**Estimated original price:** "
            f"{format_number(selected_game['price_original'], 2)}"
        )


with information_columns[1]:
    if "owners_mid" in selected_game.index:
        st.markdown(
            f"**Estimated owners midpoint:** "
            f"{format_number(selected_game['owners_mid'])}"
        )

    if "peak_ccu" in selected_game.index:
        st.markdown(
            f"**Peak CCU:** "
            f"{format_number(selected_game['peak_ccu'])}"
        )

    average_playtime = selected_game["average_playtime"]

    if pd.notna(average_playtime):
        st.markdown(
            f"**Average playtime:** "
            f"{average_playtime:,.0f} minutes "
            f"({average_playtime / 60:.1f} hours)"
        )


with information_columns[2]:
    if "positive_reviews" in selected_game.index:
        st.markdown(
            f"**Positive reviews:** "
            f"{format_number(selected_game['positive_reviews'])}"
        )

    if "negative_reviews" in selected_game.index:
        st.markdown(
            f"**Negative reviews:** "
            f"{format_number(selected_game['negative_reviews'])}"
        )

    if "total_reviews" in selected_game.index:
        st.markdown(
            f"**Total reviews:** "
            f"{format_number(selected_game['total_reviews'])}"
        )


# ---------------------------------------------------------
# Multilabel metadata
# ---------------------------------------------------------

st.subheader("Steam metadata")


def show_multilabel_field(
    title: str,
    column: str,
    max_visible: int | None = None,
) -> None:
    if column not in selected_game.index:
        return

    value = selected_game[column]

    if pd.isna(value) or not str(value).strip():
        st.markdown(f"**{title}:** Not available")
        return

    labels = [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]

    st.markdown(f"**{title}**")

    if max_visible is not None and len(labels) > max_visible:
        st.write(" · ".join(labels[:max_visible]))

        with st.expander(
            f"Show all {len(labels)} {title.lower()}"
        ):
            st.write(" · ".join(labels))
    else:
        st.write(" · ".join(labels))


show_multilabel_field("Genres", "genres")
show_multilabel_field("Categories", "categories")
show_multilabel_field("Tags", "tags", max_visible=10)


# ---------------------------------------------------------
# Interpretation
# ---------------------------------------------------------

st.subheader("How to interpret this result")

st.markdown(
    """
    The observed score is constructed from the game's positive and negative
    reviews using Bayesian smoothing. The predicted score is generated using
    metadata, engagement indicators, Genres, Categories and Tags.

    A prediction error does not necessarily indicate that the model ignored
    an obvious characteristic. User reception may also depend on factors that
    are absent from the dataset, such as bugs, updates, writing quality,
    monetization decisions, developer reputation and community controversies.
    """
)
