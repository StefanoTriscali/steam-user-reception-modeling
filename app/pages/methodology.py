import streamlit as st


st.title("Methodology & Limitations")

st.markdown(
    """
    This page explains how the reception target was constructed, how the
    machine-learning pipeline was evaluated and how the results should be
    interpreted.
    """
)

target_tab, workflow_tab, limitations_tab, resources_tab = st.tabs(
    [
        "Target construction",
        "Modeling workflow",
        "Limitations",
        "Resources",
    ]
)


# =========================================================
# Target construction
# =========================================================

with target_tab:
    st.header("Measuring Steam user reception")

    st.markdown(
        """
        Steam does not provide a direct Net Promoter Score. The project
        therefore constructs an NPS-inspired measure from the number of
        positive and negative reviews received by each game.
        """
    )

    st.subheader("1. Total number of reviews")

    st.latex(
        r"""
        n_i = Positive_i + Negative_i
        """
    )

    st.markdown(
        """
        Games without reviews were excluded because no observable reception
        signal could be constructed for them.
        """
    )

    st.subheader("2. Raw review reception score")

    st.latex(
        r"""
        RawScore_i =
        \frac{Positive_i - Negative_i}
        {Positive_i + Negative_i}
        \times 100
        """
    )

    st.markdown(
        """
        The raw score ranges from −100 to 100:

        - values close to **100** indicate predominantly positive reviews;
        - values close to **−100** indicate predominantly negative reviews;
        - values close to **0** indicate a balance between positive and
          negative reviews.
        """
    )

    st.warning(
        """
        Raw review scores can be highly unstable for games with very few
        reviews. A game with one positive review and no negative reviews
        would receive a raw score of 100 despite the limited evidence.
        """
    )

    st.subheader("3. Bayesian smoothing")

    st.markdown(
        """
        To reduce this instability, the observed review balance is shrunk
        toward the global Steam review balance.
        """
    )

    st.latex(
        r"""
        R_i =
        \frac{Positive_i - Negative_i}
        {Positive_i + Negative_i}
        """
    )

    st.latex(
        r"""
        C =
        \frac{
        \sum_i Positive_i - \sum_i Negative_i
        }{
        \sum_i \left(Positive_i + Negative_i\right)
        }
        """
    )

    st.latex(
        r"""
        BayesianScore_i =
        \left(
        \frac{n_i}{n_i + m} R_i
        +
        \frac{m}{n_i + m} C
        \right)
        \times 100
        """
    )

    st.markdown(
        """
        where:

        - $n_i$ is the number of reviews received by game $i$;
        - $R_i$ is the observed positive–negative review balance;
        - $C$ is the global review balance;
        - $m$ controls the strength of the smoothing.

        In this project, $m = 23$, corresponding to the median number of
        reviews in the modeling dataset.
        """
    )

    st.info(
        """
        Games with few reviews are pulled more strongly toward the global
        balance. Games with many reviews remain much closer to their observed
        raw score.
        """
    )

    st.subheader("Interpretation of the target")

    st.markdown(
        """
        The resulting variable, `review_score_bayes`, is used as the
        continuous target in all regression models.

        It should be interpreted as a Bayesian-smoothed measure of Steam user
        reception. It is inspired by the logic of Net Promoter Score, but it
        is not a true survey-based NPS because Steam reviews are binary and do
        not distinguish promoters, passives and detractors.
        """
    )


# =========================================================
# Modeling workflow
# =========================================================

with workflow_tab:
    st.header("End-to-end modeling workflow")

    st.markdown(
        """
        The analysis follows a sequential pipeline designed to separate model
        development from final evaluation.
        """
    )

    st.subheader("1. Data preparation")

    st.markdown(
        """
        - Load the cleaned Steam dataset.
        - Remove games without an observable review signal.
        - Construct the raw and Bayesian-smoothed reception scores.
        - Exclude variables directly involved in target construction.
        - Engineer numerical and multilabel predictors.
        """
    )

    st.subheader("2. Numerical feature engineering")

    numerical_left, numerical_right = st.columns(2)

    with numerical_left:
        st.markdown(
            """
            **Commercial and temporal features**

            - Estimated original price
            - Estimated owners midpoint
            - Release year
            - Release month
            """
        )

    with numerical_right:
        st.markdown(
            """
            **Post-release engagement features**

            - Average playtime
            - Median playtime
            - Peak concurrent users
            """
        )

    st.subheader("3. Multilabel feature engineering")

    st.markdown(
        """
        Genres, Categories and Steam Tags can contain multiple labels for the
        same game. Each retained label is therefore represented as a separate
        binary feature.

        Label vocabularies were learned from the training set and then applied
        unchanged to the test set.
        """
    )

    categorical_left, categorical_right = st.columns(2)

    with categorical_left:
        st.markdown(
            """
            **Genres and Categories**

            Labels were retained when they appeared at least 100 times in the
            training set.
            """
        )

    with categorical_right:
        st.markdown(
            """
            **Steam Tags**

            Tags were retained when they appeared at least 200 times in the
            training set.
            """
        )

    st.subheader("4. Train-test separation")

    st.markdown(
        """
        The dataset was divided into:

        - **80% training data**
        - **20% held-out test data**

        A fixed random seed of `42` was used for reproducibility.
        """
    )

    st.info(
        """
        Model selection, feature-block selection and hyperparameter tuning
        were performed using the training set. The held-out test set was
        reserved for final evaluation.
        """
    )

    st.subheader("5. Model comparison")

    st.markdown(
        """
        The numerical baseline stage compared:

        - Dummy Regressor;
        - Ridge Regression;
        - Lasso Regression;
        - numerical-only XGBoost.

        XGBoost obtained the strongest cross-validated numerical performance
        and was selected for the subsequent feature-extension stages.
        """
    )

    st.subheader("6. Progressive feature extension")

    st.markdown(
        """
        Three increasingly rich feature sets were evaluated:

        1. numerical predictors only;
        2. numerical predictors plus Genres and Categories;
        3. numerical predictors plus Genres, Categories and Tags.
        """
    )

    st.subheader("7. Final evaluation")

    st.markdown(
        """
        The selected XGBoost configuration was retrained on the full training
        set and evaluated once on the held-out test set using:

        - **RMSE**, which penalizes larger errors more heavily;
        - **MAE**, which measures the average absolute prediction error;
        - **R²**, which measures the share of target variation explained.
        """
    )

    st.warning(
        """
        Positive reviews, negative reviews and total review counts are used to
        construct the target and are not included among the model predictors.
        Including them would create direct target leakage.
        """
    )

    st.subheader("8. Interpretation and error analysis")

    st.markdown(
        """
        The final analysis includes:

        - permutation importance;
        - predicted-versus-observed reception;
        - signed prediction errors;
        - absolute errors across reception quintiles;
        - inspection of regression toward the mean.
        """
    )


# =========================================================
# Limitations
# =========================================================

with limitations_tab:
    st.header("Important limitations")

    st.markdown(
        """
        The application is intended to explore predictive patterns in Steam
        user reception. Its results should be interpreted within the following
        limitations.
        """
    )

    with st.expander(
        "Predictive rather than causal",
        expanded=True,
    ):
        st.markdown(
            """
            Feature importance indicates whether a variable helps the fitted
            model make predictions. It does not show that the variable causes
            higher or lower user reception.

            For example, a Steam Tag may identify games that tend to receive
            similar evaluations without the tag itself being responsible for
            those evaluations.
            """
        )

    with st.expander("Post-release rather than pre-release"):
        st.markdown(
            """
            Estimated ownership, average playtime, median playtime and Peak CCU
            are observed only after a game has been released.

            The model is therefore a post-release reception model and should
            not be presented as a tool for forecasting commercial success
            before launch.
            """
        )

    with st.expander("NPS-inspired target"):
        st.markdown(
            """
            The target is derived from binary Steam reviews. Traditional Net
            Promoter Score instead comes from survey responses and separates
            promoters, passives and detractors.

            The Bayesian-smoothed score is therefore an NPS-inspired reception
            proxy, not a direct Net Promoter Score.
            """
        )

    with st.expander("Self-selected user reviews"):
        st.markdown(
            """
            Users who choose to leave reviews may not represent the complete
            player population.

            Reviews can also be influenced by unusually strong experiences,
            community campaigns, controversies, review bombing and platform
            dynamics.
            """
        )

    with st.expander("Unobserved qualitative information"):
        st.markdown(
            """
            The structured dataset does not directly measure several factors
            that may strongly affect reception:

            - bugs and technical problems;
            - update history;
            - writing and gameplay quality;
            - monetization decisions;
            - developer and publisher reputation;
            - marketing exposure;
            - review-text sentiment;
            - community controversies.
            """
        )

    with st.expander("Regression toward the mean"):
        st.markdown(
            """
            The final model tends to overestimate poorly received games and
            underestimate highly received games.

            Predictions should therefore be interpreted as broad estimates of
            reception rather than exact evaluations of individual games.
            """
        )

    with st.expander("Temporal generalization"):
        st.markdown(
            """
            The project uses a random train-test split. It therefore evaluates
            generalization across games contained in the same historical
            dataset rather than performance on future Steam releases.

            A chronological hold-out set would be needed to evaluate how well
            the model generalizes to later market periods.
            """
        )

    with st.expander("Games without reviews"):
        st.markdown(
            """
            Games without positive or negative reviews were excluded because
            their target could not be constructed.

            The results therefore apply to games with an observable Steam
            review signal and may not generalize to the least visible titles.
            """
        )

    with st.expander("Steam Tags and correlated predictors"):
        st.markdown(
            """
            Steam Tags may reflect community perception and platform
            conventions rather than stable, objective characteristics.

            Moreover, correlated predictors may share similar information.
            This can reduce the measured permutation importance of individual
            features.
            """
        )


# =========================================================
# Resources
# =========================================================

with resources_tab:
    st.header("Project documentation")

    st.markdown(
        """
        The application provides an interactive summary of the project.
        Complete formulas, model-selection details, results and references
        are available in the technical report.
        """
    )

    st.link_button(
        "Read the technical report",
        (
            "https://github.com/StefanoTriscali/"
            "steam-user-reception-modeling/blob/main/"
            "reports/steam_user_reception_technical_report.pdf"
        ),
        icon="📄",
    )

    st.link_button(
        "Read the industry report",
        (
            "https://github.com/StefanoTriscali/"
            "steam-user-reception-modeling/blob/main/"
            "reports/steam_user_reception_industry_report.pdf"
        ),
        icon="👾",
    )

    st.link_button(
        "Open the GitHub repository",
        (
            "https://github.com/StefanoTriscali/"
            "steam-user-reception-modeling"
        ),
        icon="💻",
    )

    st.link_button(
        "View the cleaned dataset on Kaggle",
        (
            "https://www.kaggle.com/datasets/"
            "stefanotriscali/steam-database-2026-fixed"
        ),
        icon="🗃️",
    )

    st.divider()

    st.subheader("Suggested reading order")

    st.markdown(
        """
        1. Start from **Overview** for the research question and main results.
        2. Use **Game Explorer** to inspect individual held-out predictions.
        3. Open **Model Performance** for aggregate diagnostics.
        4. Return to this page for methodological details and limitations.
        """
    )

    st.caption(
        "Steam User Reception Modeling — Stefano Triscali, 2026."
    )
