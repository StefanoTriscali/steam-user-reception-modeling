# Steam User Reception Modeling

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://steam-user-reception-modeling.streamlit.app/)

## Overview

This project investigates whether Steam game metadata, post-release engagement signals and multilabel categorical metadata can be used to predict user reception.

The analysis is framed as a supervised regression problem. The target is a Bayesian-smoothed review reception score constructed from Steam positive and negative review counts.

> **Important:** this is a post-release reception modeling project, not a pre-release success forecasting model. Several predictors, including estimated ownership, playtime and Peak CCU, are observed only after release.

## Interactive application

A lightweight interactive application is available on Streamlit Community Cloud:

**[Launch the Steam User Reception Modeling app](https://steam-user-reception-modeling.streamlit.app/)**

The application presents the project as a post-release analytical product rather than a pre-release forecasting tool. It includes four sections:

- **Overview** — research question, project framing and headline findings;
- **Game Explorer** — observed and predicted reception for individual games;
- **Model Performance** — benchmark comparison, permutation importance and error analysis;
- **Methodology & Limitations** — target construction, modeling workflow and interpretation boundaries.

## Research question

> Can Steam game metadata and post-release engagement signals be used to predict user reception?

## Main results

The final model is an XGBoost regressor trained using numerical metadata, Genres, Categories and Steam Tags.

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Dummy mean | 21.035 | 15.585 | 0.000 |
| XGBoost - numerical only | 19.001 | 13.879 | 0.184 |
| XGBoost + Genres/Categories | 17.956 | 13.105 | 0.271 |
| Final XGBoost + Genres/Categories/Tags | **17.383** | **12.618** | **0.317** |

Within the numerical-only comparison, XGBoost also achieved a lower cross-validated RMSE than Ridge and Lasso, supporting the use of a non-linear model for the subsequent feature-extension stages.

## Key findings

- Numerical post-release indicators contain meaningful predictive signal.
- XGBoost outperforms the regularized linear baselines.
- Genres and Categories improve performance over numerical features alone.
- Steam Tags provide the strongest additional categorical improvement.
- `release_year`, Peak CCU, average playtime, reconstructed original price and estimated ownership are among the most influential predictors according to permutation importance.
- The final model tends to regress toward the mean, overestimating poorly received games and underestimating highly received games.

The results suggest that structured Steam metadata can capture broad patterns in user reception, but cannot fully explain extreme positive or negative outcomes.

## Selected figures

### Target construction

The Bayesian smoothing procedure reduces the concentration of extreme scores
for games with very few reviews.

<p align="center">
  <img src="reports/figures/raw_vs_bayesian_review_score.png"
       alt="Raw and Bayesian-smoothed review score distributions"
       width="750">
</p>

### Model interpretation

Permutation importance shows that the final model relies on both numerical
post-release signals and multilabel metadata.

<p align="center">
  <img src="reports/figures/permutation_importance.png"
       alt="Permutation importance"
       width="750">
</p>

### Error analysis

The model tends to overestimate poorly received games and underestimate highly
received games.

<p align="center">
  <img src="reports/figures/error_by_reception_quintile.png"
       alt="Mean signed error by reception quintile"
       width="750">
</p>

## Dataset

The project is based on the [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset).

A cleaned and reproducible version of the dataset is available here:

[Steam Dataset 2026 Cleaned](https://www.kaggle.com/datasets/stefanotriscali/steam-database-2026-fixed)

After excluding games without reviews, the modeling dataset contains 82,766 observations.

The dataset includes:

- pricing and release information;
- estimated ownership;
- average and median playtime;
- Peak CCU;
- Genres;
- Categories;
- Steam Tags;
- positive and negative review counts.

The review-count variables are used exclusively to construct the target and are excluded from the predictive feature set to prevent target leakage.

## Target construction

The raw review score is defined as:

```text
(Positive - Negative) / (Positive + Negative) * 100
```

The score ranges from -100 to 100. However, raw scores can be unstable for games with very few reviews.

To address this issue, the project applies a Bayesian smoothing procedure inspired by the IMDb weighted-rating approach. Games with fewer reviews are shrunk more strongly toward the global review balance, while games with many reviews remain closer to their observed raw score.

The final regression target is:

```text
review_score_bayes
```

## Methodology

The workflow consists of:

1. loading the cleaned Steam dataset;
2. filtering games without review information;
3. constructing the raw and Bayesian-smoothed targets;
4. engineering numerical features;
5. creating a fixed 80/20 train-test split;
6. comparing Dummy, Ridge, Lasso and numerical-only XGBoost models;
7. encoding Genres and Categories as multilabel binary features;
8. extending the feature space with Steam Tags;
9. selecting feature blocks through training-set cross-validation;
10. tuning the final XGBoost model with `GridSearchCV`;
11. evaluating the selected model on the held-out test set;
12. performing permutation importance and error analysis.

No model-selection, feature-selection or hyperparameter-selection decision is based on held-out test performance.

## Model interpretation

Permutation importance is used to measure the increase in RMSE produced by shuffling each feature while leaving the remaining predictors unchanged.

The results show that the final model uses both numerical engagement signals and multilabel metadata. The importance values should be interpreted as measures of predictive usefulness rather than causal effects.

## Error analysis

The model performs best around the central portion of the target distribution.

Mean signed prediction errors show a clear regression-to-the-mean pattern:

- games in the lowest reception quintile are substantially overestimated;
- games in the highest reception quintile are underestimated;
- predictions are better centered for games with intermediate reception.

This suggests that extreme reception may depend on qualitative or contextual information not represented in the structured dataset.

## Limitations

- The analysis is predictive, not causal.
- The model cannot be used for pre-release forecasting.
- The target is NPS-inspired but is not a true survey-based Net Promoter Score.
- Steam reviews are self-selected and may not represent the full player base.
- Tags may reflect community perceptions and platform conventions rather than objective game characteristics.
- The dataset does not include review text, bug reports, update history, marketing exposure, wishlist data, developer reputation or external community sentiment.

## Repository structure

```text
steam-user-reception-modeling/
├── app/
│   ├── data/
│   │   ├── error_by_quintile.csv
│   │   ├── model_metrics.csv
│   │   ├── permutation_importance.csv
│   │   └── test_predictions.csv.gz
│   ├── pages/
│   │   ├── game_explorer.py
│   │   ├── methodology.py
│   │   ├── model_performance.py
│   │   └── overview.py
│   ├── export_app_data.py
│   └── utils.py
├── notebooks/
│   └── steam_user_reception_modeling.ipynb
├── reports/
│   ├── steam_user_reception_technical_report.pdf
│   └── figures/
│       ├── raw_vs_bayesian_review_score.png
│       ├── permutation_importance.png
│       ├── predicted_vs_actual.png
│       └── error_by_reception_quintile.png
├── src/
│   ├── evaluation.py
│   └── features.py
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── requirements-analysis.txt
├── requirements.txt
└── streamlit_app.py
```

## Technical report

The complete technical report is available here:

[Read the technical report](reports/steam_user_reception_technical_report.pdf)

## How to run

The deployed Streamlit application is available at:

**[Launch the Steam User Reception Modeling app](https://steam-user-reception-modeling.streamlit.app/)**

The following instructions explain how to run the application or reproduce the modeling analysis locally.

### 1. Clone the repository

```bash
git clone https://github.com/StefanoTriscali/steam-user-reception-modeling.git
cd steam-user-reception-modeling
```

### 2. Create a virtual environment

Using an isolated Python environment is recommended to avoid dependency conflicts and restrictions imposed by system-managed Python installations.

```bash
python -m venv .venv
```

If `python` is not available on Linux or macOS, use `python3` instead:

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

Choose the command corresponding to your operating system and shell.

**Linux or macOS — Bash/Zsh**

```bash
source .venv/bin/activate
```

**Linux or macOS — Fish**

```fish
source .venv/bin/activate.fish
```

**Windows — PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows — Command Prompt**

```bat
.venv\Scripts\activate.bat
```

After activation, the terminal prompt should display the name of the virtual environment.

Upgrade pip inside the environment:

```bash
python -m pip install --upgrade pip
```

### 4A. Run the Streamlit application locally

Install the lightweight application dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run streamlit_app.py
```

Streamlit will display the local application address in the terminal, typically:

```text
http://localhost:8501
```

### 4B. Reproduce the modeling analysis

Install the complete analysis dependencies:

```bash
python -m pip install -r requirements-analysis.txt
```

Launch the final notebook:

```bash
python -m jupyter notebook notebooks/steam_user_reception_modeling.ipynb
```

The notebook downloads the cleaned dataset through KaggleHub. Kaggle credentials may be required depending on the execution environment.

### 5. Deactivate the environment

When finished, close the application or notebook and deactivate the virtual environment:

```bash
deactivate
```

## Reproducibility

Before publishing, the notebook was restarted and executed from beginning to end using the fixed random seed included in the analysis.

The main libraries used for the modeling analysis are:

- pandas;
- NumPy;
- scikit-learn;
- XGBoost;
- Matplotlib;
- KaggleHub.

The interactive application is built with:

- Streamlit;
- Plotly.

## Project status

The modeling pipeline, final notebook, technical report and interactive Streamlit application are complete.

## Potential extensions

Future work could include:

- incorporating review-text sentiment;
- evaluating the model with a chronological train-test split;
- adding developer and publisher reputation measures;
- studying prediction errors across Genres, Tags and popularity levels;
- investigating the relationship between niche positioning, product differentiation and the growing relevance of indie games on Steam.

### Author

**Stefano Triscali**

### License

This project is released under the [MIT License](LICENSE).
