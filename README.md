# Steam User Reception Modeling

This project models Steam game user reception using game metadata, post-release engagement indicators and multilabel categorical metadata such as Genres, Categories and Tags.

The target variable is a Bayesian-smoothed review reception score derived from Steam positive and negative review counts. The smoothing step reduces instability for games with very few reviews.

> Important: this is a post-release reception modeling project, not a pre-release success forecasting model, because several predictors such as estimated owners, playtime and Peak CCU are observed after release.

## Project goal

The goal is to answer the following question:

> Can Steam game metadata and post-release engagement signals predict user reception?

The project focuses on predictive modeling, feature engineering, model comparison and error analysis.

## Dataset

The project uses a cleaned Steam games dataset downloaded through KaggleHub.

The dataset includes game-level information such as pricing, release date, estimated owners, playtime indicators, concurrent user activity, review counts and multilabel categorical metadata.

Main feature groups include:

* Numerical metadata and engagement indicators
* Estimated ownership
* Release year and release month
* Average and median playtime
* Peak concurrent users
* Genres
* Categories
* Tags

## Target variable

The raw review score is computed as:

```
(Positive - Negative) / (Positive + Negative) * 100
```

This produces a score between -100 and 100.

However, raw review scores can be unstable for games with very few reviews. To reduce this issue, the final regression target is a Bayesian-smoothed review reception score. Games with fewer reviews are shrunk more strongly toward the global average, while games with many reviews remain closer to their raw score.

## Methodology

The workflow includes:

1. Dataset loading and cleaning
2. Feature engineering
3. Bayesian-smoothed target construction
4. Fixed train/test split
5. Numerical baseline models
6. Multilabel encoding of Genres and Categories
7. Tags extension
8. Feature block selection using training-set cross-validation
9. Final XGBoost tuning with GridSearchCV
10. Held-out test evaluation
11. Feature importance and permutation importance
12. Error analysis

## Models

The project compares several regression models:

* DummyRegressor as a naive baseline
* Ridge regression
* Lasso regression
* XGBoost using numerical features only
* XGBoost with Genres and Categories
* Final tuned XGBoost with numerical features, Genres, Categories and Tags

## Results

| Model                              |   RMSE |    MAE |     R² |
| ---------------------------------- | -----: | -----: | -----: |
| Dummy mean                         | 21.035 | 15.585 | -0.000 |
| XGB numeric only                   | 19.025 | 13.893 |  0.182 |
| XGB + Genres/Categories            | 17.974 | 13.115 |  0.270 |
| Final XGB + Genres/Categories/Tags | 17.379 | 12.608 |  0.317 |

The final model improves over both the naive baseline and the numerical-only model. The strongest performance is achieved by combining numerical post-release indicators with multilabel categorical metadata, especially Tags.

## Key findings

Adding Genres and Categories improves performance compared to numerical features alone.

Adding Tags provides a further improvement, suggesting that granular descriptors of gameplay style, mechanics, themes and audience expectations contain useful predictive information.

XGBoost outperforms the regularized linear baselines, suggesting that the relationship between Steam metadata and user reception is partly non-linear.

The final model still tends to regress toward the mean. It tends to overestimate poorly received games and underestimate highly received games.

## Error analysis

The error analysis shows that aggregate metrics such as RMSE and MAE do not fully describe model behavior.

The model performs better around the central part of the target distribution, while extreme reception outcomes remain harder to predict. This suggests that metadata and engagement indicators are informative, but not sufficient to fully explain very negative or very positive user reception.

## Interpretation

Feature importance and permutation importance are used to understand which variables contribute most to the final model.

The interpretation should be read as predictive, not causal. A feature being important for prediction does not imply that it directly causes higher or lower user reception.

## Limitations

This project is predictive, not causal.

It should not be used as a pre-release success forecasting model, since several predictors are only available after release.

The model does not include potentially relevant information such as:

* Review text sentiment
* Bug reports
* Update history
* Developer and publisher reputation
* Community sentiment
* Marketing exposure
* Wishlist data
* External social media signals

The target is also based on Steam review counts, which may reflect user behavior, visibility and platform dynamics in addition to game quality.

## Repository structure

```
steam-user-reception-modeling/
├── notebooks/
│   └── steam_user_reception_modeling.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── features.py
│   └── evaluation.py
├── models/
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run

Install dependencies:

```
pip install -r requirements.txt
```

Then open and run the notebook:

```
notebooks/steam_user_reception_modeling.ipynb
```

The notebook downloads the dataset using KaggleHub.

## Project status

This project is complete as a modeling notebook and portfolio case study.

Planned extensions may include a technical report, selected figures in the README, a lightweight Streamlit dashboard and an optional FastAPI inference endpoint.

## Future work

Possible future improvements include:

* Refactoring the notebook into reusable training scripts
* Saving and loading the final trained model
* Adding a Streamlit dashboard for interactive exploration
* Adding a FastAPI endpoint for model inference
* Incorporating review text sentiment
* Comparing model behavior across release years
* Monitoring prediction errors by game genre, tag and popularity level

## Author

Stefano Triscali
