# Exoplanet Classification (Kepler KOI)

Binary classification of Kepler Objects of Interest (KOI) as `CONFIRMED` exoplanets or
`FALSE POSITIVE` detections, using tabular features from NASA's Kepler mission.

## Objective

Build a machine learning model that predicts whether a Kepler Object of Interest is:

- **CONFIRMED** — a confirmed or validated exoplanet
- **FALSE POSITIVE** — a signal determined not to be caused by an exoplanet

Objects classified as `CANDIDATE` are excluded, since their final status is not yet resolved.

## Results

Three models were trained and compared on a held-out test set (20%, stratified split):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting (tuned)** | **91.00%** | 85.52% | **91.44%** | **88.38%** | **96.84%** |
| Random Forest (tuned) | 90.38% | **85.54%** | 89.44% | 87.44% | 96.78% |
| Logistic Regression (baseline) | 77.56% | 64.67% | 88.34% | 74.67% | 87.75% |

**Best model: Gradient Boosting**, saved at `models/best_model.joblib`.

For reference, a naive classifier that always predicts the majority class (`FALSE POSITIVE`)
achieves ~62.6% accuracy — both tree-based models clear that bar by a wide margin.

## Dataset

Source: [Kepler Objects of Interest — NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
Downloaded automatically in `01_download_data.ipynb`.

7,586 rows × 10 columns, already limited to `CONFIRMED` / `FALSE POSITIVE` (no `CANDIDATE` rows
in this extract).

| Column | Description |
|---|---|
| `kepoi_name` | Unique identifier of the Kepler Object of Interest | Text |
| `koi_disposition` | Final classification of the object | Category |
| `koi_period` | Time between consecutive transits, approximately equal to the orbital period | Days |
| `koi_duration` | Duration of a single transit event | Hours |
| `koi_depth` | Decrease in stellar brightness during the transit | Parts per million |
| `koi_impact` | Distance between the transit path and the center of the stellar disk, normalized by the stellar radius | Dimensionless |
| `koi_prad` | Estimated radius of the object | Earth radii |
| `koi_teq` | Estimated equilibrium temperature of the object | Kelvin |
| `koi_insol` | Amount of stellar radiation received relative to Earth | Earth flux |
| `koi_model_snr` | Signal-to-noise ratio of the fitted transit model | Dimensionless |

### Key EDA findings

- **Class balance**: 63.8% FALSE POSITIVE / 36.2% CONFIRMED — moderate imbalance, handled via
  stratified splitting and `class_weight="balanced"` where supported.
- **Missing data is informative**: 259 rows (3.4%) have missing values in `koi_depth`, `koi_impact`,
  `koi_prad`, `koi_teq`, `koi_model_snr` (same rows). These missing rows are overwhelmingly
  FALSE POSITIVE (5.3% missing rate vs. 0.07% for CONFIRMED) — likely signals rejected early enough
  in Kepler's own pipeline that downstream parameters were never computed. Dropped for this project;
  worth revisiting as an engineered `is_missing` flag.
- **Skewed distributions**: 7 of 8 numeric features are strongly right-skewed; `log1p` transform applied before scaling for the linear model.
- **Extreme values are signal, not noise**: some `koi_prad` values exceed 20,000 Earth radii —
  physically impossible for a real planet, and these rows are exclusively FALSE POSITIVE. Most
  likely eclipsing binaries misidentified as transits. Not capped/removed, since they separate the
  classes well.
- **All 8 features are statistically significant** (Mann-Whitney U test, p < 0.001 for every
  feature). `koi_prad`, `koi_impact`, `koi_insol`, and `koi_teq` are the strongest individual
  separators.
- **`koi_teq` and `koi_insol` are correlated (r ≈ 0.85)** — expected physically (insolation drives
  equilibrium temperature); a mild multicollinearity risk for linear models.

Full analysis, plots, and statistical tests: `notebooks/02_EDA.ipynb`.

## Repository structure

```
exoplanet_classification/
├── data/
│   ├── kepler_exoplanet_classification.csv   # raw data (downloaded in 01_download_data.ipynb)
│   └── processed/                            # train/test splits, produced by 03_preprocessing.ipynb
│       ├── X_train.csv / X_test.csv          # scaled + log-transformed features
│       ├── X_train_raw.csv / X_test_raw.csv  # unscaled features (for tree models)
│       ├── y_train.csv / y_test.csv
│       └── preprocessor.joblib
├── models/
│   ├── baseline_logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   ├── best_model.joblib                     # selected in 07_model_comparison.ipynb
│   ├── best_model_name.txt
│   ├── baseline_results.csv                  # metrics, appended by each model notebook
│   └── model_comparison_final.csv            # final sorted comparison table
├── notebooks/
│   ├── 01_download_data.ipynb                # fetch raw data from NASA Exoplanet Archive
│   ├── 02_EDA.ipynb                          # exploratory data analysis
│   ├── 03_preprocessing.ipynb                # cleaning, split, log-transform, scaling
│   ├── 04_baseline_model.ipynb               # logistic regression baseline
│   ├── 05_random_forest.ipynb                # random forest + GridSearchCV
│   ├── 06_gradient_boosting.ipynb            # gradient boosting + GridSearchCV
│   └── 07_model_comparison.ipynb             # final comparison table + best model selection
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd exoplanet_classification
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the pipeline

Run the notebooks in order:

1. `01_download_data.ipynb` — downloads the raw dataset into `data/`
2. `02_EDA.ipynb` — exploratory analysis
3. `03_preprocessing.ipynb` — cleans data, splits train/test, saves processed features to
   `data/processed/`
4. `04_baseline_model.ipynb` — trains and evaluates the logistic regression baseline
5. `05_random_forest.ipynb` — trains and tunes the random forest model
6. `06_gradient_boosting.ipynb` — trains and tunes the gradient boosting model
7. `07_model_comparison.ipynb` — builds the final comparison table and saves the best model

Each modeling notebook (04–06) appends its results to `models/baseline_results.csv`, so
notebook 07 can be re-run any time to regenerate the comparison table from whatever models
have been trained so far.

## Using the trained model

```python
import joblib
import pandas as pd

model = joblib.load("models/best_model.joblib")

# gradient boosting / random forest expect raw (unscaled) features - see notebooks 05/06
# if the best model is logistic regression, use the fitted preprocessor first:
# preprocessor = joblib.load("data/processed/preprocessor.joblib")
# X_scaled = preprocessor.transform(X_new)

X_new = pd.DataFrame([{
    "koi_period": 9.49,
    "koi_duration": 2.96,
    "koi_depth": 615.8,
    "koi_impact": 0.146,
    "koi_prad": 2.26,
    "koi_teq": 793.0,
    "koi_insol": 93.59,
    "koi_model_snr": 35.8,
}])

prediction = model.predict(X_new)[0]
probability = model.predict_proba(X_new)[0, 1]

print("CONFIRMED" if prediction == 1 else "FALSE POSITIVE", f"(p={probability:.3f})")
```

## Tech stack

- Python 3.13, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, joblib

