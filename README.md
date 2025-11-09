<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Status](https://img.shields.io/badge/status-active-success.svg) ![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter) ![Domain](https://img.shields.io/badge/domain-Banking-blue.svg) ![XGBoost](https://img.shields.io/badge/XGBoost-%23FF6600.svg?logo=xgboost&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-%2302569B.svg?logo=lightgbm&logoColor=white) ![CatBoost](https://img.shields.io/badge/CatBoost-%23FFCC00.svg?logo=catboost&logoColor=black) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)

# Home Loan Default Prediction

Machine learning classification model for predicting loan defaults from 58.4 million records. Achieves 83% accuracy and 52.8% recall with tuned XGBoost classifier at threshold 0.60.

</div>

---

## Overview

Predicts home loan default risk from 7 interconnected datasets with 218 features spanning application data, credit bureau history, and payment transactions. Addresses memory constraints (7.7 GB data), missing values (24%), class imbalance (8% default rate), multicollinearity (VIF > 47), and temporal aggregation.

**Final Model:** XGBoost Classifier (200 estimators, depth=5)

- Test: Accuracy = 83%, Recall = 52.8%, ROC AUC = 0.785
- Cross-validation: F1-score = 0.301 +/- 0.0025
- Features: 143 (from 187 engineered), top predictors: EXT_SOURCE_2/3
- Overfitting gap: 1.63%

See [Model_Comparison_Report.md](reports/Model_Comparison_Report.md) for comparison with LightGBM and CatBoost.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Main dependencies: xgboost, lightgbm, catboost, insightfulpy==0.1.7

### Load Trained Model

```python
import joblib
model = joblib.load('models/final_xgb_model.pkl')
predictions = model.predict_proba(X_new)[:, 1]  # Default probabilities
default_flags = (predictions >= 0.60).astype(int)  # Tuned threshold
```

### Run Full Pipeline

Open `notebooks/PRCP-1006-HomeLoanDef.ipynb` for complete data preparation, modeling, and evaluation workflow (400+ cells).

## Dataset

| Property | Details |
|----------|---------|
| **Source** | Home Credit Default Risk (Kaggle) |
| **Total Records** | 58,441,149 across 7 datasets |
| **Main Dataset** | application_train (307,511 rows, 122 columns) |
| **Target** | Binary (1=Default: 8.07%, 0=Non-default: 91.93%) |
| **Split** | 246,008 train / 61,503 test (stratified) |
| **Memory** | 7.7 GB raw, 1.17 GB tuned (68.5% reduction) |

**Datasets:**
- application_train: Main dataset with loan outcomes
- bureau: Credit history from other institutions (1.7M rows)
- bureau_balance: Monthly credit balances (27.3M rows)
- POS_CASH_balance: POS/cash loan snapshots (10M rows)
- credit_card_balance: Monthly card balances (3.8M rows)
- previous_application: Prior applications (1.7M rows)
- installments_payments: Payment history (13.6M rows)

Download: [PRCP-1006-HomeLoanDef.zip](https://d3libtxj3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip)

## Project Structure

**Core directories:**
- `data/raw/` - 7 CSV files (application_train, bureau, etc.)
- `data/processed/` - Aggregated client-level dataset
- `notebooks/` - Main analysis (PRCP-1006-HomeLoanDef.ipynb)
- `src/` - Reusable modules (utils, visualization, statistical analysis, model evaluation)
- `models/` - Trained model (final_xgb_model.pkl)
- `reports/` - Complete analysis, model comparison, challenges reports
- `results/` - EDA outputs, figures, model performance

## Working with the Notebook

**Import pattern used:**
The notebook imports functions from src/ modules using:
```python
import sys
sys.path.append('..')
from src.utils import memory_usage, reduce_mem_usage, garbage_collection
from src.visualization import plot_histograms, plot_all_evaluation_metrics
from src.statistical_analysis import chi_square_test, spearman_correlation_with_target, calculate_vif
from src.model_evaluation import evaluate_model, threshold_analysis
```

**Running analysis:**
The notebook contains the full ML pipeline. Execute cells sequentially for:
1. Data loading and memory tuning (reduce 68.5% memory)
2. Statistical analysis (normality tests, chi-square, Spearman correlation)
3. Feature engineering (ratios, aggregations, temporal features)
4. Multicollinearity resolution (VIF analysis)
5. Model comparison (XGBoost, LightGBM, CatBoost, Random Forest)
6. Hyperparameter tuning (GridSearchCV with 5-fold stratified CV)
7. Threshold tuning (0.50 to 0.60 for business objectives)
8. Final model evaluation and persistence

## Model Training Workflow

**Base model evaluation:**
```python
from src.model_evaluation import evaluate_model

metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
# Returns: Accuracy, Precision, Recall, F1-Score, ROC AUC, CV Accuracy,
#          Training Accuracy, Overfit, Training Time
```

**Threshold tuning:**
```python
from src.model_evaluation import threshold_analysis

df_results, best_threshold = threshold_analysis(
    model, X_test, y_test,
    thresholds=np.arange(0.1, 1.0, 0.1)
)
# Returns performance metrics at each threshold
```

**Handling class imbalance:**
```python
from xgboost import XGBClassifier

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # 11.36

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # Handle 8% default rate
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10,  # L2 regularization
    reg_alpha=1,    # L1 regularization
    random_state=42
)
```

## Statistical Analysis Functions

**Normality testing:**
```python
from src.statistical_analysis import normality_test_with_skew_kurt

normal_cols, non_normal_cols = normality_test_with_skew_kurt(df)
# Uses Shapiro-Wilk with skewness/kurtosis analysis
```

**Multicollinearity detection:**
```python
from src.statistical_analysis import calculate_vif

vif_data = calculate_vif(data, threshold=5.0)
# Returns VIF scores; features >5 indicate multicollinearity
```

**Spearman correlation (for non-normal features):**
```python
from src.statistical_analysis import spearman_correlation_with_target

corr_data = spearman_correlation_with_target(
    data,
    non_normal_cols=non_normal_cols,
    target_col='TARGET',
    plot=True
)
```

**Hypothesis testing:**
```python
from src.statistical_analysis import chi_square_test, fisher_exact_test

# For categorical vs TARGET
chi_results = chi_square_test(data, target_col='TARGET')
fisher_results = fisher_exact_test(data, target_col='TARGET')
```

## Memory Tuning

Necessary for handling 58M records:

```python
from src.utils import reduce_mem_usage

# Apply after loading each dataset
at = reduce_mem_usage(at)  # 286 MB to 60 MB (79.2% reduction)
bu = reduce_mem_usage(bu)  # 223 MB to 79 MB (64.7% reduction)
bub = reduce_mem_usage(bub)  # 625 MB to 156 MB (75.0% reduction)
```

## Visualization Functions

**Model evaluation plots:**
```python
from src.visualization import plot_all_evaluation_metrics

plot_all_evaluation_metrics(model, X_test, y_test)
# Generates: ROC curve, Precision-Recall curve, Calibration curve,
#            Confusion matrix, Lift curve, Gain curve
```

## Model Persistence

**Saving and loading:**
```python
import joblib

# Save
joblib.dump(model, 'models/final_xgb_model.pkl')

# Load
model = joblib.load('models/final_xgb_model.pkl')
predictions = model.predict_proba(X_new)[:, 1]
```

## Main Design Decisions

**Model selection criteria (weighted):**
1. Recall (detect defaults) - 35%
2. Generalization (CV stability) - 25%
3. F1-score (precision-recall balance) - 20%
4. Overfitting gap - 15%
5. Training efficiency - 5%

**Why XGBoost over LightGBM:**
- LightGBM achieved higher recall (68.46%) but lower precision (19.11%)
- XGBoost provided higher balance: 52.8% recall, 24.4% precision at threshold 0.60
- XGBoost showed highest CV stability (std dev 0.0025 vs 0.0032)
- Lower overfitting: 1.63% gap vs 1.39% (marginal difference)
- Trade-off: Accept 15.7 pp lower recall for 5.3 pp higher precision and higher stability

**Why threshold 0.60 instead of 0.50:**
- F1-score change: 0.302 to 0.333
- Accuracy change: 75.3% to 83.0%
- Precision change: 19.6% to 24.4%
- Recall reduction acceptable: 66.3% to 52.8% (still detects half of defaults)
- Business impact: Reduces false positives from 10,143 to 8,088 (20% reduction)

**Feature engineering highlights:**
- WEIGHTED_EXT_SOURCE: Combines 3 external credit scores (rank 3 in importance)
- CREDIT_UTILIZATION_RATIO: Balance / credit limit (rank 9)
- Payment behavior aggregations: Mean late payment rate, payment trends
- Temporal features: Recency and frequency of credit inquiries

## Reports

Detailed analysis in `reports/`:

- [Complete_Data_Analysis_Report.md](reports/Complete_Data_Analysis_Report.md) - Full methodology (47 pages, 18,500 words)
- [Model_Comparison_Report.md](reports/Model_Comparison_Report.md) - Model selection rationale
- [Challenges_Report.md](reports/Challenges_Report.md) - 9 technical challenges and solutions

## Main Challenges Addressed

1. **Memory Management** - Reduced 7.7 GB to 1.17 GB via dtype tuning
2. **Missing Data** - Context-aware imputation for 24% missing values
3. **Outliers** - Winsorization at 99th percentile (income max: 117M)
4. **Data Leakage** - Removed 12 features through temporal validation
5. **Multicollinearity** - VIF reduction from 47.3 to 3.2
6. **High Cardinality** - Frequency encoding for 58 organization types
7. **Temporal Aggregation** - Compressed 54.7M rows to 307K client-level
8. **Class Imbalance** - scale_pos_weight + threshold tuning
9. **Overfitting** - L1/L2 regularization + early stopping

## Development

### Environment Setup

```bash
pip install -r requirements.txt
```

### Jupyter Notebook

```bash
# Launch notebook
jupyter notebook notebooks/PRCP-1006-HomeLoanDef.ipynb

# Or Jupyter Lab
jupyter lab
```

### Code Quality

```bash
# Format code
black src/
isort src/ --profile black

# Format notebooks
nbqa black notebooks/
```

---

## Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 83.0% | At threshold 0.60 |
| **Precision** | 24.4% | 1 in 4 flagged defaults are real |
| **Recall** | 52.8% | Detects half of actual defaults |
| **F1-Score** | 0.333 | Balanced precision-recall |
| **ROC AUC** | 0.785 | Solid class separation |
| **CV F1** | 0.301 +/- 0.0025 | Highly stable |
| **Training Time** | 98 seconds | 200 estimators |
| **Inference Time** | <1ms per prediction | After model loading |

**Confusion Matrix (threshold 0.60):**
- True Negatives: 48,450 (85.7% of non-defaults correctly identified)
- False Positives: 8,088 (14.3% of non-defaults flagged)
- False Negatives: 2,356 (47.2% of defaults missed)
- True Positives: 2,609 (52.8% of defaults detected)

**Top 5 Features:**
1. EXT_SOURCE_3 (18.2%) - External credit score
2. EXT_SOURCE_2 (17.4%) - External credit score
3. WEIGHTED_EXT_SOURCE (9.6%) - Engineered combined score
4. DAYS_BIRTH (6.8%) - Age
5. NAME_EDUCATION_TYPE (5.1%) - Education level

---

## License & Author

- MIT License - Copyright (c) 2025 Dhanesh B. B.
- GitHub: [https://github.com/dhaneshbb](https://github.com/dhaneshbb)
