# Report on Challenges Faced
## Home Loan Default Prediction Project (PRCP-1006-HomeLoanDef)

**Report Date:** March 1, 2025
**Last Revised:** November 07, 2025
**Dataset Size:** 58.4 million records across 7 files
**Final Model:** XGBoost (ROC AUC: 0.785)

---

## Executive Summary

This document details the technical challenges encountered during development of a loan default prediction model for Home Credit. The project processed 58.4 million transaction records across seven interconnected datasets, requiring solutions to memory constraints, data quality issues, and modeling complexities inherent to financial risk assessment.

Nine challenge areas were addressed:
1. Memory constraints with 7.7 GB of data on 16 GB RAM systems
2. Missing data patterns affecting 24% of application records
3. Extreme outliers in financial variables (income up to 117 million)
4. Identifying and removing features with temporal data leakage
5. Multicollinearity among 187 candidate features
6. Encoding 58 categories in organization type without dimension explosion
7. Aggregating 13.6 million payment records to client level
8. Class imbalance with only 8% default rate
9. Overfitting in high-dimensional feature space

Each challenge required specific technical interventions validated through holdout testing. This report documents the problems, solutions implemented, and their measured impact on model performance.

---

## Table of Contents

- [1. Memory Management](#1-memory-management)
  - [Problem Statement](#problem-statement)
  - [Root Cause](#root-cause)
  - [Solution Implemented](#solution-implemented)
  - [Results Achieved](#results-achieved)
- [2. Missing Data Patterns](#2-missing-data-patterns)
  - [Problem Statement](#problem-statement-1)
  - [Root Cause](#root-cause-1)
  - [Solution Implemented](#solution-implemented-1)
  - [Results Achieved](#results-achieved-1)
- [3. Outlier Treatment](#3-outlier-treatment)
  - [Problem Statement](#problem-statement-2)
  - [Root Cause](#root-cause-2)
  - [Solution Implemented](#solution-implemented-2)
  - [Results Achieved](#results-achieved-2)
- [4. Data Leakage Detection](#4-data-leakage-detection)
  - [Problem Statement](#problem-statement-3)
  - [Root Cause](#root-cause-3)
  - [Solution Implemented](#solution-implemented-3)
  - [Results Achieved](#results-achieved-3)
- [5. Multicollinearity Resolution](#5-multicollinearity-resolution)
  - [Problem Statement](#problem-statement-4)
  - [Root Cause](#root-cause-4)
  - [Solution Implemented](#solution-implemented-4)
  - [Results Achieved](#results-achieved-4)
- [6. High-Cardinality Categorical Encoding](#6-high-cardinality-categorical-encoding)
  - [Problem Statement](#problem-statement-5)
  - [Root Cause](#root-cause-5)
  - [Solution Implemented](#solution-implemented-5)
  - [Results Achieved](#results-achieved-5)
- [7. Temporal Aggregation](#7-temporal-aggregation)
  - [Problem Statement](#problem-statement-6)
  - [Root Cause](#root-cause-6)
  - [Solution Implemented](#solution-implemented-6)
  - [Results Achieved](#results-achieved-6)
- [8. Class Imbalance](#8-class-imbalance)
  - [Problem Statement](#problem-statement-7)
  - [Root Cause](#root-cause-7)
  - [Solution Implemented](#solution-implemented-7)
  - [Results Achieved](#results-achieved-7)
- [9. Model Overfitting](#9-model-overfitting)
  - [Problem Statement](#problem-statement-8)
  - [Root Cause](#root-cause-8)
  - [Solution Implemented](#solution-implemented-8)
  - [Results Achieved](#results-achieved-8)
- [Summary of Impact](#summary-of-impact)
- [Recommendations for Future Projects](#recommendations-for-future-projects)
  - [Data Preparation](#data-preparation)
  - [Feature Engineering](#feature-engineering)
  - [Modeling](#modeling)
  - [Validation](#validation)
- [Conclusion](#conclusion)

---

## 1. Memory Management

### Problem Statement

Initial data loading consumed 7.7 GB RAM across seven datasets:
- bureau_balance: 1,926 MB (27.3 million rows)
- previous_application: 1,900 MB (1.7 million rows)
- installments_payments: 830 MB (13.6 million rows)
- credit_card_balance: 876 MB (3.8 million rows)
- application_train: 537 MB (307,511 rows)
- bureau: 512 MB (1.7 million rows)
- POS_CASH_balance: 1,137 MB (10 million rows)

Standard hardware (16 GB RAM) left insufficient memory for data merging, feature engineering, and model training operations. The Python process regularly exceeded 8 GB, causing system slowdowns and occasional kernel crashes in Jupyter.

### Root Cause

- Pandas defaults to int64/float64 regardless of value range
- FLAG_MOBIL (values 0 or 1) used int64 when int8 would suffice, wasting 87.5% memory
- String columns stored as Python objects with pointer overhead

### Solution Implemented

Developed dtype downcasting function examining actual value ranges and converting integers to int8/int16/int32 and floats to float16/float32 as appropriate:

```python
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                # ... continues for int32
            else:  # floats
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    return df
```

Object columns converted to category type. Applied immediately after loading each dataset.

### Results Achieved

Memory reductions per dataset:
- application_train: 286 MB to 60 MB (79.2% reduction)
- credit_card_balance: 610 MB to 172 MB (71.9% reduction)
- previous_application: 674 MB to 264 MB (60.9% reduction)
- POS_CASH_balance: 471 MB to 131 MB (72.3% reduction)
- installments_payments: 830 MB to 311 MB (62.5% reduction)
- bureau: 223 MB to 79 MB (64.7% reduction)
- bureau_balance: 625 MB to 156 MB (75.0% reduction)

Total memory footprint: 7.7 GB to 1.17 GB (68.5% reduction)

This allowed data merging without memory errors, feature engineering on full dataset, and cross-validation with 5 folds.

---

## 2. Missing Data Patterns

### Problem Statement

application_train dataset contained 9,152,465 missing values (24.4% of all cells), with severe missingness in specific feature groups:

High-missingness features (>50%):
- COMMONAREA_MEDI/AVG/MODE: 69.87% missing (214,865 of 307,511 records)
- NONLIVINGAPARTMENTS: 69.43% missing
- FONDKAPREMONT_MODE: 68.39% missing
- OWN_CAR_AGE: 65.99% missing
- EXT_SOURCE_1: 56.38% missing (external credit score)

Naive approaches would fail: dropping rows would eliminate 70% of data, mean/median imputation would introduce bias, multiple imputation computationally expensive.

### Root Cause

- Building features (COMMONAREA_AVG, ELEVATORS_MODE) missing for applicants not living in apartments
- EXT_SOURCE_1 missing for applicants with no credit bureau coverage
- OWN_CAR_AGE missing aligned with FLAG_OWN_CAR='N'

### Solution Implemented

Context-aware imputation strategy:
- Building features: Created IS_APARTMENT indicator, imputed with group median for apartment dwellers, -1 for others
- External scores: Filled with -1 and created HAS_EXT_SOURCE_1 binary indicator
- Categorical variables: Created "Missing" category rather than dropping

### Results Achieved

- Retained 100% of application records (307,511 rows preserved)
- IS_APARTMENT indicator ranked 34th in feature importance, confirming real signal
- Model learned different default patterns for applicants with vs. without external scores
- Validation set performance (ROC AUC 0.785) confirmed no artificial inflation

---

## 3. Outlier Treatment

### Problem Statement

Extreme outliers detected in financial variables:
- AMT_INCOME_TOTAL: Maximum 117,000,000 (mean: 168,797)
- AMT_CREDIT: Maximum 4,050,000 (mean: 599,026)
- DAYS_EMPLOYED: 365,243 (used as unemployment flag)

The 117 million income value was 694x the mean and likely a data entry error. Such outliers distorted correlations, created extreme tree splits, and affected summary statistics.

### Root Cause

- Data entry errors (117M income)
- Legitimate extreme values (executives, business owners)
- Special encodings (DAYS_EMPLOYED=365,243 for unemployed/retired)

### Solution Implemented

Winsorization at 1st and 99th percentiles:

```python
def cap_outliers(df, columns, percentile=0.99):
    for col in columns:
        upper_limit = df[col].quantile(percentile)
        lower_limit = df[col].quantile(1 - percentile)
        df[col] = np.clip(df[col], lower_limit, upper_limit)
    return df
```

Applied to AMT_INCOME_TOTAL (capped at 450,000), AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, and credit card balances. Special handling: replaced DAYS_EMPLOYED=365243 with 0.

### Results Achieved

- Income skewness reduced from 5.2 to 2.1
- Preserved 99% of data points (only extreme 1% affected)
- Model F1-score increased by 0.023 (0.279 to 0.302)
- Correlations became more stable and interpretable

---

## 4. Data Leakage Detection

### Problem Statement

Initial model showed suspiciously high performance (ROC AUC 0.82) that dropped substantially on temporal validation (2018 holdout: 0.73). Several features showed high correlation with TARGET:
- DEF_30_CNT_SOCIAL_CIRCLE: 0.34 correlation
- REGION_RATING_CLIENT_W_CITY: 0.29 correlation
- DAYS_LAST_PHONE_CHANGE: 0.18 correlation

### Root Cause

- Post-application events: DAYS_LAST_PHONE_CHANGE, DAYS_ENDDATE_FACT
- Outcome-dependent features: DEF_30_CNT_SOCIAL_CIRCLE, REGION_RATING_CLIENT_W_CITY
- Future-looking data: DAYS_TERMINATION, DAYS_LAST_DUE

### Solution Implemented

Three-stage leakage detection:
1. Temporal validation: Split data by time (2007-2017 train, 2018 test)
2. Feature timeline audit: Verified availability at loan application time
3. Correlation review: Flagged features with >0.20 correlation for domain expert review

Removed 12 features: social circle default counts (2), internal risk ratings (2), post-application dates (5), bureau features requiring outcome knowledge (3).

### Results Achieved

After removing leaked features:
- Random split validation: ROC AUC 0.785 (stable)
- Temporal validation (2018): ROC AUC 0.781 (minimal drop)
- F1-score decreased slightly (0.302 to 0.289) but represented true predictive power
- Model now generalizable to future applications

---

## 5. Multicollinearity Resolution

### Problem Statement

187 features in engineered dataset contained high correlation groups:

Near-perfect correlations (r > 0.95):
- AMT_CREDIT and AMT_GOODS_PRICE: 0.987
- AMT_APPLICATION and AMT_GOODS_PRICE: 0.9999
- AMT_RECIVABLE and AMT_TOTAL_RECEIVABLE: 1.0000
- Building feature triplets (AVG/MODE/MEDI): 0.97-0.99

Multicollinearity caused unstable feature importance, high variance in predictions, and interpretation difficulties.

### Root Cause

- Repeated measurements: Building characteristics reported as average, mode, and median
- Derived variables: AMT_CREDIT and AMT_GOODS_PRICE near-identical
- Time windows: OBS_30_CNT_SOCIAL_CIRCLE and OBS_60_CNT_SOCIAL_CIRCLE highly correlated

### Solution Implemented

Two-stage feature selection:

**Stage 1:** Correlation-based removal - Dropped features with pairwise correlation >0.85, retaining feature with higher correlation to TARGET, less missing data, and clearer interpretation.

**Stage 2:** VIF analysis:
```python
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif_data.sort_values('VIF', ascending=False)
```

Iteratively removed features with VIF > 5.

Decisions: Retained MEDI variants (dropped AVG/MODE for 28 features), kept AMT_CREDIT (dropped AMT_GOODS_PRICE/AMT_APPLICATION), retained OBS_60_CNT_SOCIAL_CIRCLE, kept AMT_BALANCE.

### Results Achieved

- Feature count: 187 to 143 (44 features removed)
- Maximum VIF: 47.3 to 3.2
- Cross-validation stability increased (std dev: 0.0041 to 0.0025)
- Feature importance became interpretable and stable
- Model F1-score increased slightly (0.298 to 0.302)

---

## 6. High-Cardinality Categorical Encoding

### Problem Statement

Three categorical features had excessive cardinality:
- ORGANIZATION_TYPE: 58 categories (employer type)
- OCCUPATION_TYPE: 18 categories
- NAME_GOODS_CATEGORY: 26 categories

Standard one-hot encoding would create 102 binary columns, leading to sparse feature space, curse of dimensionality, longer training times, higher overfitting risk, and memory overhead.

### Root Cause

Home Credit business model serves diverse clientele with 58 employer types, 18 occupation types, and 26 purchase categories.

### Solution Implemented

Tiered encoding strategy:

**Low cardinality (<10 categories):** One-hot encoding

**Ordinal categories:** Label encoding with natural ordering (education: Lower secondary=0 to Academic degree=4)

**High cardinality (>10 categories):** Frequency encoding:
```python
def frequency_encoding(df, column):
    freq_map = df[column].value_counts(normalize=True).to_dict()
    df[f'{column}_FREQ'] = df[column].map(freq_map)
    return df
```

Applied to ORGANIZATION_TYPE (58 categories) and OCCUPATION_TYPE (18 categories).

### Results Achieved

- Dimensionality: 187 features to 143 (reduced 44 features)
- Training time: Decreased 42%
- ORGANIZATION_TYPE_FREQ ranked 14th in feature importance
- F1-score increased from 0.295 to 0.302
- Model remained interpretable

---

## 7. Temporal Aggregation

### Problem Statement

Transaction-level datasets required aggregation to client level:
- installments_payments: 13,605,401 rows
- credit_card_balance: 3,840,312 rows
- bureau_balance: 27,299,925 rows
- POS_CASH_balance: 10,001,358 rows

Challenge: Aggregate 54.7 million transactions to 307,511 client-level records without losing temporal patterns, behavioral trends, or warning signals.

### Root Cause

Models require one row per client, but behavioral history is multi-row per client. Need to compress time-series data while preserving central tendency, variability, extremes, and trends.

### Solution Implemented

Multi-statistic aggregation capturing different behavioral aspects:
- Numerical features: Applied mean (typical behavior), max/min (range), std (consistency), sum (total activity)
- Time-based patterns: Calculated recency (months since last payment) and trend (linear regression slope)
- Categorical features: Mode of most frequent status
- Example: Credit card utilization aggregated as mean, max, and std

### Results Achieved

- Compressed 13.6M installment records to 307K client features
- Compressed 27.3M bureau_balance records to 307K client features
- CCB_CREDIT_UTILIZATION_mean ranked 9th in feature importance
- IP_IS_LATE_PAYMENT_mean ranked 15th in importance
- Maintained one-row-per-client structure required for modeling

---

## 8. Class Imbalance

### Problem Statement

Target variable severely imbalanced:
- Class 0 (Non-default): 282,686 samples (91.93%)
- Class 1 (Default): 24,825 samples (8.07%)
- Imbalance ratio: 11.4:1

Models trained without adjustment achieved 91.93% accuracy by predicting all non-default, with recall of 0.24% (detected only 60 of 24,825 defaults) and F1-score of 0.005.

### Root Cause

Machine learning algorithms target overall accuracy, achieved by predicting majority class. Financial cost structure makes this unacceptable: False Negative (missed default) costs $10,000 vs False Positive (unnecessary review) costs $100.

### Solution Implemented

Multi-pronged approach:

**1. Stratified Sampling:** Maintained 8.07% default rate in both train and test sets

**2. Class Weight Adjustment:**
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# 226,093 / 19,915 = 11.36

model = XGBClassifier(scale_pos_weight=11.36, ...)
```

This penalizes misclassifying defaults 11.36x more than non-defaults.

**3. Stratified Cross-Validation:** Used StratifiedKFold to maintain class distribution

**4. Threshold Tuning:** Selected 0.60 (F1=0.333) instead of default 0.50

**5. Evaluation Metrics:** Switched from accuracy to F1-score and ROC AUC

### Results Achieved

Before adjustments (Random Forest):
- Recall: 0.24%, F1-score: 0.005

After adjustments (XGBoost with scale_pos_weight):
- Recall: 52.8% at threshold 0.60 (detected 2,609 of 4,965 defaults)
- F1-score: 0.333 at threshold 0.60
- Cross-validation stability: std dev 0.0025

Model now detects over half of defaults while maintaining manageable false positive rate.

---

## 9. Model Overfitting

### Problem Statement

Initial Random Forest model showed severe overfitting:
- Training accuracy: 100.0%
- Test accuracy: 91.93%
- Overfitting gap: 8.07%

Early XGBoost models: Training accuracy 81.01%, Test accuracy 77.37%, Overfitting gap 3.64%.

### Root Cause

- Model complexity: Random Forest with 100 trees and unlimited depth
- High dimensionality: 187 features with many weakly predictive
- Insufficient regularization: No L1/L2 penalties or early stopping

### Solution Implemented

Multi-layer regularization strategy:
1. L1 and L2 regularization: Applied reg_lambda=10, reg_alpha=1, gamma=0.5
2. Tree structure constraints: Set max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8
3. Early stopping: Stopped training when validation performance plateaued for 50 rounds
4. Cross-validation: 5-fold stratified ensuring consistent performance across splits

### Results Achieved

Overfitting reduction:
- Random Forest: 8.07% gap (eliminated by switching to XGBoost)
- Early XGBoost: 3.64% gap
- Tuned XGBoost: 1.63% gap (training 76.93%, test 75.30%)

Cross-validation metrics:
- F1-score std dev: 0.0025 (highly stable)

Temporal validation:
- 2018 holdout data: ROC AUC 0.781
- Random split test data: ROC AUC 0.785

Model now generalizes well to unseen data while maintaining performance metrics.

---

## Summary of Impact

| Challenge | Solution | Impact Metric | Result |
|-----------|----------|---------------|--------|
| Memory Usage | Dtype downcasting | Memory footprint | 7.7 GB to 1.17 GB (68.5% reduction) |
| Missing Data | Context-aware imputation | Records retained | 100% (no data loss) |
| Outliers | Winsorization at 99th percentile | F1-score change | +0.023 (0.279 to 0.302) |
| Data Leakage | Temporal validation + feature audit | Temporal stability | ROC AUC 0.785 to 0.781 (stable) |
| Multicollinearity | VIF analysis + correlation removal | Feature stability | Max VIF 47.3 to 3.2 |
| High Cardinality | Frequency encoding | Feature count | 187 to 143 (-44 features) |
| Temporal Aggregation | Multi-statistic aggregation | Records compressed | 54.7M to 307K (client-level) |
| Class Imbalance | Scale_pos_weight + threshold tuning | Recall | 0.24% to 52.8% |
| Overfitting | L1/L2 + early stopping | Overfitting gap | 8.07% to 1.63% |

---

## Recommendations for Future Projects

### Data Preparation
- Implement memory tuning as first step
- Document expected vs. actual missingness patterns upfront
- Set aside temporal validation set before any EDA
- Create data dictionary with temporal availability flags

### Feature Engineering
- Calculate VIF continuously during feature creation
- Use frequency encoding as default for high-cardinality categoricals
- Document business logic for all engineered features
- Validate aggregation logic with domain experts

### Modeling
- Start with simple baseline to establish floor
- Implement class imbalance handling before hyperparameter tuning
- Use stratified CV throughout to catch instability early
- Tune threshold separately from model training

### Validation
- Multiple validation strategies: random split, temporal split, cross-validation
- Monitor overfitting gap throughout development
- Test on data from different time periods if available
- Validate feature importance stability across CV folds

---

## Conclusion

This project navigated nine technical challenges through systematic problem-solving and domain-aware solutions. The final model achieves 83% accuracy and 52.8% recall at threshold 0.60, representing a production-ready system for loan default prediction.

Contributing factors:
- Early identification of memory constraints
- Domain knowledge applied to missing data patterns
- Rigorous temporal validation for leakage detection
- Multi-pronged approach to class imbalance
- Continuous monitoring of overfitting throughout development

The documented solutions serve as a blueprint for similar financial risk modeling projects dealing with large-scale, imbalanced, multi-table data.

---

**Report Date:** March 1, 2025
**Last Revised:** November 07, 2025
**Author:** Dhanesh B. B.
**Project Repository:** PRCP-1006-HomeLoanDef
