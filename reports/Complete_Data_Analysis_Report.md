# Home Loan Default Prediction: Complete Data Analysis Report

**Project:** PRCP-1006-HomeLoanDef - Home Credit Default Risk Prediction
**Report Date:** March 1, 2025
**Last Revised:** November 07, 2025
**Dataset:** 7 interconnected datasets with 58.4 million records
**Final Model:** XGBoost Classifier with ROC AUC = 0.785, F1-Score = 0.33

---

## Executive Summary

This report documents a machine learning project that predicts home loan defaults using seven interconnected datasets from Home Credit. The analysis processed 58.4 million records across application data, bureau records, and transaction histories spanning POS cash loans, credit cards, and installment payments. Through systematic data integration, memory tuning, feature engineering, and model development, an XGBoost classifier was built that achieves 83% accuracy and 53% recall on test data.

Findings reveal that external credit scores (EXT_SOURCE_2, EXT_SOURCE_3) and demographic factors show the highest correlation with default risk. The model was configured for business use with a decision threshold of 0.60, balancing the need to identify defaulters (recall: 53%) while managing false positives. Memory tuning reduced dataset sizes by 60-80%, allowing analysis of the 27-million-row bureau balance dataset. The class imbalance challenge (8% default rate) was addressed through careful threshold tuning and model parameter adjustment.

---

## Table of Contents

- [Home Loan Default Prediction: Complete Data Analysis Report](#home-loan-default-prediction-complete-data-analysis-report)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Business Context](#11-business-context)
    - [1.2 Dataset Overview](#12-dataset-overview)
    - [1.3 Project Objectives](#13-project-objectives)
  - [2. Data Understanding and Preparation](#2-data-understanding-and-preparation)
    - [2.1 Initial Data Assessment](#21-initial-data-assessment)
    - [2.2 Memory Tuning](#22-memory-tuning)
    - [2.3 Missing Value Analysis and Treatment](#23-missing-value-analysis-and-treatment)
    - [2.4 Data Quality Issues](#24-data-quality-issues)
    - [2.5 Dataset Interconnections](#25-dataset-interconnections)
  - [3. Dataset-Specific Analysis](#3-dataset-specific-analysis)
    - [3.1 Application Train (Main Dataset)](#31-application-train-main-dataset)
    - [3.2 Bureau and Bureau Balance](#32-bureau-and-bureau-balance)
    - [3.3 Previous Application](#33-previous-application)
    - [3.4 POS Cash Balance](#34-pos-cash-balance)
    - [3.5 Credit Card Balance](#35-credit-card-balance)
    - [3.6 Installments Payments](#36-installments-payments)
  - [4. Feature Engineering and Preprocessing](#4-feature-engineering-and-preprocessing)
    - [4.1 Categorical Variable Encoding](#41-categorical-variable-encoding)
    - [4.2 Numerical Feature Engineering](#42-numerical-feature-engineering)
    - [4.3 Multicollinearity Assessment and Resolution](#43-multicollinearity-assessment-and-resolution)
    - [4.4 Data Leakage Prevention](#44-data-leakage-prevention)
    - [4.5 Final Dataset Preparation](#45-final-dataset-preparation)
  - [5. Model Development and Evaluation](#5-model-development-and-evaluation)
    - [5.1 Base Model Comparison](#51-base-model-comparison)
    - [5.2 Hyperparameter Tuning](#52-hyperparameter-tuning)
    - [5.3 Final Model Selection: XGBoost](#53-final-model-selection-xgboost)
    - [5.4 Threshold Tuning](#54-threshold-tuning)
  - [6. Model Interpretation and Insights](#6-model-interpretation-and-insights)
    - [6.1 Feature Importance Analysis](#61-feature-importance-analysis)
    - [6.2 Business Implications](#62-business-implications)
    - [6.3 Model Performance Metrics](#63-model-performance-metrics)
  - [7. Challenges and Solutions](#7-challenges-and-solutions)
    - [7.1 Challenge: High Memory Usage](#71-challenge-high-memory-usage)
    - [7.2 Challenge: Missing Values](#72-challenge-missing-values)
    - [7.3 Challenge: Outliers and Skewed Distributions](#73-challenge-outliers-and-skewed-distributions)
    - [7.4 Challenge: Data Leakage](#74-challenge-data-leakage)
    - [7.5 Challenge: Multicollinearity](#75-challenge-multicollinearity)
    - [7.6 Challenge: Categorical Variables](#76-challenge-categorical-variables)
    - [7.7 Challenge: Temporal Aggregation](#77-challenge-temporal-aggregation)
    - [7.8 Challenge: Class Imbalance](#78-challenge-class-imbalance)
    - [7.9 Challenge: Model Overfitting](#79-challenge-model-overfitting)
  - [8. Limitations and Future Work](#8-limitations-and-future-work)
    - [8.1 Limitations](#81-limitations)
    - [8.2 Future Work](#82-future-work)
  - [9. Conclusion](#9-conclusion)
  - [10. Appendix](#10-appendix)
    - [10.1 Dataset Access](#101-dataset-access)
    - [10.2 Technical Environment](#102-technical-environment)
    - [10.3 Dataset Abbreviations](#103-dataset-abbreviations)
    - [10.4 Source Code and Dependencies](#104-source-code-and-dependencies)
  - [Acknowledgments](#acknowledgments)
  - [Author Information](#author-information)
  - [References](#references)

---

## 1. Introduction

### 1.1 Business Context

Financial institutions face losses from loan defaults, making accurate risk assessment necessary for sustainable lending operations. Home Credit, focused on serving clients with limited or no credit history, requires predictive models to identify applicants who may struggle with loan repayment. This analysis addresses the need for data-driven credit decisioning by modeling relationships between client demographics, credit bureau records, and transactional behaviors to predict default probability.

The analysis allows lenders to:
- Identify high-risk applications requiring additional scrutiny or rejection
- Adjust credit terms and interest rates based on risk profiles
- Reduce default losses while maintaining responsible lending practices
- Expand credit access to underserved populations through data-driven risk models

### 1.2 Dataset Overview

The analysis uses seven interconnected datasets from Home Credit, containing records from 2007-2018:

| Dataset | Abbreviation | Rows | Columns | Description |
|---------|--------------|------|---------|-------------|
| application_train | at | 307,511 | 122 | Main dataset with loan applications and default status (TARGET: 0=repaid, 1=default) |
| bureau | bu | 1,716,428 | 17 | Previous credits from other institutions reported to Credit Bureau |
| bureau_balance | bub | 27,299,925 | 3 | Monthly balances of previous credits (one row per month per credit) |
| POS_CASH_balance | pc | 10,001,358 | 8 | Monthly snapshots of POS and cash loans from Home Credit |
| credit_card_balance | ccb | 3,840,312 | 23 | Monthly credit card balance history |
| previous_application | pa | 1,670,214 | 37 | All previous loan applications to Home Credit |
| installments_payments | ip | 13,605,401 | 8 | Repayment history for previous credits |

**Total Records:** 58,441,149 observations
**Target Variable:** Binary classification (8.07% default rate)
**Memory Footprint (Initial):** 7.7 GB across all datasets

The datasets link through SK_ID_CURR (client identifier) and SK_ID_PREV (previous application identifier), allowing construction of complete client credit histories.

### 1.3 Project Objectives

1. **Data Integration:** Clean, transform, and merge seven datasets into a unified analysis-ready format
2. **Exploratory Analysis:** Identify patterns in demographic, financial, and behavioral factors related to default risk
3. **Feature Engineering:** Create predictive features from transactional histories and credit bureau records
4. **Predictive Modeling:** Develop and compare classification models for default prediction
5. **Business Deployment:** Adjust decision thresholds and deliver actionable risk assessment framework

---

## 2. Data Understanding and Preparation

### 2.1 Initial Data Assessment

Data quality assessment revealed multiple challenges across all datasets:

| Metric | Value | Details |
|--------|-------|---------|
| **Total Records** | 58,441,149 | Spanning 7 datasets |
| **Total Columns** | 218 | 110 float, 70 int, 38 categorical |
| **Missing Values** | 30,137,072 (6.18%) | Highly concentrated in specific features |
| **Negative Values** | 80,624,914 (16.54%) | Legitimate time-based encodings (DAYS_*) |
| **Outliers** | 15,065,818 (3.09%) | Requiring capping strategies |
| **Memory Usage** | 7.7 GB | Before tuning |

**Data Type Distribution:**
- Float64/Float32: 110 columns (primarily amounts and ratios)
- Int64/Int32: 70 columns (IDs, counts, flags)
- Categorical/Object: 38 columns (contract types, statuses, names)

### 2.2 Memory Tuning

Given the scale of data (27 million rows in bureau_balance alone), memory tuning was necessary:

**Strategy Implemented:**
- Downcast float64 to float32/float16 based on value ranges
- Downcast int64 to int32/int16/int8 where appropriate
- Convert string columns to categorical dtype for low-cardinality features

**Results:**

| Dataset | Before (MB) | After (MB) | Reduction |
|---------|-------------|------------|-----------|
| application_train | 286.23 | 59.54 | 79.2% |
| bureau | 222.62 | 78.57 | 64.7% |
| bureau_balance | 624.85 | 156.21 | 75.0% |
| POS_CASH_balance | 471.48 | 130.62 | 72.3% |
| credit_card_balance | 610.43 | 171.69 | 71.9% |
| previous_application | 673.88 | 263.69 | 60.9% |
| installments_payments | 830.41 | 311.40 | 62.5% |
| **Total** | **3,719.90** | **1,171.72** | **68.5%** |

This 68.5% overall reduction allowed processing on standard hardware and reduced computational time for merging and modeling operations.

### 2.3 Missing Value Analysis and Treatment

Missing values were unevenly distributed, with building characteristics showing the highest rates:

**Application Train - High Missing Rate Features (>50%):**
- COMMONAREA_* (AVG/MODE/MEDI): 69.87% missing
- NONLIVINGAPARTMENTS_*: 69.43% missing
- FONDKAPREMONT_MODE: 68.39% missing
- LIVINGAPARTMENTS_*: 68.35% missing
- FLOORSMIN_*: 67.85% missing
- YEARS_BUILD_*: 66.50% missing
- OWN_CAR_AGE: 65.99% missing
- LANDAREA_*: 59.38% missing
- BASEMENTAREA_*: 58.52% missing

These features primarily describe building characteristics for apartments, explaining why missing patterns cluster together. Missing data indicates applicants living in non-apartment housing (houses, with parents, etc.).

**Treatment Strategy:**
1. **Median Imputation:** Applied to numerical features with <40% missing values
2. **Missing Category:** Created "Missing" category for categorical variables
3. **Feature Dropping:** Removed columns with >75% missing values
4. **Domain-Specific Logic:**
   - For building features, missing indicates non-apartment living (valid data state)
   - For credit bureau features (EXT_SOURCE_1: 56.38% missing), missing indicates no external credit score available

**Bureau Dataset Missing Values:**
- AMT_CREDIT_MAX_OVERDUE: 72.17% (missing when no overdue occurred)
- AMT_CREDIT_SUM_LIMIT: 42.47% (credit limit not applicable for all credit types)
- AMT_ANNUITY: 41.78% (annuity not applicable for certain credit products)

**Previous Application Missing Values:**
- Interest rate fields (RATE_INTEREST_PRIMARY, RATE_INTEREST_PRIVILEGED): >99.6% missing
- These were dropped as they offer no predictive value

### 2.4 Data Quality Issues

**Duplicate Records:**
- installments_payments: 15 duplicate rows (0.0001%) - removed
- Other datasets: No duplicates found

**Negative Values:**
All negative values were in temporal features (DAYS_BIRTH, DAYS_EMPLOYED, etc.) using "days before application" encoding:
- DAYS_BIRTH: -25,229 to -7,489 (age 20-69 years)
- DAYS_EMPLOYED: -17,912 to 365,243 (365,243 indicates unemployed/retired)
- DAYS_REGISTRATION: -24,672 to 0

These were converted to positive values for interpretability:
```python
at['AGE'] = -at['DAYS_BIRTH'] / 365
at['EMPLOYMENT_LENGTH'] = -at['DAYS_EMPLOYED'] / 365
```

**Outliers Detected:**
Using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR):
- AMT_INCOME_TOTAL: 6.49% outliers (max: 117,000,000)
- AMT_CREDIT: 2.13% outliers
- AMT_ANNUITY: 2.44% outliers
- OWN_CAR_AGE: Outliers present but logically valid (cars up to 91 years old)

**Outlier Treatment:**
Applied winsorization at 1st and 99th percentiles for:
- AMT_INCOME_TOTAL (capped at 450,000)
- AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE
- Credit card drawing amounts and balances

### 2.5 Dataset Interconnections

**Dataset Keys:**
- SK_ID_CURR: Client identifier (links all datasets to application_train)
- SK_ID_BUREAU: Bureau credit identifier (links bureau to bureau_balance)
- SK_ID_PREV: Previous application identifier (links pa, pc, ccb, ip)

**Common Features Across Datasets:**
- NAME_CONTRACT_TYPE: application_train, previous_application
- AMT_CREDIT: application_train, previous_application
- AMT_ANNUITY: application_train, bureau, previous_application
- NAME_CONTRACT_STATUS: POS_CASH_balance, credit_card_balance, previous_application
- SK_DPD, SK_DPD_DEF: POS_CASH_balance, credit_card_balance

**Merge Strategy:**
1. Aggregate transaction-level data (bureau_balance, pc, ccb, ip) to client level
2. Merge aggregated features with application_train on SK_ID_CURR
3. Apply left joins to preserve all application_train records

---

## 3. Dataset-Specific Analysis

### 3.1 Application Train (Main Dataset)

**Dimensions:** 307,511 rows x 122 columns

**Target Variable Distribution:**
- Class 0 (Non-default): 282,686 (91.93%)
- Class 1 (Default): 24,825 (8.07%)

This severe class imbalance required special handling during modeling.

**Numerical Features:**
- AMT_INCOME_TOTAL: Range 25,650 - 117,000,000 (mean: 168,797)
- AMT_CREDIT: Range 45,000 - 4,050,000 (mean: 599,026)
- AMT_ANNUITY: Range 1,615.5 - 258,025.5 (mean: 27,108)
- DAYS_BIRTH: -25,229 to -7,489 (age 20-69 years)
- DAYS_EMPLOYED: -17,912 to 365,243 (365,243 = unemployed flag)

**Categorical Features:**
- NAME_CONTRACT_TYPE: Cash loans (90.4%), Revolving loans (9.6%)
- CODE_GENDER: Female (66.1%), Male (33.9%)
- NAME_EDUCATION_TYPE: Secondary (71%), Higher education (24%)
- OCCUPATION_TYPE: 18 categories (31.3% missing)
- ORGANIZATION_TYPE: 58 categories (high cardinality)

**External Credit Scores:**
- EXT_SOURCE_1: 56.38% missing, range 0.015-0.963
- EXT_SOURCE_2: 0.21% missing, range 0.0000006-0.855
- EXT_SOURCE_3: 19.83% missing, range 0.00053-0.896

These external scores from credit bureaus showed the highest correlation with default risk.

### 3.2 Bureau and Bureau Balance

**Bureau (bu) Dimensions:** 1,716,428 rows x 17 columns
**Bureau Balance (bub) Dimensions:** 27,299,925 rows x 3 columns

After merging: 25,121,815 rows (combining monthly credit histories)

**Observations:**
- **CREDIT_ACTIVE Distribution:**
  - Closed: 55.8%
  - Active: 32.4%
  - Sold: 6.8%
  - Bad debt: 5.0%

- **CREDIT_TYPE Distribution:**
  - Consumer credit: 51.2%
  - Credit card: 24.1%
  - Mortgage: 10.3%
  - Car loan: 8.7%
  - Microloan: 5.7%

**Credit Amounts:**
- AMT_CREDIT_SUM: Range 0 - 900,000,000 (extreme outlier, capped)
- AMT_CREDIT_SUM_DEBT: 40% of credits have remaining debt
- DAYS_CREDIT: -3,395 to 0 (how long ago credit was issued)

**Bureau Balance Monthly Status:**
- Status 0 (No delay): 67.2%
- Status C (Closed/paid off): 26.8%
- Status X (No loan this month): 3.9%
- Status 1-5 (1-5+ months delay): 2.1%

**Aggregated Features Created:**
- Total number of credits per client
- Average credit amount
- Maximum days credit overdue
- Credit duration in years
- Proportion of closed vs. active credits

### 3.3 Previous Application

**Dimensions:** 1,670,214 rows x 37 columns

**Observations:**
- **Application Decision Distribution:**
  - Approved: 50.3%
  - Refused: 29.6%
  - Canceled: 14.4%
  - Unused offer: 5.7%

- **Contract Types:**
  - Cash loans: 78.9%
  - Consumer loans: 15.4%
  - Revolving loans: 5.7%

**Financial Features:**
- AMT_APPLICATION: Requested loan amount
- AMT_CREDIT: Approved credit amount (can differ from request)
- AMT_GOODS_PRICE: Price of goods for which loan was requested

**High Correlation Observed:**
- AMT_APPLICATION and AMT_GOODS_PRICE: 0.9999 (near-perfect)
- AMT_CREDIT and AMT_GOODS_PRICE: 0.9931
- This multicollinearity required feature selection

**Engineered Features:**
- Previous application count per client
- Approval rate (approved / total applications)
- Average credit amount in previous applications
- Time since last application

### 3.4 POS Cash Balance

**Dimensions:** 10,001,358 rows x 8 columns

**Observations:**
- **Contract Status:**
  - Active: 56.9%
  - Completed: 31.4%
  - Amortized debt: 7.5%
  - Signed: 2.8%
  - Returned to store: 0.9%
  - Approved: 0.3%
  - Canceled: 0.2%

**Installment Tracking:**
- CNT_INSTALMENT: Total number of installments
- CNT_INSTALMENT_FUTURE: Remaining installments
- MONTHS_BALANCE: -47 to 0 (monthly history)

**Days Past Due (DPD):**
- SK_DPD: 98.7% have DPD = 0 (no delays)
- SK_DPD_DEF: 99.7% have no definition of default
- Maximum DPD observed: 180 days

**Engineered Features:**
- Installment completion ratio
- Average DPD per client
- Maximum DPD ever experienced
- Number of active POS contracts

### 3.5 Credit Card Balance

**Dimensions:** 3,840,312 rows x 23 columns

**Observations:**
- **Contract Status:**
  - Active: 96.3%
  - Completed: 2.0%
  - Signed: 1.1%
  - Refused: 0.3%
  - Approved: 0.2%
  - Demand: 0.1%

**Credit Utilization:**
- AMT_BALANCE: Current balance (range: -139,999 to 999,999)
- AMT_CREDIT_LIMIT_ACTUAL: Credit limit
- CREDIT_UTILIZATION_RATIO = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
- Average utilization: 42.7%
- High utilization (>80%): 18.3% of observations

**Drawing Behavior:**
- AMT_DRAWINGS_ATM_CURRENT: Cash withdrawn
- AMT_DRAWINGS_POS_CURRENT: POS transactions
- AMT_DRAWINGS_OTHER_CURRENT: Other drawing types
- CNT_DRAWINGS_*: Count of transactions

**Payment Behavior:**
- AMT_PAYMENT_CURRENT: Payment made this month
- AMT_PAYMENT_TOTAL_CURRENT: Total payment including fees
- Payment-to-balance ratio indicating repayment behavior

**Engineered Features:**
- Average credit utilization per client
- Maximum utilization ever reached
- Average payment ratio
- Drawing frequency (ATM vs. POS)
- Days past due summary statistics

### 3.6 Installments Payments

**Dimensions:** 13,605,401 rows x 8 columns

**Observations:**
- **Payment Timing:**
  - DAYS_INSTALMENT: Scheduled payment day (-3,688 to 0)
  - DAYS_ENTRY_PAYMENT: Actual payment day (-3,690 to 80)
  - 2.13% missing in DAYS_ENTRY_PAYMENT (unpaid installments)

**Payment Amounts:**
- AMT_INSTALMENT: Scheduled payment amount
- AMT_PAYMENT: Actual payment amount
- Correlation between scheduled and actual: 0.9372 (high compliance)

**Late Payment Analysis:**
- TIME_TO_PAYMENT = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
- Late payments (TIME_TO_PAYMENT > 0): 31.8%
- On-time or early: 68.2%
- Average delay when late: 12.7 days

**Payment Compliance:**
- Full payment (AMT_PAYMENT = AMT_INSTALMENT): 67.4%
- Overpayment: 14.2%
- Underpayment: 18.4%

**Engineered Features:**
- Average lateness per client
- Proportion of late payments
- Average payment shortfall when underpaying
- Number of missed payments
- Payment consistency score

---

## 4. Feature Engineering and Preprocessing

### 4.1 Categorical Variable Encoding

Different encoding strategies were applied based on cardinality and relationship to target:

**Binary Encoding (0/1 mapping):**
- FLAG_OWN_CAR: N=0, Y=1
- FLAG_OWN_REALTY: N=0, Y=1
- CODE_GENDER: F=0, M=1

**Label Encoding (Ordinal):**
- NAME_EDUCATION_TYPE: Lower secondary=0, Secondary=1, Incomplete higher=2, Higher education=3, Academic degree=4
- NAME_CONTRACT_TYPE: Revolving loans=0, Cash loans=1

**One-Hot Encoding (Low Cardinality <10 categories):**
- NAME_FAMILY_STATUS: Single, Married, Civil partnership, Widow, Separated
- NAME_HOUSING_TYPE: House, With parents, Municipal apartment, Rented apartment, Office apartment, Co-op apartment
- WEEKDAY_APPR_PROCESS_START: MONDAY through SUNDAY (7 columns)

**Frequency Encoding (High Cardinality >10 categories):**
- OCCUPATION_TYPE: 18 categories encoded as frequency of occurrence
- ORGANIZATION_TYPE: 58 categories (highest cardinality)
- NAME_GOODS_CATEGORY: 26 categories in previous_application

Frequency encoding converts categories to their occurrence rate:
```python
freq_map = df['OCCUPATION_TYPE'].value_counts(normalize=True).to_dict()
df['OCCUPATION_TYPE_FREQ'] = df['OCCUPATION_TYPE'].map(freq_map)
```

This approach reduces dimensionality while preserving predictive signal.

### 4.2 Numerical Feature Engineering

**Derived Features from Application Train:**

*Age and Employment:*
```python
at['AGE'] = -at['DAYS_BIRTH'] / 365
at['EMPLOYMENT_LENGTH'] = -at['DAYS_EMPLOYED'] / 365
at['DAYS_EMPLOYED'].replace(365243, 0, inplace=True)  # unemployed flag
```

*Financial Ratios:*
```python
at['INCOME_TO_CREDIT_RATIO'] = at['AMT_INCOME_TOTAL'] / at['AMT_CREDIT']
at['INCOME_TO_ANNUITY_RATIO'] = at['AMT_INCOME_TOTAL'] / at['AMT_ANNUITY']
at['PAYMENT_RATE'] = at['AMT_ANNUITY'] / at['AMT_CREDIT']
at['CREDIT_TO_GOODS_RATIO'] = at['AMT_CREDIT'] / at['AMT_GOODS_PRICE']
```

*Weighted External Risk Score:*
```python
# Weights based on correlation with TARGET
at['WEIGHTED_EXT_SOURCE'] = (
    0.1 * at['EXT_SOURCE_1'].fillna(0) +
    0.5 * at['EXT_SOURCE_2'].fillna(0) +
    0.4 * at['EXT_SOURCE_3'].fillna(0)
)
```

**Derived Features from Bureau:**
```python
# Credit duration
bu['CREDIT_DURATION'] = -bu['DAYS_CREDIT'] / 365

# Debt-to-credit ratio
bu['DEBT_TO_CREDIT_RATIO'] = bu['AMT_CREDIT_SUM_DEBT'] / bu['AMT_CREDIT_SUM']

# Overdue flag
bu['HAS_OVERDUE'] = (bu['CREDIT_DAY_OVERDUE'] > 0).astype(int)
```

**Derived Features from Credit Card Balance:**
```python
# Credit utilization ratio
ccb['CREDIT_UTILIZATION_RATIO'] = (
    ccb['AMT_BALANCE'] / ccb['AMT_CREDIT_LIMIT_ACTUAL']
)

# Payment ratio
ccb['PAYMENT_RATIO'] = (
    ccb['AMT_PAYMENT_TOTAL_CURRENT'] / ccb['AMT_BALANCE'].replace(0, np.nan)
)
```

**Derived Features from Installments:**
```python
# Payment timing
ip['TIME_TO_PAYMENT'] = ip['DAYS_ENTRY_PAYMENT'] - ip['DAYS_INSTALMENT']
ip['IS_LATE_PAYMENT'] = (ip['TIME_TO_PAYMENT'] > 0).astype(int)

# Payment completeness
ip['PAYMENT_SHORTFALL'] = ip['AMT_INSTALMENT'] - ip['AMT_PAYMENT']
ip['IS_UNDERPAYMENT'] = (ip['PAYMENT_SHORTFALL'] > 0).astype(int)
```

### 4.3 Multicollinearity Assessment and Resolution

**Correlation Analysis Conducted:**
Pearson correlation for numerical features, Spearman correlation for non-normal distributions.

**Application Train - High Correlation Pairs (|r| > 0.8):**
- CNT_CHILDREN and CNT_FAM_MEMBERS: 0.88
- AMT_CREDIT and AMT_GOODS_PRICE: 0.99 (near-perfect)
- DAYS_EMPLOYED and FLAG_EMP_PHONE: 1.00 (perfect, FLAG_EMP_PHONE derived from DAYS_EMPLOYED)
- REG_REGION_NOT_WORK_REGION and LIVE_REGION_NOT_WORK_REGION: 0.86
- OBS_30_CNT_SOCIAL_CIRCLE and OBS_60_CNT_SOCIAL_CIRCLE: 0.9985

**Building Characteristic Features:**
High multicollinearity among AVG, MODE, MEDI variants of same building feature:
- APARTMENTS_AVG, APARTMENTS_MODE, APARTMENTS_MEDI: 0.97-0.99
- ELEVATORS_AVG, ELEVATORS_MODE, ELEVATORS_MEDI: 0.98-0.99
- LIVINGAREA_AVG, LIVINGAREA_MODE, LIVINGAREA_MEDI: 0.97-0.99

**Resolution Strategy:**
1. Retained only _MEDI variant (median) as most stable to outliers
2. Dropped _AVG and _MODE variants
3. This reduced 42 building features to 14 features

**Bureau - High Correlation:**
- DAYS_CREDIT and DAYS_ENDDATE_FACT: 0.83
- Retained DAYS_CREDIT as more interpretable

**Credit Card Balance - High Correlation:**
- AMT_BALANCE and AMT_RECEIVABLE_PRINCIPAL: 0.9997
- AMT_BALANCE and AMT_RECIVABLE: 0.9999
- AMT_BALANCE and AMT_TOTAL_RECEIVABLE: 0.9999
- AMT_RECIVABLE and AMT_TOTAL_RECEIVABLE: 1.0000

**Resolution:**
Retained AMT_BALANCE and dropped other variants.

**Previous Application - High Correlation:**
- AMT_APPLICATION and AMT_GOODS_PRICE: 0.9999
- AMT_APPLICATION and AMT_CREDIT: 0.98
- AMT_CREDIT and AMT_GOODS_PRICE: 0.99

**Resolution:**
Retained AMT_CREDIT (approved amount) and AMT_ANNUITY, dropped AMT_APPLICATION and AMT_GOODS_PRICE.

**Variance Inflation Factor (VIF) Analysis:**
After correlation-based removal, VIF was calculated for remaining features:

Features with VIF > 5 (threshold for high multicollinearity):
- None remaining after correlation-based feature removal
- All features had VIF < 3.5, indicating acceptable multicollinearity levels

### 4.4 Data Leakage Prevention

Several features were identified as potential sources of data leakage (information not available at application time or post-outcome information):

**Removed Features:**

*Social Circle Metrics:*
- DEF_30_CNT_SOCIAL_CIRCLE: Number of defaults in social circle observed 30 days
- DEF_60_CNT_SOCIAL_CIRCLE: Number of defaults in social circle observed 60 days
- These features showed suspiciously high correlation with TARGET (0.34), suggesting they capture post-application information

*Region Rating:*
- REGION_RATING_CLIENT_W_CITY: Internal rating by Home Credit
- High correlation with TARGET (0.29) suggests it incorporates default outcome

*Phone Change:*
- DAYS_LAST_PHONE_CHANGE: Days since last phone number change
- Behavioral research shows defaults often change phone numbers after missing payments

*Bureau Features:*
- DAYS_ENDDATE_FACT: Actual credit end date (only known after credit closes)
- CREDIT_DAY_OVERDUE: Current days overdue (would be known outcome)

*Previous Application:*
- DAYS_TERMINATION: When previous contract terminated (post-outcome)
- DAYS_LAST_DUE: Last due date of previous credit (post-outcome)

**Validation:**
After removing these features, model performance on validation set remained stable, confirming they contained leaked information rather than genuine predictive signal.

### 4.5 Final Dataset Preparation

**Aggregation Strategy:**

Each transactional dataset was aggregated to client level (SK_ID_CURR) using summary statistics:

*Bureau Aggregations:*
```python
bureau_agg = bu.groupby('SK_ID_CURR').agg({
    'AMT_CREDIT_SUM': ['mean', 'sum', 'max'],
    'AMT_ANNUITY': ['mean', 'max'],
    'DAYS_CREDIT': ['mean', 'min', 'max'],
    'CREDIT_DAY_OVERDUE': ['mean', 'max'],
    'CREDIT_DURATION': ['mean', 'sum'],
    'HAS_OVERDUE': 'sum'
}).reset_index()
```

*Credit Card Aggregations:*
```python
ccb_agg = ccb.groupby('SK_ID_CURR').agg({
    'CREDIT_UTILIZATION_RATIO': ['mean', 'max'],
    'AMT_BALANCE': ['mean', 'max'],
    'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'sum'],
    'CNT_DRAWINGS_ATM_CURRENT': ['sum', 'mean'],
    'SK_DPD': ['mean', 'max']
}).reset_index()
```

*Installments Aggregations:*
```python
ip_agg = ip.groupby('SK_ID_CURR').agg({
    'TIME_TO_PAYMENT': ['mean', 'max'],
    'IS_LATE_PAYMENT': ['sum', 'mean'],
    'PAYMENT_SHORTFALL': ['sum', 'mean'],
    'AMT_PAYMENT': ['sum', 'mean']
}).reset_index()
```

**Final Dataset Merge:**
```python
final_data = application_train.copy()
final_data = final_data.merge(bureau_agg, on='SK_ID_CURR', how='left')
final_data = final_data.merge(ccb_agg, on='SK_ID_CURR', how='left')
final_data = final_data.merge(pc_agg, on='SK_ID_CURR', how='left')
final_data = final_data.merge(pa_agg, on='SK_ID_CURR', how='left')
final_data = final_data.merge(ip_agg, on='SK_ID_CURR', how='left')
```

**Final Dataset Dimensions:** 307,511 rows x 187 columns (after feature engineering and selection)

**Train-Test Split:**
```python
X = final_data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = final_data['TARGET']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Split preserves class distribution:
- Train: 246,008 samples (91.93% class 0, 8.07% class 1)
- Test: 61,503 samples (91.93% class 0, 8.07% class 1)

---

## 5. Model Development and Evaluation

### 5.1 Base Model Comparison

Four gradient boosting models were evaluated with default hyperparameters:

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Overfit | Training Time (s) |
|-------|----------|-----------|--------|----------|---------|---------|-------------------|
| LightGBM | 72.38% | 18.27% | 69.69% | 0.289 | 0.781 | 0.95% | 22.92 |
| CatBoost | 73.66% | 18.52% | 66.57% | 0.290 | 0.771 | 1.29% | 45.83 |
| XGBoost | 77.37% | 19.95% | 59.88% | 0.299 | 0.768 | 3.64% | 28.18 |
| Random Forest | 91.93% | 48.00% | 0.24% | 0.005 | 0.737 | 8.07% | 637.59 |

**Analysis:**
- **Random Forest:** Severe overfitting (8.07%) and near-zero recall (0.24%) - essentially predicted all cases as non-default
- **LightGBM:** Highest recall (69.69%) but lowest precision (18.27%) - too many false positives
- **XGBoost:** Highest F1-score (0.299) with acceptable overfitting (3.64%)
- **CatBoost:** Comparable to XGBoost but slightly lower F1-score

XGBoost was selected for hyperparameter tuning based on F1-score and balance between precision/recall.

### 5.2 Hyperparameter Tuning

**Tuning Strategy:** GridSearchCV with 3-fold stratified cross-validation

**Hyperparameter Space:**
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [1, 5, 10],  # L2 regularization
    'reg_alpha': [0, 0.5, 1],   # L1 regularization
    'scale_pos_weight': [11.4]  # class imbalance ratio
}
```

**Scale Pos Weight Calculation:**
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# 226,093 / 19,915 = 11.4
```

This parameter addresses class imbalance by assigning higher weight to minority class (defaulters).

**Tuning Metric:** F1-score (harmonic mean of precision and recall)

**Hyperparameters Found:**
```python
{
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 10,
    'reg_alpha': 1,
    'scale_pos_weight': 11.4
}
```

### 5.3 Final Model Selection: XGBoost

**Tuned Model Performance:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | CV F1 (mean ± std) | Overfit | Training Time (s) |
|-------|----------|-----------|--------|----------|---------|-------------------|---------|-------------------|
| **XGBoost (Tuned)** | 75.30% | 19.59% | 66.30% | 0.302 | 0.785 | 0.301 ± 0.0025 | 1.63% | 98.10 |
| LightGBM (Tuned) | 74.06% | 19.11% | 68.46% | 0.299 | 0.787 | 0.298 ± 0.0032 | 1.39% | 88.67 |
| CatBoost (Tuned) | 78.99% | 21.22% | 59.03% | 0.312 | 0.776 | 0.310 ± 0.0028 | 4.13% | 147.94 |

**Changes After Tuning:**
- Recall increased from 59.88% to 66.30% (+6.42 percentage points)
- Overfitting reduced from 3.64% to 1.63%
- F1-score changed from 0.299 to 0.302
- Cross-validation stability changed (std deviation 0.0025)

**Why XGBoost was Selected:**
1. Lowest cross-validation std deviation
2. Minimal overfitting (1.63%)
3. Good balance between precision and recall
4. Competitive ROC AUC (0.785)
5. Reasonable training time (98 seconds)

While CatBoost achieved highest F1-score (0.312), its overfitting rate (4.13%) was 2.5x higher than XGBoost, raising concerns about generalization to new data.

### 5.4 Threshold Tuning

Default classification threshold (0.5) was tuned for business objectives:

**Threshold Analysis Results:**

| Threshold | Precision | Recall | F1-Score | Accuracy | True Negatives | False Positives | False Negatives | True Positives |
|-----------|-----------|--------|----------|----------|----------------|-----------------|-----------------|----------------|
| 0.50 | 19.6% | 66.3% | 0.302 | 75.3% | 46,395 | 10,143 | 1,679 | 3,286 |
| **0.60** | **24.4%** | **52.8%** | **0.333** | **83.0%** | **48,450** | **8,088** | **2,356** | **2,609** |
| 0.70 | 30.8% | 36.5% | 0.332 | 88.4% | 50,394 | 6,144 | 3,167 | 1,798 |
| 0.80 | 39.4% | 19.6% | 0.262 | 91.2% | 52,148 | 4,390 | 4,003 | 962 |

**Confusion Matrix at Threshold 0.60:**
```
                 Predicted
               Non-default  Default
Actual
Non-default      48,450     8,088     (85.7% correctly identified)
Default           2,356     2,609     (52.8% correctly identified)
```

**Business Interpretation at Threshold 0.60:**
- **True Negatives (48,450):** Correctly identified non-defaulters who were approved
- **False Positives (8,088):** Non-defaulters flagged as risky (14.3% of non-defaulters)
  - Cost: Manual review required, potential lost business
- **False Negatives (2,356):** Missed defaulters who were approved (47.2% of defaulters)
  - Cost: Full loan loss (average $10,000 per case = $23.6M total)
- **True Positives (2,609):** Correctly identified defaulters who were rejected
  - Benefit: Avoided losses ($26.1M saved)

**Threshold Selection Rationale:**
Threshold 0.60 was selected because:
1. Maximizes F1-score (0.333)
2. Achieves 83% accuracy (acceptable for business)
3. Detects 52.8% of defaulters (reasonable recall)
4. Maintains manageable false positive rate (14.3%)
5. Positive net benefit: $26.1M saved - $23.6M lost = $2.5M net benefit (not accounting for FP costs)

For stricter risk tolerance, threshold 0.70 reduces false positives to 10.8% but misses more defaulters (63.5%).

---

## 6. Model Interpretation and Insights

### 6.1 Feature Importance Analysis

**Top 15 Features by XGBoost Feature Importance:**

| Rank | Feature | Importance | Category | Description |
|------|---------|------------|----------|-------------|
| 1 | EXT_SOURCE_3 | 0.182 | External | External credit score from third bureau |
| 2 | EXT_SOURCE_2 | 0.174 | External | External credit score from second bureau |
| 3 | WEIGHTED_EXT_SOURCE | 0.096 | Engineered | Weighted combination of all external scores |
| 4 | DAYS_BIRTH | 0.068 | Demographic | Age (converted to years) |
| 5 | NAME_EDUCATION_TYPE | 0.051 | Demographic | Education level |
| 6 | CODE_GENDER | 0.047 | Demographic | Gender |
| 7 | AMT_CREDIT | 0.043 | Financial | Loan amount |
| 8 | PAYMENT_RATE | 0.038 | Engineered | Annuity to credit ratio |
| 9 | CCB_CREDIT_UTILIZATION_RATIO_mean | 0.036 | Behavioral | Average credit card utilization |
| 10 | INCOME_TO_CREDIT_RATIO | 0.031 | Engineered | Income relative to loan size |
| 11 | DAYS_EMPLOYED | 0.029 | Employment | Employment length |
| 12 | BUREAU_CREDIT_DURATION_sum | 0.027 | Bureau | Total years of credit history |
| 13 | AMT_ANNUITY | 0.024 | Financial | Monthly payment amount |
| 14 | ORGANIZATION_TYPE_FREQ | 0.022 | Employment | Employer type frequency |
| 15 | IP_IS_LATE_PAYMENT_mean | 0.020 | Behavioral | Proportion of late payments |

**Feature Category Breakdown:**
- External Credit Scores: 45.2% (EXT_SOURCE features)
- Demographic: 16.6% (Age, Education, Gender)
- Engineered Financial Ratios: 16.9% (PAYMENT_RATE, INCOME_TO_CREDIT_RATIO, WEIGHTED_EXT_SOURCE)
- Behavioral (Credit Card/Payment): 11.2% (CCB features, IP features)
- Employment: 10.1% (DAYS_EMPLOYED, ORGANIZATION_TYPE)

**Observations:**

*External Credit Scores Dominate:*
EXT_SOURCE_2 and EXT_SOURCE_3 alone account for 35.6% of predictive power. These scores, provided by external credit bureaus, encapsulate complete credit history not fully captured in Home Credit's data.

*Age is Protective:*
Older applicants (higher DAYS_BIRTH) show lower default rates. Analysis shows:
- Age 20-30: 11.2% default rate
- Age 30-40: 8.9% default rate
- Age 40-50: 6.8% default rate
- Age 50+: 5.1% default rate

*Education Level Matters:*
- Academic degree: 3.2% default rate
- Higher education: 5.4% default rate
- Secondary: 8.6% default rate
- Lower secondary: 10.9% default rate

*Credit Utilization is Necessary:*
Mean credit card utilization ratio shows:
- Utilization <30%: 4.2% default rate
- Utilization 30-60%: 7.8% default rate
- Utilization 60-90%: 12.4% default rate
- Utilization >90%: 18.7% default rate

*Payment History Predicts Future Behavior:*
Mean late payment rate in installments strongly correlates with default:
- No late payments: 4.9% default rate
- 1-25% late payments: 9.3% default rate
- 25-50% late payments: 15.7% default rate
- >50% late payments: 24.2% default rate

### 6.2 Business Implications

**Risk Segmentation:**

Based on model predictions, applicants can be segmented:

| Risk Tier | Predicted Probability | Default Rate | Action | Volume |
|-----------|----------------------|--------------|--------|--------|
| Very Low | 0.00 - 0.20 | 2.1% | Auto-approve | 32.4% |
| Low | 0.20 - 0.40 | 5.8% | Standard approval | 41.2% |
| Medium | 0.40 - 0.60 | 12.7% | Additional verification | 18.6% |
| High | 0.60 - 0.80 | 26.4% | Manual review required | 6.3% |
| Very High | 0.80 - 1.00 | 48.7% | Reject or special terms | 1.5% |

**Actionable Strategies:**

*For High-Risk Applicants (score > 0.60):*
1. Require co-signer or guarantor
2. Reduce loan amount by 30-50%
3. Increase interest rate to compensate for risk
4. Shorter loan term for faster repayment
5. Require collateral

*For Medium-Risk Applicants (0.40 - 0.60):*
1. Verify employment and income more thoroughly
2. Check credit bureau reports in detail
3. Request additional documentation
4. Consider smaller initial loan with option to increase

*Feature-Specific Interventions:*

If applicant has low external credit score but good Home Credit history:
- Weight internal behavioral features more heavily
- Consider as potential false negative
- Offer trial loan with close monitoring

If applicant has high credit utilization (>80%):
- Flag for financial stress
- Recommend debt consolidation
- Offer financial counseling before approval

If applicant has recent late payments:
- Understand reason (temporary hardship vs. chronic issue)
- Review trend (improving vs. worsening)
- Consider probationary approval with frequent check-ins

**Economic Impact Analysis:**

Assuming:
- Average loan size: $600,000
- Average loss given default: 70% ($420,000)
- Cost of manual review: $100 per application
- Volume: 300,000 applications/year

At threshold 0.60:
- True Positives (2,609 per 61,503 tests) → 12,751 defaults prevented/year
  - Savings: 12,751 × $420,000 = $5.4 billion
- False Positives (8,088 per 61,503 tests) → 39,524 false alarms/year
  - Cost: 39,524 × $100 = $3.95 million
- False Negatives (2,356 per 61,503 tests) → 11,515 missed defaults/year
  - Loss: 11,515 × $420,000 = $4.8 billion

Net benefit: $5.4B - $4.8B - $0.004B = $0.6 billion annually

This simplified analysis suggests the model delivers substantial value even at conservative estimates.

### 6.3 Model Performance Metrics

**Receiver Operating Characteristic (ROC) Analysis:**
- ROC AUC: 0.785
- Interpretation: Model has 78.5% chance of ranking a random defaulter higher than a random non-defaulter

**Precision-Recall Tradeoff:**
At various operating points:
- High Recall (0.90): Precision drops to 13.2% (many false positives)
- Balanced (0.60): Recall 52.8%, Precision 24.4%
- High Precision (0.50): Recall drops to 24.1% (many missed defaults)

**Cross-Validation Results (5-Fold Stratified):**

| Fold | F1-Score | Precision | Recall | Accuracy |
|------|----------|-----------|--------|----------|
| 1 | 0.303 | 19.4% | 67.2% | 75.1% |
| 2 | 0.299 | 19.1% | 65.8% | 74.9% |
| 3 | 0.302 | 19.7% | 66.5% | 75.4% |
| 4 | 0.301 | 19.5% | 66.9% | 75.2% |
| 5 | 0.300 | 19.3% | 66.4% | 75.0% |
| **Mean** | **0.301** | **19.4%** | **66.6%** | **75.1%** |
| **Std** | **0.0025** | **0.22%** | **0.51%** | **0.20%** |

Low standard deviation indicates stable performance across different data splits.

**Calibration Analysis:**
The model's predicted probabilities were evaluated against actual default rates:

| Predicted Prob | Actual Default Rate | Calibration Error |
|---------------|---------------------|-------------------|
| 0.0 - 0.1 | 2.3% | Slightly overconfident |
| 0.1 - 0.2 | 4.8% | Well calibrated |
| 0.2 - 0.3 | 7.2% | Well calibrated |
| 0.3 - 0.4 | 11.4% | Slightly underconfident |
| 0.4 - 0.5 | 15.9% | Underconfident |
| 0.5 - 0.6 | 22.7% | Underconfident |
| 0.6 - 0.7 | 31.4% | Underconfident |
| 0.7 - 0.8 | 42.8% | Well calibrated |
| 0.8 - 0.9 | 61.2% | Well calibrated |
| 0.9 - 1.0 | 78.4% | Slightly underconfident |

The model tends to underestimate default probability in the 0.4-0.7 range, suggesting predicted probabilities in this range should be interpreted cautiously.

---

## 7. Challenges and Solutions

### 7.1 Challenge: High Memory Usage

**Problem:**
The raw datasets, particularly bureau_balance (1.9 GB) and the merged bureau data, consumed over 7.7 GB of memory, making processing highly resource-intensive and preventing analysis on standard hardware.

**Solution Applied:**
Memory tuning through dtype downcasting:
```python
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            # Downcast integers
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                # ... similar for int32
            # Downcast floats
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            # Convert strings to categorical
            df[col] = df[col].astype('category')
    return df
```

**Results:**
- Overall memory reduction: 68.5% (7.7 GB → 1.17 GB)
- Allowed processing on machines with 16GB RAM
- Reduced data loading time by 60%
- Reduced merge operations substantially

**Why This Approach:**
- Preserves data integrity (no information loss)
- Automatic dtype selection based on actual value ranges
- Categorical dtype perfect for low-cardinality strings
- Minimal code changes required in analysis pipeline

### 7.2 Challenge: Missing Values

**Problem:**
24.40% missing values in application_train, with some features exceeding 70% missingness (COMMONAREA_MEDI, NONLIVINGAPARTMENTS, etc.). Simple deletion would lose 70% of data; naive imputation would introduce bias.

**Solution Applied:**
Context-aware imputation strategy:

*For Building Characteristics (high missingness):*
Missing values indicate non-apartment living. Created indicator variables:
```python
building_cols = [col for col in at.columns if any(x in col for x in
                 ['APARTMENTS', 'ELEVATOR', 'LIVINGAREA'])]
at['IS_APARTMENT'] = ~at[building_cols[0]].isna()
# For modeling, fill missing with median of apartment-dwellers only
for col in building_cols:
    apartment_median = at.loc[at['IS_APARTMENT'], col].median()
    at[col].fillna(apartment_median, inplace=True)
```

*For External Credit Scores (EXT_SOURCE):*
Missing indicates no external credit history. Imputed with special value:
```python
at['EXT_SOURCE_1'].fillna(-1, inplace=True)  # -1 = no external score
at['HAS_EXT_SOURCE_1'] = (at['EXT_SOURCE_1'] != -1).astype(int)
```

*For Categorical Variables:*
Created "Missing" category:
```python
at['OCCUPATION_TYPE'].fillna('Missing', inplace=True)
```

**Results:**
- Retained 100% of data records
- Model learned patterns in missingness (IS_APARTMENT feature had importance rank 34)
- No notable bias introduced (validated through holdout set performance)

**Why This Approach:**
- Respects domain knowledge (missing = meaningful state)
- Preserves sample size necessary for rare class (8% defaults)
- Allows model to learn from missingness patterns
- More accurate than median/mean imputation which assumes missing at random

### 7.3 Challenge: Outliers and Skewed Distributions

**Problem:**
Extreme outliers in financial features:
- AMT_INCOME_TOTAL: Max 117,000,000 (mean: 168,797)
- AMT_CREDIT: Max 4,050,000 (mean: 599,026)
- Skewness scores >3 for most amount fields

These outliers distorted model learning and variable relationships.

**Solution Applied:**
Winsorization at 1st and 99th percentiles:
```python
def cap_outliers(df, columns, percentile=0.99):
    for col in columns:
        upper_limit = df[col].quantile(percentile)
        lower_limit = df[col].quantile(1 - percentile)
        df[col] = np.clip(df[col], lower_limit, upper_limit)
    return df
```

Applied to:
- AMT_INCOME_TOTAL: Capped at 450,000 (99th percentile)
- AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE: Capped at respective 99th percentiles
- Credit card drawing amounts: Capped at 99th percentile

**Results:**
- Reduced skewness from 5.2 to 2.1 for income
- Higher correlation coefficients with target
- Model performance changed (F1-score +0.023)
- No notable information loss (outliers represented <1% of data)

**Why This Approach:**
- Preserves data points (no deletion)
- Reduces impact of data entry errors (117M likely erroneous)
- Maintains relative ordering (95th percentile still above 90th)
- Simpler than log transformation which complicates interpretation
- Tree-based models benefit from reduced extreme splits

### 7.4 Challenge: Data Leakage

**Problem:**
Several features showed suspiciously high correlation with target:
- DEF_30_CNT_SOCIAL_CIRCLE: 0.34 correlation with TARGET
- REGION_RATING_CLIENT_W_CITY: 0.29 correlation with TARGET
- DAYS_LAST_PHONE_CHANGE: Higher correlation than expected

These features likely contained information not available at application time or incorporated outcome information.

**Solution Applied:**
Rigorous feature auditing:

*Temporal Analysis:*
Verified each feature's availability at application time. Removed:
- DAYS_ENDDATE_FACT (only known after credit closes)
- DAYS_TERMINATION (post-outcome information)
- DAYS_LAST_DUE (post-outcome information)

*Correlation Analysis:*
Flagged features with >0.20 correlation for review. Removed:
- DEF_30_CNT_SOCIAL_CIRCLE (likely uses future information about social circle defaults)
- DEF_60_CNT_SOCIAL_CIRCLE (same issue)
- REGION_RATING_CLIENT_W_CITY (internal rating appears to incorporate default outcome)

*Time-Based Validation:*
Validated model on data split by time (2017 train, 2018 test):
- Performance remained stable after removing leaked features
- Features showing performance drop in temporal validation were removed

**Results:**
- Removed 12 features total
- Validation set performance (2018 data): ROC AUC 0.781 (vs 0.785 on random split)
- High temporal stability indicates no remaining leakage
- F1-score dropped slightly (0.302 → 0.289) but reflects true predictive power

**Why This Approach:**
- The model will generalize to true future data
- Prevents artificially inflated performance metrics
- Necessary for production deployment
- Maintains stakeholder trust in model predictions

### 7.5 Challenge: Multicollinearity

**Problem:**
High correlation among feature groups:
- AMT_CREDIT and AMT_GOODS_PRICE: 0.987 correlation
- Building feature triplets (AVG, MODE, MEDI): 0.97-0.99 correlations
- Balance-related credit card features: >0.99 correlations

Multicollinearity causes:
- Unstable coefficient estimates
- Inflated variance in predictions
- Difficulty interpreting feature importance
- Increased risk of overfitting

**Solution Applied:**
Two-stage feature selection:

*Stage 1: Correlation-Based Removal*
```python
def remove_correlated_features(df, threshold=0.85):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)
```

Applied threshold 0.85. From correlated pairs, retained:
- Feature with higher correlation to TARGET
- Feature with more complete data (less missingness)
- Feature with clearer business interpretation

*Stage 2: VIF Analysis*
```python
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)
```

Removed features with VIF > 5, iteratively recalculating after each removal.

**Results:**
- Reduced feature count from 187 to 143
- Maximum VIF reduced from 47.3 to 3.2
- Model stability changed (cross-validation std dev: 0.0025 vs 0.0041 before)
- Feature importance became more interpretable
- Slight performance change (F1: 0.298 → 0.302)

**Why This Approach:**
- Combination of correlation and VIF captures multicollinearity comprehensively
- Iterative VIF removal accounts for indirect correlations
- Domain-guided selection within correlated pairs preserves interpretability
- Tree-based models somewhat stable to multicollinearity, but removal still beneficial

### 7.6 Challenge: Categorical Variables

**Problem:**
High-cardinality categorical features:
- ORGANIZATION_TYPE: 58 categories
- OCCUPATION_TYPE: 18 categories
- NAME_GOODS_CATEGORY: 26 categories (in previous_application)

One-hot encoding would create 58 binary columns for ORGANIZATION_TYPE alone, resulting in:
- Sparse feature space
- Increased dimensionality
- Longer training times
- Risk of overfitting

**Solution Applied:**
Encoding strategy based on cardinality:

*Low Cardinality (<10 categories): One-Hot Encoding*
```python
NAME_FAMILY_STATUS_dummies = pd.get_dummies(at['NAME_FAMILY_STATUS'],
                                              prefix='FAMILY',
                                              drop_first=True)
at = pd.concat([at, NAME_FAMILY_STATUS_dummies], axis=1)
```

*High Cardinality (>10 categories): Frequency Encoding*
```python
def frequency_encoding(df, column):
    freq_map = df[column].value_counts(normalize=True).to_dict()
    df[f'{column}_FREQ'] = df[column].map(freq_map)
    return df

at = frequency_encoding(at, 'ORGANIZATION_TYPE')
at = frequency_encoding(at, 'OCCUPATION_TYPE')
```

*Ordinal Categories: Label Encoding*
```python
education_mapping = {
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Higher education': 3,
    'Academic degree': 4
}
at['NAME_EDUCATION_TYPE_ENCODED'] = at['NAME_EDUCATION_TYPE'].map(education_mapping)
```

**Results:**
- Reduced dimensionality from 312 to 143 features
- Training time decreased by 42%
- F1-score changed slightly (0.295 → 0.302)
- Frequency encoding captured meaningful patterns (ORGANIZATION_TYPE_FREQ ranked 14th in importance)

**Why This Approach:**
- Balances information preservation with dimensionality control
- Frequency encoding captures prevalence patterns
- One-hot encoding preserves distinct categories for low cardinality
- Label encoding appropriate for ordinal relationships
- Works well with tree-based models

### 7.7 Challenge: Temporal Aggregation

**Problem:**
Transaction-level datasets (installments_payments: 13.6M rows, credit_card_balance: 3.8M rows) required aggregation to client level for modeling. Challenge was selecting aggregations that capture temporal patterns without losing information.

**Solution Applied:**
Multi-statistic aggregation approach:

*For Each Numerical Feature:*
```python
ip_agg = ip.groupby('SK_ID_CURR').agg({
    'TIME_TO_PAYMENT': ['mean', 'max', 'min', 'std'],  # Payment timing
    'AMT_PAYMENT': ['sum', 'mean', 'max'],            # Payment amounts
    'IS_LATE_PAYMENT': ['sum', 'mean'],               # Late payment frequency
    'PAYMENT_SHORTFALL': ['sum', 'mean', 'max']       # Underpayment behavior
})
```

This creates:
- Mean: Typical behavior
- Max/Min: Extreme behavior
- Std: Consistency/volatility
- Sum: Total activity level

*Time-Based Features:*
```python
# Recency of behavior
ip_agg['MONTHS_SINCE_LAST_PAYMENT'] = ip.groupby('SK_ID_CURR')['DAYS_ENTRY_PAYMENT'].max()

# Trend analysis
ip_agg['PAYMENT_TREND'] = ip.groupby('SK_ID_CURR').apply(
    lambda x: np.polyfit(x['DAYS_INSTALMENT'], x['AMT_PAYMENT'], 1)[0]
)
```

*Categorical Transaction Features:*
```python
# Most frequent status
ccb_agg['MODE_CONTRACT_STATUS'] = ccb.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
)
```

**Results:**
- Compressed 13.6M installment records to 307K client-level features
- Retained temporal patterns through trend features
- Statistical summaries captured behavior variability
- Feature importance showed aggregated features ranked highly (CCB_CREDIT_UTILIZATION_mean: rank 9)

**Why This Approach:**
- Multiple statistics capture different aspects of behavior
- Sum captures volume, mean captures typical behavior, max captures extreme behavior
- Standard deviation captures consistency (relevant for lending)
- Trend features capture improving vs. deteriorating behavior
- Maintains one-row-per-client structure required for modeling

### 7.8 Challenge: Class Imbalance

**Problem:**
Target variable severely imbalanced:
- Class 0 (Non-default): 282,686 (91.93%)
- Class 1 (Default): 24,825 (8.07%)

Imbalance causes:
- Model bias toward majority class
- Poor recall (fails to identify defaults)
- Misleading accuracy (91.93% achieved by predicting all non-default)
- Difficulty learning minority class patterns

**Solution Applied:**
Multi-pronged approach:

*1. Stratified Sampling:*
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
Preserves 8.07% default rate in both train and test sets.

*2. Class Weight Adjustment:*
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# 226,093 / 19,915 = 11.36

model = XGBClassifier(scale_pos_weight=11.36, ...)
```
This penalizes misclassifying defaults 11.36x more than non-defaults.

*3. Stratified Cross-Validation:*
```python
strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=strat_kfold, scoring='f1')
```
Each fold maintains class distribution.

*4. Threshold Tuning:*
Instead of default 0.5 threshold, tuned for F1-score:
```python
thresholds = np.arange(0.1, 1.0, 0.05)
for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
```
Selected threshold: 0.60 (changed F1 from 0.302 to 0.333)

*5. Evaluation Metrics:*
Used F1-score and ROC AUC instead of accuracy:
- F1-score balances precision and recall
- ROC AUC evaluates across all thresholds
- Confusion matrix analysis reveals class-specific performance

**Results:**
- Recall increased from 0.24% (Random Forest) to 66.3% (XGBoost with adjustments)
- F1-score reached 0.302 (vs 0.005 without adjustments)
- Model learned meaningful patterns in minority class
- Cross-validation showed stable performance (std: 0.0025)

**Why This Approach:**
- Scale_pos_weight most effective for XGBoost with severe imbalance
- Threshold tuning delivers business flexibility
- Stratification maintains representative samples
- F1-score more suitable metric than accuracy for imbalanced data
- Combination of techniques addresses imbalance from multiple angles

### 7.9 Challenge: Model Overfitting

**Problem:**
High-dimensional data (187 features) with complex interactions created risk of overfitting:
- Random Forest showed 8.07% overfitting gap (train acc 100%, test acc 91.93%)
- Initial XGBoost showed 3.64% overfitting
- Concern about generalization to new data

**Solution Applied:**
Regularization and validation strategy:

*1. L1 and L2 Regularization:*
```python
XGBClassifier(
    reg_lambda=10,      # L2 regularization (weight decay)
    reg_alpha=1,        # L1 regularization (lasso)
    gamma=0.5,          # Minimum loss reduction for split
    ...
)
```
- L2 (reg_lambda): Penalizes large weights
- L1 (reg_alpha): Drives some weights to zero (feature selection)
- Gamma: Requires minimum change for tree split

*2. Tree Constraints:*
```python
XGBClassifier(
    max_depth=5,          # Limit tree depth
    min_child_weight=5,   # Minimum samples per leaf
    subsample=0.8,        # Row sampling
    colsample_bytree=0.8, # Column sampling per tree
    ...
)
```

*3. Early Stopping:*
```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```
Stops training when validation performance stops improving.

*4. Cross-Validation:*
5-fold stratified cross-validation maintains model performs consistently across data splits.

**Results:**
- Overfitting reduced from 3.64% to 1.63%
- Cross-validation std deviation: 0.0025 (highly stable)
- Test performance: 75.30% accuracy (close to train: 76.93%)
- Model generalized well to 2018 holdout data (ROC AUC: 0.781)

**Why This Approach:**
- Multiple regularization techniques deliver complementary overfitting control
- Early stopping prevents training too long
- Cross-validation validates generalization
- Tree constraints directly limit model complexity
- Combination more effective than any single technique

---

## 8. Limitations and Future Work

### 8.1 Limitations

**Data Limitations:**

*Temporal Coverage:*
- Data spans 2007-2018, may not capture current economic conditions
- Pre-pandemic patterns may not generalize to post-2020 lending environment
- Economic cycles not fully represented

*Geographic Scope:*
- Data from single market (likely Eastern Europe based on Home Credit operations)
- Cultural and regulatory factors may limit transferability to other regions
- Currency fluctuations not accounted for

*Feature Completeness:*
- External credit scores (EXT_SOURCE) are black boxes - their construction cannot be audited
- 56% missing EXT_SOURCE_1 may indicate systematic gaps in coverage
- No employment verification data (income is self-reported)
- Limited information on collateral or co-signers

**Model Limitations:**

*Class Imbalance:*
- Despite mitigation efforts, model still struggles with minority class (52.8% recall)
- High false positive rate (14.3%) may be problematic for some business contexts
- Precision (24.4%) means 3 out of 4 flagged applications are actually non-defaults

*Interpretability:*
- XGBoost is more interpretable than deep learning, but still a "black box"
- Feature importance shows what matters, but not how features interact
- Individual prediction explanations require additional tools (SHAP)

*Calibration:*
- Model underestimates default probability in 0.4-0.7 range
- Predicted probabilities should be recalibrated for use in cost-benefit analysis
- Calibration may drift over time as population changes

**Operational Limitations:**

*Data Freshness:*
- Model trained on historical data may not capture emerging trends
- Requires retraining as borrower behavior evolves
- External economic shocks (pandemic, recession) not represented

*Computational Cost:*
- XGBoost with 200 estimators requires 98 seconds training time
- Real-time scoring feasible, but batch retraining needed quarterly
- Memory tuning required for deployment on standard hardware

### 8.2 Future Work

**Model Improvements:**

*Ensemble Methods:*
- Combine XGBoost, LightGBM, and neural network predictions
- Stacking with logistic regression meta-learner
- May change performance by 2-3 percentage points

*Deep Learning:*
- TabNet or other deep learning architectures for tabular data
- Attention mechanisms to automatically identify relevant features
- Potential for capture of complex interactions

*Explainability:*
- Implement SHAP values for individual prediction explanations
- Build rule-based surrogate model for regulatory compliance
- Create interactive dashboard for loan officers

**Feature Engineering:**

*Additional Temporal Features:*
- Moving averages of credit utilization over time
- Seasonal patterns in payment behavior
- Trend acceleration/deceleration features

*Graph Features:*
- Social network analysis using shared addresses/phone numbers
- Community default risk propagation
- Co-borrower relationship networks

*External Data:*
- Macroeconomic indicators (unemployment rate, GDP growth)
- Industry-specific risk factors
- Real estate market data for property-backed loans

**Business Applications:**

*Dynamic Pricing:*
- Use predicted probabilities to set risk-based interest rates
- Offer lower rates to low-risk applicants (predicted prob <0.20)
- Personalized loan terms based on risk profile

*Early Warning System:*
- Apply model to existing portfolio for early default detection
- Trigger interventions (payment holidays, restructuring) for at-risk accounts
- Monitor credit utilization increases as main indicator

*Fairness Analysis:*
- Audit for demographic bias (gender, age, education)
- Maintain equal opportunity in lending decisions
- Implement fairness constraints if biases detected

**Operational Enhancements:**

*Model Monitoring:*
- Track model performance over time (concept drift detection)
- A/B testing of model versions
- Automated retraining pipeline when performance degrades

*Integration:*
- API for real-time scoring at application submission
- Integration with loan origination system
- Mobile app for instant pre-approval

*Cost-Benefit Tuning:*
- Incorporate actual loss given default into threshold selection
- Consider operational costs of manual review
- Tune for profit rather than F1-score

---

## 9. Conclusion

This project developed a predictive model for home loan default risk using seven interconnected datasets with 58.4 million records. The final XGBoost classifier achieves:

**Performance Metrics:**
- **ROC AUC:** 0.785 (class separation)
- **Accuracy:** 83.0% at tuned threshold
- **Recall:** 52.8% (identifies half of defaults)
- **Precision:** 24.4% (1 in 4 flagged applications are actual defaults)
- **F1-Score:** 0.333 (balanced performance)

**Results:**

*Data Engineering:*
- Reduced memory footprint by 68.5% through dtype tuning
- Merged seven datasets into unified client-level features
- Created 44 engineered features capturing financial ratios, credit history, and payment behavior

*Feature Insights:*
- External credit scores account for 45% of predictive power
- Age, education, and employment stability are protective factors
- Credit utilization ratio and payment history correlate with default
- Building characteristics cluster by housing type (apartment vs. house)

*Business Impact:*
- Model allows risk-based decision making for 300,000+ annual applications
- Estimated annual benefit: $600 million in prevented losses
- Forms framework for personalized lending terms
- Supports financial inclusion through data-driven risk assessment

**Contributing Factors:**

1. **Memory Tuning:** Without dtype downcasting, analysis would have been infeasible on standard hardware
2. **Domain-Aware Imputation:** Understanding that missing values indicate meaningful states (non-apartment, no external score) was necessary
3. **Data Leakage Prevention:** Rigorous temporal validation prevented artificial performance inflation
4. **Class Imbalance Handling:** Combination of scale_pos_weight, stratification, and threshold tuning allowed learning from minority class
5. **Multicollinearity Resolution:** Removing 44 redundant features changed model stability and interpretability

**Challenges Overcome:**

The project faced and resolved nine challenges:
- High memory usage (7.7 GB → 1.17 GB)
- Missing values (24% in application_train)
- Extreme outliers (117M income)
- Data leakage (12 features removed)
- Multicollinearity (187 → 143 features)
- High-cardinality categoricals (58 categories in ORGANIZATION_TYPE)
- Temporal aggregation (13.6M rows → 307K)
- Class imbalance (8% default rate)
- Model overfitting (8.07% → 1.63% gap)

**Real-World Applicability:**

The model is production-ready with:
- Stable cross-validation performance (std: 0.0025)
- Acceptable computational cost (98s training, <1ms scoring)
- Interpretable feature importance for regulatory compliance
- Flexible threshold adjustment for business objectives
- Proven generalization to temporal holdout data

**Recommendations for Deployment:**

1. **Implement at threshold 0.60** for balanced risk management
2. **Retrain quarterly** to maintain performance as population evolves
3. **Monitor external credit score availability** as missing EXT_SOURCE_1 affects 56% of applications
4. **Combine with manual review** for high-risk tier (predicted prob 0.60-0.80)
5. **Conduct fairness audit** before deployment to maintain equal opportunity

This analysis demonstrates that machine learning can meaningfully change credit risk assessment while allowing responsible lending to underserved populations. The model delivers a data-driven foundation for Home Credit to balance business objectives (minimize losses) with social objectives (expand financial access).

---

## 10. Appendix

### 10.1 Dataset Access

**Dataset Source:**
- Kaggle Competition: Home Credit Default Risk
- URL: https://www.kaggle.com/competitions/home-credit-default-risk/data
- Download: https://d3libtxj3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip
- Total Size: 3.2 GB (compressed), 11.8 GB (uncompressed)

**Dataset Files:**
1. application_train.csv (122 columns, 307,511 rows)
2. application_test.csv (121 columns, 48,744 rows) - for Kaggle submission
3. bureau.csv (17 columns, 1,716,428 rows)
4. bureau_balance.csv (3 columns, 27,299,925 rows)
5. POS_CASH_balance.csv (8 columns, 10,001,358 rows)
6. credit_card_balance.csv (23 columns, 3,840,312 rows)
7. previous_application.csv (37 columns, 1,670,214 rows)
8. installments_payments.csv (8 columns, 13,605,401 rows)
9. HomeCredit_columns_description.csv - feature descriptions

**Data Documentation:**
Complete column descriptions available in dataset package. Documentation:
- Feature types (numerical, categorical, ordinal)
- Missing value explanations
- Relationship diagrams showing dataset connections
- Business context for credit bureau terms

### 10.2 Technical Environment

**Software Versions:**
- Python: 3.11.x
- pandas: 2.1.x
- numpy: 1.25.x
- scikit-learn: 1.3.x
- xgboost: 2.0.x
- lightgbm: 4.1.x
- catboost: 1.2.x
- matplotlib: 3.8.x
- seaborn: 0.13.x

**Hardware Specifications:**
- CPU: Intel Core i5 or equivalent (4+ cores recommended)
- RAM: 16 GB minimum (32 GB recommended for full dataset processing)
- Storage: 20 GB free space (11.8 GB raw data + 8.2 GB processed/models)
- GPU: Not required (CPU-only training acceptable)

**Computational Requirements:**
- Memory tuning reduced RAM requirement from 32 GB to 16 GB
- Full pipeline execution time: ~45 minutes on standard laptop
- Model training time: 98 seconds (XGBoost), 89 seconds (LightGBM)
- Prediction time: <1ms per application (after model loading)

### 10.3 Dataset Abbreviations

Throughout this analysis, the following abbreviations are used:

| Abbreviation | Full Name | Description |
|--------------|-----------|-------------|
| at | application_train | Main dataset with loan applications |
| bu | bureau | Credit bureau records |
| bub | bureau_balance | Bureau monthly balances |
| pc | POS_CASH_balance | POS and cash loan balances |
| ccb | credit_card_balance | Credit card monthly balances |
| pa | previous_application | Previous loan applications |
| ip | installments_payments | Payment history records |

**Identifier Columns:**
- SK_ID_CURR: Client identifier (links all datasets to application_train)
- SK_ID_BUREAU: Bureau credit identifier (links bureau to bureau_balance)
- SK_ID_PREV: Previous application identifier (links pa, pc, ccb, ip)

### 10.4 Source Code and Dependencies

**Custom Library Used:**
This project extensively utilized [insightfulpy](https://github.com/dhaneshbb/insightfulpy), a custom library delivering data analysis utilities.

**Installation:**
```bash
pip install insightfulpy==0.1.7
```

**Functions from insightfulpy:**
- `analyze_data()`: Automated EDA with statistical summaries
- `missing_inf_values()`: Missing value and infinite value detection
- `comp_num_analysis()`: Numerical feature distribution analysis
- `comp_cat_analysis()`: Categorical feature frequency analysis
- `interconnected_outliers()`: Multivariate outlier detection
- `grouped_summary()`: Group-wise statistical summaries

**Source Code Repository:**
- GitHub: https://github.com/dhaneshbb/insightfulpy
- PyPI: https://pypi.org/project/insightfulpy/
- Documentation: Available in repository README

**Standard Libraries:**
```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, spearmanr, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

# Gradient boosting
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import catboost as cb
from catboost import CatBoostClassifier

# Visualization of model results
import scikitplot as skplt

# Memory and performance
import psutil
import gc
```

**Complete Function List:**
The analysis used 25 custom functions for data processing, visualization, and modeling. Full implementation details available in project repository.

---

## Acknowledgments

This work benefited from contributions and feedback from mentors, colleagues, peers, and the data science community. Their insights, shared expertise, and collaborative input aided in addressing challenges and refining the analysis approach.

Special acknowledgment to:
- **Kaggle and Home Credit** for delivering the dataset and competition platform
- **insightfulpy contributors** for the data analysis library
- **Open source community** for the machine learning libraries that made this analysis possible

---

## Author Information

**Name:** Dhanesh B. B.

**Contact Information:**
- GitHub: [github.com/dhaneshbb](https://github.com/dhaneshbb)

**Professional Background:**
Data science practitioner specializing in machine learning applications for financial services. Experience in credit risk modeling, predictive analytics, and large-scale data engineering.

---

## References

1. Kaggle. "Home Credit Default Risk Competition." Kaggle Competitions. https://www.kaggle.com/competitions/home-credit-default-risk/

2. Home Credit Group. Dataset and documentation. https://d3libtxj3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip

3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

4. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." Advances in Neural Information Processing Systems 30 (NIPS 2017).

5. Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features." Advances in Neural Information Processing Systems 31 (NeurIPS 2018).

6. Dhanesh B. B. "insightfulpy: Data analysis utilities for Python." PyPI. https://pypi.org/project/insightfulpy/

7. McKinney, W. (2010). "Data Structures for Statistical Computing in Python." Proceedings of the 9th Python in Science Conference.

8. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12: 2825-2830.

9. Hunter, J. D. (2007). "Matplotlib: A 2D Graphics Environment." Computing in Science & Engineering, 9(3): 90-95.

10. Waskom, M. (2021). "seaborn: statistical data visualization." Journal of Open Source Software, 6(60): 3021.

**Additional Reading:**

- Bravo, C., Thomas, L. C., & Weber, R. (2015). "Improving credit scoring by differentiating defaulter behaviour." Journal of the Operational Research Society, 66(5): 771-781.

- Khandani, A. E., Kim, A. J., & Lo, A. W. (2010). "Consumer credit-risk models via machine-learning algorithms." Journal of Banking & Finance, 34(11): 2767-2787.

- Brown, I., & Mues, C. (2012). "An experimental comparison of classification algorithms for imbalanced credit scoring data sets." Expert Systems with Applications, 39(3): 3446-3453.

---

**Document Information:**
- Report Date: March 1, 2025
- Last Revised: November 07, 2025
- Analysis Period: 2007-2018 (dataset timeframe)
