# Table of Contents

- [Data Analysis](#data-analysis)
    - [Import data](#import-data)
    - [Imports & functions](#imports-functions)
  - [Data understanding](#data-understanding)
    - [1 application_train](#1-application-train)
    - [2 bureau and bureau_balance](#2-bureau-and-bureau-balance)
    - [3 previous_application](#3-previous-application)
    - [4 POS_CASH_balance](#4-pos-cash-balance)
    - [5 credit_card_balance](#5-credit-card-balance)
    - [6 installments_payments](#6-installments-payments)
- [Predictive Model](#predictive-model)
  - [Preprocessing](#preprocessing)
  - [Final_data](#final-data)
  - [Splitting](#splitting)
  - [Model Development](#model-development)
    - [Model Training & Evaluation](#model-training-evaluation)
    - [Model comparision & Interpretation](#model-comparision-interpretation)
    - [Best Model](#best-model)
    - [Saving the Model](#saving-the-model)
    - [Loading the Model Further use](#loading-the-model-further-use)
- [Table of Contents](#table-of-contents)
- [Acknowledgment](#acknowledgment)
- [Report](#report)
- [Author Information](#author-information)
- [References](#references)
- [Appendix](#appendix)
  - [Abbreviations for Datasets](#abbreviations-for-datasets)
  - [Source Code and Dependencies](#source-code-and-dependencies)


---

# Acknowledgment  

I would like to express my sincere gratitude to mentors, colleagues, peers, and the data science community for their unwavering support, constructive feedback, and encouragement throughout this project. Their insights, shared expertise, and collaborative spirit have been invaluable in overcoming challenges and refining my approach. I deeply appreciate the time and effort they have dedicated to helping me achieve success in this endeavor

---


# Report

  
  **Data Analysis Report for Project HomeLoanDef**

 **Introduction**
This report provides a detailed analysis of the datasets used in the "HomeLoanDef" project, which aims to predict loan defaults based on historical data. The datasets include application data, credit bureau data, POS/cash balance data, credit card balance data, previous application data, and installment payment data. The analysis covers data quality, feature engineering, data transformations, and insights for predictive modeling.


**Data Description**

- application_train.csv: Main dataset containing a binary target for each loan application (1: Defaulter, 0: Not a Defaulter). Each row represents one loan application.
- bureau.csv: Contains information about the client's previous credits from other financial institutions that were reported to the Credit Bureau. Each loan in sample has multiple rows corresponding to the number of previous credits the client had before the application date.
- bureau_balance.csv: Monthly balances of previous credits in the Credit Bureau. This dataset includes one row for each month of every previous credit reported to the Credit Bureau.
- POS_CASH_balance.csv: Monthly balance snapshots of previous POS and cash loans from Home Credit. This table records one row per month for each previous credit related to the loans in sample.
- credit_card_balance.csv: Details monthly balance snapshots of previous credit cards held with Home Credit. Each month of every previous credit card's history is recorded.
- **previous_application.csv: All previous loan applications made by clients who have current loans in sample. Each previous application has one row.
- installments_payments.csv: Repayment history for previously disbursed credits in Home Credit related to loans. This includes one row for each payment made or missed.

Each dataset is structured to provide a comprehensive history and status of the client's financial interactions with Home Credit and other institutions.

**1. Dataset Overview**

 1.1 Abbreviations and Memory Usage
| Dataset | Abbreviation | Rows | Columns | Memory Usage (MB) | Missing Values (%) | Negative Values (%) | Outliers (%) | Data Types |
|---------|--------------|------|---------|-------------------|--------------------|---------------------|--------------|------------|
| Application Train | at | 307,511 | 122 | 536.69 | 24.40% | 3.85% | 3.38% | float64: 65, int64: 41, object: 16 |
| Bureau | bu | 1,716,428 | 17 | 512.11 | 13.50% | 18.96% | 2.98% | float64: 8, int64: 6, object: 3 |
| Bureau Balance | bub | 27,299,925 | 3 | 1,926.61 | 0.00% | 32.59% | 0.00% | int64: 2, object: 1 |
| POS Cash Balance | pc | 10,001,358 | 8 | 1,137.25 | 0.07% | 12.50% | 2.00% | int64: 5, float64: 2, object: 1 |
| Credit Card Balance | ccb | 3,840,312 | 23 | 875.69 | 6.65% | 4.60% | 5.73% | float64: 15, int64: 7, object: 1 |
| Previous Application | pa | 1,670,214 | 37 | 1,900.63 | 17.98% | 9.20% | 2.76% | object: 16, float64: 15, int64: 6 |
| Installment Payments | ip | 13,605,401 | 8 | 830.41 | 0.01% | 25.00% | 4.18% | float64: 5, int64: 3 |

 1.2 Memory Optimization
- Memory usage was significantly reduced across datasets:
  - bu: Reduced by 64.7% (222.62 MB → 78.57 MB).
  - bub: Reduced by 75.0% (624.85 MB → 156.21 MB).
  - pc: Reduced by 72.3% (471.48 MB → 130.62 MB).
  - ip: Reduced by 62.5% (830.41 MB → 311.40 MB).

 1.3 Keys analysis

- Columns Across DataFrames

    - SK_ID_CURR: at, bu, pc, ccb, pa, ip
    - NAME_CONTRACT_TYPE: at, pa
    - AMT_CREDIT: at, pa
    - AMT_ANNUITY: at, bu, pa
    - AMT_GOODS_PRICE: at, pa
    - NAME_TYPE_SUITE: at, pa
    - WEEKDAY_APPR_PROCESS_START: at, pa
    - HOUR_APPR_PROCESS_START: at, pa
    - SK_ID_BUREAU: bu, bub
    - MONTHS_BALANCE: bub, pc, ccb
    - SK_ID_PREV: pc, ccb, pa, ip
    - NAME_CONTRACT_STATUS: pc, ccb, pa
    - SK_DPD: pc, ccb
    - SK_DPD_DEF: pc, ccb

 **2. Data Quality Assessment**

 2.1 Missing Values
- Application Train (at): The summary highlights significant missing data in several features, with the highest missing rates in attributes like COMMONAREA_MEDI, NONLIVINGAPARTMENTS_MODE, and FONDKAPREMONT_MODE, each exceeding 50%. The missing data affects both float16 and categorical data types across a wide range of variables, indicating potential gaps in data collection or entry. Notably, less critical features like NAME_TYPE_SUITE and AMT_ANNUITY show minimal missing values, suggesting better data consistency for certain financial details.
- Bureau (bu): High missing values in AMT_CREDIT_MAX_OVERDUE (72.17%), AMT_CREDIT_SUM_LIMIT (42.47%), and AMT_ANNUITY (41.78%).
- Previous Application (pa): Missing values in interest rates (RATE_INTEREST_PRIMARY, RATE_INTEREST_PRIVILEGED) exceeding 99.6%.
- The installment payments dataset tracks borrowers' repayment behavior, focusing on actual payments compared to expected installments. Missing values in DAYS_ENTRY_PAYMENT and AMT_PAYMENT (2.13%) 
- POS Cash Balance (pc) has minimal missing values, with CNT_INSTALMENT_FUTURE and CNT_INSTALMENT showing only 0.26% missing data, which is negligible. No infinite values are present, and there are no duplicate rows, ensuring data consistency. The completeness of the dataset makes it highly reliable for modeling installment loan behavior.
- credit card balance (ccb)primarily affect transaction and payment-related features, with around 20% missing data in transaction amounts and counts. These missing values likely indicate periods of inactivity rather than true data absence. Installment-related fields have lower missingness (~8%), suggesting variability in repayment structures across different credit card products. Since no infinite values or duplicates are present, data consistency is maintained. Addressing these missing values requires distinguishing between actual zero activity and true gaps, ensuring accurate imputations based on business logic.

 2.2 Negative Values
- Negative values in DAYS_BIRTH, DAYS_EMPLOYED, and DAYS_REGISTRATION were converted to absolute values for better interpretation.
- Installment Payments (ip): 99.99% of DAYS_INSTALMENT and 99.78% of DAYS_ENTRY_PAYMENT are negative, indicating past payment history.

 2.3 Outliers
- Extreme values in AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, and OWN_CAR_AGE were capped at the 1st and 99th percentiles.
- Credit Card Balance (ccb): Outliers in AMT_BALANCE, AMT_DRAWINGS_ATM_CURRENT, and AMT_PAYMENT_CURRENT were capped.

2.4 duplecated values

- The installment payments dataset tracks borrowers' repayment behavior, focusing on actual payments compared to expected installments. Missing values in DAYS_ENTRY_PAYMENT and AMT_PAYMENT (2.13%) suggest some transactions lack recorded payment details. While no infinite values were found, 15 duplicate records indicate minor redundancy.
    

 **3. Feature Engineering & Transformations**
 1. Application Train Dataset (at)
- Key Transformations:
  - Categorical Encoding:
    - Binary features (FLAG_OWN_CAR, FLAG_OWN_REALTY) mapped to 0/1.
    - Label encoding for CODE_GENDER, NAME_EDUCATION_TYPE, NAME_CONTRACT_TYPE.
    - One-hot encoding for NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, WEEKDAY_APPR_PROCESS_START.
    - Frequency encoding for OCCUPATION_TYPE, ORGANIZATION_TYPE.
  - Derived Features:
    - EMPLOYMENT_LENGTH (from DAYS_EMPLOYED), AGE (from DAYS_BIRTH).
    - Financial ratios: INCOME_TO_CREDIT_RATIO, INCOME_TO_ANNUITY_RATIO, PAYMENT_RATE.
    - Weighted external risk score: WEIGHTED_EXT_SOURCE (using EXT_SOURCE_1/2/3).
  - Data Cleaning:
    - Converted negative days to absolute values (e.g., DAYS_BIRTH, DAYS_EMPLOYED).
    - Capped outliers in AMT_INCOME_TOTAL, AMT_CREDIT, and OWN_CAR_AGE at 1st/99th percentiles.
    - Dropped leakage-prone features (e.g., REGION_RATING_CLIENT_W_CITY, DEF_30_CNT_SOCIAL_CIRCLE).

 2. Bureau & Bureau Balance (bu_bub)
- Key Transformations:
  - Merged Datasets: Joined bu and bub on SK_ID_BUREAU.
  - Feature Engineering:
    - CREDIT_DURATION: Converted DAYS_CREDIT to years.
    - TOTAL_MONTHS: Total months of active credit per client.
    - Encoded STATUS based on overdue risk (e.g., 0 = no overdue, C = closed).
  - Aggregations:
    - Summarized AMT_CREDIT_SUM, AMT_ANNUITY by SK_ID_CURR.
    - Computed mean/sum for DAYS_CREDIT_ENDDATE, DAYS_CREDIT_UPDATE.
  - Data Cleaning:
    - Dropped high-leakage columns (DAYS_ENDDATE_FACT, CREDIT_DAY_OVERDUE).
    - Clipped negative values in AMT_CREDIT_SUM_LIMIT.

 3. Previous Application (pa)
- Key Transformations:
  - New Features:
    - Financial ratios: INCOME_TO_CREDIT_RATIO, DOWN_PAYMENT_RATIO.
    - Interaction terms: CREDIT_TO_ANNUITY_RATIO, LOAN_TO_GOODS_PRICE.
  - Categorical Encoding:
    - One-hot encoding for NAME_CONTRACT_TYPE, NAME_CLIENT_TYPE.
    - Frequency encoding for NAME_GOODS_CATEGORY, PRODUCT_COMBINATION.
  - Aggregations:
    - Client-level summaries for AMT_CREDIT, AMT_ANNUITY (mean, sum, variance).
  - Data Cleaning:
    - Dropped columns with >99% missing values (RATE_INTEREST_PRIMARY).
    - Capped extreme values in AMT_CREDIT, AMT_GOODS_PRICE.

 4. POS Cash Balance (pc)
- Key Transformations:
  - New Features:
    - IS_OVERDUE: Binary flag for overdue payments (SK_DPD > 0).
    - INSTALLMENT_PROGRESS: Ratio of completed installments.
    - DAYS_TO_DEFAULT: Days between first overdue and default.
  - Aggregations:
    - Client-level stats for SK_DPD, CNT_INSTALMENT_FUTURE (min, max, mean).
  - Data Cleaning:
    - Replaced negative values in CNT_INSTALMENT with NaN.
    - Dropped redundant columns (SK_DPD, SK_DPD_DEF due to multicollinearity).

 5. Credit Card Balance (ccb)
- Key Transformations:
  - New Features:
    - CREDIT_UTILIZATION_RATIO: AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL.
    - PAYMENT_RATIO: AMT_PAYMENT_TOTAL_CURRENT / AMT_BALANCE.
  - Aggregations:
    - Client-level summaries for AMT_DRAWINGS_ATM_CURRENT, CNT_DRAWINGS_POS_CURRENT.
  - Data Cleaning:
    - Capped outliers in CNT_INSTALMENT_MATURE_CUM, AMT_BALANCE.
    - Frequency-encoded NAME_CONTRACT_STATUS.

 6. Installments Payments (ip)
- Key Transformations:
  - New Features:
    - TIME_TO_PAYMENT: Difference between scheduled and actual payment days.
    - IS_LATE_PAYMENT: Flag for late payments (TIME_TO_PAYMENT > 0).
  - Aggregations:
    - Client-level stats for AMT_INSTALMENT, AMT_PAYMENT (mean, sum).
  - Data Cleaning:
    - Capped extreme values in AMT_INSTALMENT, AMT_PAYMENT.
    - Removed duplicates to ensure data uniqueness.

 Final Dataset Integration
- Merging Strategy:
  - Aggregated features from all datasets merged into final_data using SK_ID_CURR.
  - Removed duplicate client IDs and high-missing columns (>75% missing).
- Feature Selection:
  - Dropped low-importance features using XGBoost’s feature_importances_.
  - Addressed multicollinearity via VIF analysis and correlation thresholds (|r| > 0.85).
- Memory Optimization:
  - Reduced memory usage by 60–80% using dtype downcasting (e.g., float64 → float16).

This structured approach ensures clean, interpretable, and model-ready data for predicting loan defaults.


 **4. Multicollinearity & Data Leakage**

 4.1 Multicollinearity

- Analyzing dataset: at Highly correlated pairs in at (threshold > 0.8):
     - CNT_CHILDREN and CNT_FAM_MEMBERS, Correlation: 0.8792
     - AMT_CREDIT and AMT_GOODS_PRICE, Correlation: 0.9870
     - DAYS_EMPLOYED and FLAG_EMP_PHONE, Correlation: 0.9998
     - REG_REGION_NOT_WORK_REGION and LIVE_REGION_NOT_WORK_REGION, Correlation: 0.8606
     - REG_CITY_NOT_WORK_CITY and LIVE_CITY_NOT_WORK_CITY, Correlation: 0.8256
     - APARTMENTS_AVG and ELEVATORS_AVG, Correlation: 0.8370
     - APARTMENTS_AVG and LIVINGAPARTMENTS_AVG, Correlation: 0.9440
     - APARTMENTS_AVG and LIVINGAREA_AVG, Correlation: 0.9136
     - APARTMENTS_AVG and APARTMENTS_MODE, Correlation: 0.9733
     - APARTMENTS_AVG and ELEVATORS_MODE, Correlation: 0.8226
     - APARTMENTS_AVG and LIVINGAPARTMENTS_MODE, Correlation: 0.9306
     - APARTMENTS_AVG and LIVINGAREA_MODE, Correlation: 0.8935
     - APARTMENTS_AVG and APARTMENTS_MEDI, Correlation: 0.9951
     - APARTMENTS_AVG and ELEVATORS_MEDI, Correlation: 0.8351
     - APARTMENTS_AVG and LIVINGAPARTMENTS_MEDI, Correlation: 0.9419
     - APARTMENTS_AVG and LIVINGAREA_MEDI, Correlation: 0.9123
     - APARTMENTS_AVG and TOTALAREA_MODE, Correlation: 0.8926
     - BASEMENTAREA_AVG and BASEMENTAREA_MODE, Correlation: 0.9735
     - BASEMENTAREA_AVG and BASEMENTAREA_MEDI, Correlation: 0.9943
     - YEARS_BEGINEXPLUATATION_AVG and YEARS_BEGINEXPLUATATION_MODE, Correlation: 0.9719
     - YEARS_BEGINEXPLUATATION_AVG and YEARS_BEGINEXPLUATATION_MEDI, Correlation: 0.9938
     - YEARS_BUILD_AVG and YEARS_BUILD_MODE, Correlation: 0.9894
     - YEARS_BUILD_AVG and YEARS_BUILD_MEDI, Correlation: 0.9985
     - COMMONAREA_AVG and COMMONAREA_MODE, Correlation: 0.9771
     - COMMONAREA_AVG and COMMONAREA_MEDI, Correlation: 0.9960
     - ELEVATORS_AVG and LIVINGAPARTMENTS_AVG, Correlation: 0.8118
     - ELEVATORS_AVG and LIVINGAREA_AVG, Correlation: 0.8678
     - ELEVATORS_AVG and APARTMENTS_MODE, Correlation: 0.8060
     - ELEVATORS_AVG and ELEVATORS_MODE, Correlation: 0.9788
     - ELEVATORS_AVG and LIVINGAREA_MODE, Correlation: 0.8389
     - ELEVATORS_AVG and APARTMENTS_MEDI, Correlation: 0.8345
     - ELEVATORS_AVG and ELEVATORS_MEDI, Correlation: 0.9961
     - ELEVATORS_AVG and LIVINGAPARTMENTS_MEDI, Correlation: 0.8126
     - ELEVATORS_AVG and LIVINGAREA_MEDI, Correlation: 0.8656
     - ELEVATORS_AVG and TOTALAREA_MODE, Correlation: 0.8446
     - ENTRANCES_AVG and ENTRANCES_MODE, Correlation: 0.9777
     - ENTRANCES_AVG and ENTRANCES_MEDI, Correlation: 0.9969
     - FLOORSMAX_AVG and FLOORSMAX_MODE, Correlation: 0.9857
     - FLOORSMAX_AVG and FLOORSMAX_MEDI, Correlation: 0.9970
     - FLOORSMIN_AVG and FLOORSMIN_MODE, Correlation: 0.9859
     - FLOORSMIN_AVG and FLOORSMIN_MEDI, Correlation: 0.9972
     - LANDAREA_AVG and LANDAREA_MODE, Correlation: 0.9737
     - LANDAREA_AVG and LANDAREA_MEDI, Correlation: 0.9916
     - LIVINGAPARTMENTS_AVG and LIVINGAREA_AVG, Correlation: 0.8808
     - LIVINGAPARTMENTS_AVG and APARTMENTS_MODE, Correlation: 0.9083
     - LIVINGAPARTMENTS_AVG and LIVINGAPARTMENTS_MODE, Correlation: 0.9701
     - LIVINGAPARTMENTS_AVG and LIVINGAREA_MODE, Correlation: 0.8517
     - LIVINGAPARTMENTS_AVG and APARTMENTS_MEDI, Correlation: 0.9356
     - LIVINGAPARTMENTS_AVG and ELEVATORS_MEDI, Correlation: 0.8093
     - LIVINGAPARTMENTS_AVG and LIVINGAPARTMENTS_MEDI, Correlation: 0.9938
     - LIVINGAPARTMENTS_AVG and LIVINGAREA_MEDI, Correlation: 0.8783
     - LIVINGAPARTMENTS_AVG and TOTALAREA_MODE, Correlation: 0.8480
     - LIVINGAREA_AVG and APARTMENTS_MODE, Correlation: 0.8907
     - LIVINGAREA_AVG and ELEVATORS_MODE, Correlation: 0.8526
     - LIVINGAREA_AVG and LIVINGAPARTMENTS_MODE, Correlation: 0.8731
     - LIVINGAREA_AVG and LIVINGAREA_MODE, Correlation: 0.9720
     - LIVINGAREA_AVG and APARTMENTS_MEDI, Correlation: 0.9125
     - LIVINGAREA_AVG and ELEVATORS_MEDI, Correlation: 0.8658
     - LIVINGAREA_AVG and LIVINGAPARTMENTS_MEDI, Correlation: 0.8832
     - LIVINGAREA_AVG and LIVINGAREA_MEDI, Correlation: 0.9956
     - LIVINGAREA_AVG and TOTALAREA_MODE, Correlation: 0.9250
     - NONLIVINGAPARTMENTS_AVG and NONLIVINGAPARTMENTS_MODE, Correlation: 0.9694
     - NONLIVINGAPARTMENTS_AVG and NONLIVINGAPARTMENTS_MEDI, Correlation: 0.9908
     - NONLIVINGAREA_AVG and NONLIVINGAREA_MODE, Correlation: 0.9661
     - NONLIVINGAREA_AVG and NONLIVINGAREA_MEDI, Correlation: 0.9904
     - APARTMENTS_MODE and ELEVATORS_MODE, Correlation: 0.8262
     - APARTMENTS_MODE and LIVINGAPARTMENTS_MODE, Correlation: 0.9378
     - APARTMENTS_MODE and LIVINGAREA_MODE, Correlation: 0.9104
     - APARTMENTS_MODE and APARTMENTS_MEDI, Correlation: 0.9772
     - APARTMENTS_MODE and ELEVATORS_MEDI, Correlation: 0.8089
     - APARTMENTS_MODE and LIVINGAPARTMENTS_MEDI, Correlation: 0.9145
     - APARTMENTS_MODE and LIVINGAREA_MEDI, Correlation: 0.8941
     - APARTMENTS_MODE and TOTALAREA_MODE, Correlation: 0.8636
     - BASEMENTAREA_MODE and BASEMENTAREA_MEDI, Correlation: 0.9779
     - YEARS_BEGINEXPLUATATION_MODE and YEARS_BEGINEXPLUATATION_MEDI, Correlation: 0.9635
     - YEARS_BUILD_MODE and YEARS_BUILD_MEDI, Correlation: 0.9895
     - COMMONAREA_MODE and COMMONAREA_MEDI, Correlation: 0.9799
     - ELEVATORS_MODE and LIVINGAPARTMENTS_MODE, Correlation: 0.8076
     - ELEVATORS_MODE and LIVINGAREA_MODE, Correlation: 0.8560
     - ELEVATORS_MODE and APARTMENTS_MEDI, Correlation: 0.8256
     - ELEVATORS_MODE and ELEVATORS_MEDI, Correlation: 0.9828
     - ELEVATORS_MODE and LIVINGAREA_MEDI, Correlation: 0.8558
     - ELEVATORS_MODE and TOTALAREA_MODE, Correlation: 0.8208
     - ENTRANCES_MODE and ENTRANCES_MEDI, Correlation: 0.9807
     - FLOORSMAX_MODE and FLOORSMAX_MEDI, Correlation: 0.9882
     - FLOORSMIN_MODE and FLOORSMIN_MEDI, Correlation: 0.9884
     - LANDAREA_MODE and LANDAREA_MEDI, Correlation: 0.9808
     - LIVINGAPARTMENTS_MODE and LIVINGAREA_MODE, Correlation: 0.8785
     - LIVINGAPARTMENTS_MODE and APARTMENTS_MEDI, Correlation: 0.9322
     - LIVINGAPARTMENTS_MODE and LIVINGAPARTMENTS_MEDI, Correlation: 0.9756
     - LIVINGAPARTMENTS_MODE and LIVINGAREA_MEDI, Correlation: 0.8744
     - LIVINGAPARTMENTS_MODE and TOTALAREA_MODE, Correlation: 0.8339
     - LIVINGAREA_MODE and APARTMENTS_MEDI, Correlation: 0.8961
     - LIVINGAREA_MODE and ELEVATORS_MEDI, Correlation: 0.8411
     - LIVINGAREA_MODE and LIVINGAPARTMENTS_MEDI, Correlation: 0.8574
     - LIVINGAREA_MODE and LIVINGAREA_MEDI, Correlation: 0.9747
     - LIVINGAREA_MODE and TOTALAREA_MODE, Correlation: 0.8992
     - NONLIVINGAPARTMENTS_MODE and NONLIVINGAPARTMENTS_MEDI, Correlation: 0.9786
     - NONLIVINGAREA_MODE and NONLIVINGAREA_MEDI, Correlation: 0.9758
     - APARTMENTS_MEDI and ELEVATORS_MEDI, Correlation: 0.8374
     - APARTMENTS_MEDI and LIVINGAPARTMENTS_MEDI, Correlation: 0.9425
     - APARTMENTS_MEDI and LIVINGAREA_MEDI, Correlation: 0.9159
     - APARTMENTS_MEDI and TOTALAREA_MODE, Correlation: 0.8866
     - ELEVATORS_MEDI and LIVINGAPARTMENTS_MEDI, Correlation: 0.8142
     - ELEVATORS_MEDI and LIVINGAREA_MEDI, Correlation: 0.8683
     - ELEVATORS_MEDI and TOTALAREA_MODE, Correlation: 0.8380
     - LIVINGAPARTMENTS_MEDI and LIVINGAREA_MEDI, Correlation: 0.8847
     - LIVINGAPARTMENTS_MEDI and TOTALAREA_MODE, Correlation: 0.8461
     - LIVINGAREA_MEDI and TOTALAREA_MODE, Correlation: 0.9194
     - OBS_30_CNT_SOCIAL_CIRCLE and OBS_60_CNT_SOCIAL_CIRCLE, Correlation: 0.9985

- Analyzing dataset: BU_BUB Highly correlated pairs in bu_bub (threshold > 0.8):
     - DAYS_CREDIT and DAYS_ENDDATE_FACT, Correlation: 0.8257

- Analyzing dataset: PC Highly correlated pairs in pc (threshold > 0.8):
     - CNT_INSTALMENT and CNT_INSTALMENT_FUTURE, Correlation: 0.8713

- Analyzing dataset: CCB Highly correlated pairs in ccb (threshold > 0.8):
     - AMT_BALANCE and AMT_INST_MIN_REGULARITY, Correlation: 0.8967
     - AMT_BALANCE and AMT_RECEIVABLE_PRINCIPAL, Correlation: 0.9997
     - AMT_BALANCE and AMT_RECIVABLE, Correlation: 0.9999
     - AMT_BALANCE and AMT_TOTAL_RECEIVABLE, Correlation: 0.9999
     - AMT_DRAWINGS_ATM_CURRENT and AMT_DRAWINGS_CURRENT, Correlation: 0.8002
     - AMT_INST_MIN_REGULARITY and AMT_RECEIVABLE_PRINCIPAL, Correlation: 0.8960
     - AMT_INST_MIN_REGULARITY and AMT_RECIVABLE, Correlation: 0.8976
     - AMT_INST_MIN_REGULARITY and AMT_TOTAL_RECEIVABLE, Correlation: 0.8976
     - AMT_PAYMENT_CURRENT and AMT_PAYMENT_TOTAL_CURRENT, Correlation: 0.9948
     - AMT_RECEIVABLE_PRINCIPAL and AMT_RECIVABLE, Correlation: 0.9997
     - AMT_RECEIVABLE_PRINCIPAL and AMT_TOTAL_RECEIVABLE, Correlation: 0.9997
     - AMT_RECIVABLE and AMT_TOTAL_RECEIVABLE, Correlation: 1.0000
     - CNT_DRAWINGS_CURRENT and CNT_DRAWINGS_POS_CURRENT, Correlation: 0.9505


- Analyzing dataset: PA Highly correlated pairs in pa (threshold > 0.8):
     - AMT_ANNUITY and AMT_APPLICATION, Correlation: 0.8089
     - AMT_ANNUITY and AMT_CREDIT, Correlation: 0.8164
     - AMT_ANNUITY and AMT_GOODS_PRICE, Correlation: 0.8209
     - AMT_APPLICATION and AMT_CREDIT, Correlation: 0.9758
     - AMT_APPLICATION and AMT_GOODS_PRICE, Correlation: 0.9999
     - AMT_CREDIT and AMT_GOODS_PRICE, Correlation: 0.9931
     - DAYS_FIRST_DRAWING and DAYS_LAST_DUE_1ST_VERSION, Correlation: 0.8035
     - DAYS_LAST_DUE and DAYS_TERMINATION, Correlation: 0.9280


- Analyzing dataset: IP Highly correlated pairs in ip (threshold > 0.8):
     - DAYS_INSTALMENT and DAYS_ENTRY_PAYMENT, Correlation: 0.9995
     - AMT_INSTALMENT and AMT_PAYMENT, Correlation: 0.9372


 4.2 Data Leakage
- Application Train (at):
  - REGION_RATING_CLIENT_W_CITY, REGION_RATING_CLIENT, and social circle features (OBS_30_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE) showed high correlations with TARGET and were removed.
- Previous Application (pa):
  - High leakage variables (DAYS_TERMINATION, DAYS_LAST_DUE) were dropped.



**5. Statistical Insights**

 5.1 Application Train (at)
- Older applicants (higher DAYS_BIRTH) had lower default rates.
- EXT_SOURCE_2 had the strongest negative correlation with TARGET, implying lower risk scores were linked to better loan repayment.

 5.2 Bureau & Bureau Balance (bu_bub)
- CREDIT_ACTIVE: Majority of records belong to closed accounts.
- CREDIT_TYPE: Cash loans were the most frequent.

 5.3 Credit Card Balance (ccb)
- 96.31% of credit card contracts are active.
- High credit utilization ratios were observed, indicating potential financial stress.

 5.4 Installment Payments (ip)
- Most borrowers follow the installment schedule, but late payments exist.
- Strong correlation between scheduled and actual payments suggests timely payments.



 **6. Conclusion**
This  analysis ensures that the datasets are cleaned, transformed, and optimized for predictive modeling. Key findings include:
- Missing values and outliers were handled effectively.
- Feature engineering enhanced predictive power.
- Multicollinearity and data leakage issues were addressed.
- Aggregated features provide a holistic view of client credit history.

The final datasets are now ready for use in machine learning models, ensuring minimal bias and improved interpretability.

- note: here only the major finding and key insights explained for more information visit their dataset section wise alaysis,

----

**Final Model Comparison and Report: XGBoost Classifier**

 Executive Summary  
The XGBoost model is selected as the final model for loan default prediction due to its balanced performance across critical metrics (accuracy: 83%, recall: 53%, precision: 24%, F1-score: 0.33) and robustness to class imbalance. Key findings include:
- Best Threshold: 0.60 optimizes the trade-off between detecting defaulters (recall) and minimizing false alarms (precision).  
- Feature Importance: External credit scores (EXT_SOURCE), demographic factors, and credit utilization drive predictions.  
- Generalization: Stable cross-validation performance (F1-score: 0.301 ± 0.0025) and moderate overfitting (1.6% gap between training/test accuracy).



 Model Comparison  
 Base Models (Before Tuning)
| Model          | Accuracy | Precision | Recall | F1-Score | ROC AUC | Overfit (%) | Training Time (s) |
|----------------|----------|-----------|--------|----------|---------|-------------|-------------------|
| LightGBM   | 72.38%   | 18.27%    | 69.69% | 0.289    | 0.781   | 0.95        | 22.92            |
| CatBoost   | 73.66%   | 18.52%    | 66.57% | 0.290    | 0.771   | 1.29        | 45.83            |
| XGBoost    | 77.37%   | 19.95%    | 59.88% | 0.299    | 0.768   | 3.64        | 28.18            |
| Random Forest  | 91.93%   | 48.00%    | 0.24%  | 0.005    | 0.737   | 8.07        | 637.59           |

Insights:  
- Random Forest failed due to extreme overfitting and near-zero recall.  
- LightGBM prioritized recall (70%) but sacrificed precision (18%).  
- XGBoost offered the best balance between accuracy (77%) and recall (60%).



 Optimized Models (After Tuning)
| Model          | Accuracy | Precision | Recall | F1-Score | ROC AUC | Overfit (%) | Training Time (s) |
|----------------|----------|-----------|--------|----------|---------|-------------|-------------------|
| XGBoost    | 75.30%   | 19.59%    | 66.30% | 0.302    | 0.785   | 1.63        | 98.10            |
| LightGBM   | 74.06%   | 19.11%    | 68.46% | 0.299    | 0.787   | 1.39        | 88.67            |
| CatBoost   | 78.99%   | 21.22%    | 59.03% | 0.312    | 0.776   | 4.13        | 147.94           |

Key Improvements:  
- XGBoost increased recall from 60% to 66% while maintaining precision.  
- LightGBM achieved the highest ROC AUC (0.787) but lower precision.  
- CatBoost improved accuracy (79%) but overfit more heavily (4.13%).



 Feature Importance  
Top 5 Features:  
1. EXT_SOURCE_3 (External credit score)  
2. EXT_SOURCE_2 (External credit score)  
3. WEIGHTED_EXT_SOURCE (Composite credit score)  
4. NAME_EDUCATION_TYPE (Education level)  
5. CODE_GENDER (Gender)  

Insights:  
- External credit scores and demographic variables dominate predictions.  
- Credit card utilization (CCB_CREDIT_UTILIZATION_RATIO) and repayment history (PAYMENT_RATE) are critical behavioral factors.  



 Final Model Performance  
 Threshold Analysis (XGBoost)  
| Threshold | Precision | Recall | F1-Score | Accuracy | Business Impact                          |
|-----------|-----------|--------|----------|----------|------------------------------------------|
| 0.60  | 24%       | 53%    | 0.33     | 83%      | Optimal: Balances risk detection and FP reduction. |
| 0.50      | 19.6%     | 66%    | 0.30     | 75%      | Higher recall but costly FP rate.        |
| 0.70      | 30.8%     | 36%    | 0.33     | 88%      | Lower risk tolerance (fewer FPs).        |

 Confusion Matrix at Threshold 0.60  
- True Negatives (TN): 48,450 (Correctly identified non-defaulters)  
- False Positives (FP): 8,088 (Non-defaulters flagged as risky)  
- False Negatives (FN): 2,356 (Missed defaulters)  
- True Positives (TP): 2,609 (Correctly identified defaulters)  



 Business Implications  
1. Risk vs. Cost Trade-off:  
   - At threshold 0.60, the model detects 53% of defaulters but flags 14% of non-defaulters as risky.  
   - Lower thresholds increase defaulter detection (recall) but raise operational costs (e.g., manual reviews of FPs).  

2. Calibration:  
   - The model underestimates default probabilities (calibration curve skews downward).  
   - Prioritize applicants with predicted probabilities >0.60 for additional scrutiny.  

3. Economic Impact:  
   - FN Cost: Missing a defaulter could result in a full loan loss (~$10,000 average).  
   - FP Cost: Reviewing a false alarm might cost ~$100 in labor.  



 Recommendations  
1. Deploy XGBoost with a decision threshold of 0.60 for balanced risk management.  
2. Monitor feature stability (e.g., external credit scores) to ensure model reliability.  
3. Retrain quarterly to adapt to shifting borrower behavior.  



 Conclusion  
The XGBoost model strikes the best balance between accuracy, recall, and computational efficiency for loan default prediction. By prioritizing critical features like credit scores and repayment history, it provides actionable insights while minimizing financial risk. Further refinement of probability calibration could enhance its precision in high-risk segments.  

 
Model Saved As: final_xgb_model.pkl  
Key Metric: ROC AUC = 0.785 (Strong class separation).


----

**Challenges Faced Report**

 1. Overview  
The HomeLoanDef project involved analyzing seven interconnected datasets to predict loan defaults. During this process,  encountered several data challenges spanning memory management, missing values, outlier detection, multicollinearity, data leakage, categorical variable processing, temporal aggregation, class imbalance, and model-specific issues. The following bullet-point paragraphs detail each challenge alongside the techniques used to overcome them, including the reasoning behind choices.


 2. Key Challenges & Solutions

- High Memory Usage  
  • *Challenge*: The raw datasets, particularly bureau_balance (over 1.9 GB) and the merged bureau datasets, consumed excessive memory, making processing and merging operations highly resource-intensive.  
  • *Solution*:  optimized memory usage by downcasting numerical columns (e.g., converting int64 to int16 and float64 to float32) and converting string-based columns to categorical data types.  
  • *Reason*: These techniques resulted in a significant reduction (approximately 70–80%) in memory usage, enabling more efficient data processing and faster model training.

- Missing Values  
  • *Challenge*: Datasets such as application_train.csv and previous_application.csv exhibited high missing value percentages (e.g., 24.4% missing in key features like EXT_SOURCE_1 and over 50% in some fields), complicating analysis and model development.  
  • *Solution*:  implemented targeted imputation strategies by using the median for numerical features with skewed distributions and creating a dedicated “Missing” category for categorical variables. In some cases,  dropped columns that exceeded a 75% missing threshold.  
  • *Reason*: This approach helped maintain data integrity and minimized bias, ensuring that imputation did not distort the underlying relationships crucial for accurate prediction.

- Outliers & Skewed Distributions  
  • *Challenge*: Extreme values in variables such as AMT_INCOME_TOTAL and DAYS_EMPLOYED distorted the data distribution and potentially skewed model outcomes.  
  • *Solution*:  applied winsorization by capping outliers at the 1st and 99th percentiles and considered log transformations where appropriate (although in some cases, transformations were avoided due to interpretability concerns).  
  • *Reason*: This technique mitigated the undue influence of outliers while preserving the natural variance in the data, leading to more stable and interpretable model performance.

- Data Leakage  
  • *Challenge*: Certain features, like DAYS_LAST_PHONE_CHANGE and social circle metrics (e.g., DEF_30_CNT_SOCIAL_CIRCLE), risked introducing data leakage by including post-application information that could inadvertently hint at the target variable.  
  • *Solution*:  proactively removed features identified as potential leakage, ensuring that only information available at the time of application was used for modeling.  
  • *Reason*: Eliminating these variables was critical to ensure model generalizability and to prevent artificially inflated performance metrics that would not hold in real-world applications.

- Multicollinearity  
  • *Challenge*: The presence of highly correlated features—such as the near-perfect correlation between AMT_CREDIT and AMT_GOODS_PRICE—posed risks to model stability and interpretability.  
  • *Solution*:  conducted Variance Inflation Factor (VIF) analysis to identify and remove features with VIF values exceeding 8, and then applied domain knowledge to retain only the most informative variables from highly correlated pairs.  
  • *Reason*: This reduction in multicollinearity improved model interpretability and stability by ensuring that redundant information did not adversely affect parameter estimates.

- Categorical Variables  
  • *Challenge*: High-cardinality categorical features (e.g., ORGANIZATION_TYPE with 58 categories) and inconsistent data types (such as string representations for dates) complicated the modeling process.  
  • *Solution*:  employed frequency encoding for high-cardinality features and one-hot encoding for features with fewer categories, standardizing mixed-type variables where necessary.  
  • *Reason*: These encoding strategies balanced the need to reduce dimensionality while preserving the inherent information within the categorical variables, leading to better model performance.

- Temporal & Behavioral Aggregation  
  • *Challenge*: Datasets like installments_payments.csv (with over 13 million rows) and credit_card_balance.csv required aggregation to the client level to derive meaningful insights without losing temporal patterns.  
  • *Solution*:  aggregated transactional data by computing summary statistics (mean, sum, max) for features such as AMT_PAYMENT and created time-based variables like TIME_TO_PAYMENT and CREDIT_DURATION.  
  • *Reason*: This allowed us to convert granular, high-frequency data into actionable client-level insights, facilitating more effective predictive modeling.

- Class Imbalance  
  • *Challenge*: The target variable was heavily imbalanced, with only about 8% of cases being defaults.  
  • *Solution*:  addressed this imbalance by adjusting model parameters (using XGBoost’s scale_pos_weight), and fine-tuned the decision threshold (optimizing at a threshold of 0.60) to enhance the F1-score.  
  • *Reason*: Focusing on recall for the minority class (defaulters) while managing false positives improved the model’s real-world applicability in identifying high-risk cases.

- Hyperparameter Tuning & Model Overfitting  
  • *Challenge*: With high-dimensional data, there was a significant risk of overfitting, which could lead to poor generalization on unseen data.  
  • *Solution*:  employed grid search combined with cross-validation and early stopping strategies to optimize key hyperparameters such as max_depth, learning_rate, and regularization terms.  
  • *Reason*: These techniques ensured that models, particularly XGBoost, maintained a balance between complexity and generalizability, as evidenced by robust cross-validation performance and controlled overfitting.

 3. Conclusion

The HomeLoanDef project required overcoming multiple data challenges through a blend of robust engineering practices, advanced statistical techniques, and domain prespective, By addressing issues related to memory usage, missing values, outliers, data leakage, multicollinearity, and class imbalance,  were able to transform raw, complex data into a refined, predictive dataset. The strategies implemented, were critical in enhancing model performance and ensuring reliability.


---

# Author Information

-  Dhanesh B. B.  

- Contact Information:  
    - [Email](dhaneshbb5@gmail.com) 
    - [LinkedIn](https://www.linkedin.com/in/dhanesh-b-b-2a8971225/) 
    - [GitHub](https://github.com/dhaneshbb)


----

# References

- [1] Kaggle, "Home Credit Default Risk Competition Data Diagram," available online: https://www.kaggle.com/competitions/home-credit-default-risk/data.

- The memory optimization technique used in this project is adapted from commonly used practices within the Python data science community, as discussed in various technical forums and blogs.

- datasets - https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip 

---

# Appendix

## Abbreviations for Datasets

- **`at`** → `application_train.csv`
- **`bu`** → `bureau.csv`
- **`bub`** → `bureau_balance.csv`
- **`pc`** → `POS_CASH_balance.csv`
- **`ccb`** → `credit_card_balance.csv`
- **`pa`** → `previous_application.csv`
- **`ip`** → `installments_payments.csv`

---

## Source Code and Dependencies

In the development of the predictive models for this project, I extensively utilized several functions from my custom library "insightfulpy." This library, available on both GitHub and PyPI, provided crucial functionalities that enhanced the data analysis and modeling process. For those interested in exploring the library or using it in their own projects, you can inspect the source code and documentation available. The functions from "insightfulpy" helped streamline data preprocessing, feature engineering, and model evaluation, making the analytic processes more efficient and reproducible.

You can find the source and additional resources on GitHub here: [insightfulpy on GitHub](https://github.com/dhaneshbb/insightfulpy), and for installation or further documentation, visit [insightfulpy on PyPI](https://pypi.org/project/insightfulpy/). These resources provide a comprehensive overview of the functions available and instructions on how to integrate them into your data science workflows.

---

Below is an overview of each major tool (packages, user-defined functions, and imported functions) that appears in this project.

<pre>
Imported packages:
1: builtins
2: builtins
3: pandas
4: warnings
5: researchpy
6: matplotlib.pyplot
7: missingno
8: seaborn
9: numpy
10: scipy.stats
11: textwrap
12: logging
13: statsmodels.api
14: time
15: xgboost
16: lightgbm
17: catboost
18: scikitplot
19: psutil
20: os
21: gc
22: joblib
23: types
24: inspect

User-defined functions:
1: plot_boxplots
2: memory_usage
3: dataframe_memory_usage
4: garbage_collection
5: single_value_columns
6: plot_histograms
7: plot_scatter
8: plot_bar_plots
9: plot_scatter_df
10: reduce_mem_usage
11: reduce_mem_usagewithout_causing_cat
12: optimize_data_types
13: chi_square_test
14: fisher_exact_test
15: spearman_correlation_with_target
16: normality_test_with_skew_kurt
17: spearman_correlation
18: calculate_vif
19: cap_extreme_values
20: cap_outliers
21: evaluate_model
22: cross_validation_analysis_table
23: threshold_analysis
24: plot_all_evaluation_metrics
25: show_default_feature_importance

Imported functions:
1: open
2: tabulate
3: display
4: is_datetime64_any_dtype
5: skew
6: kurtosis
7: shapiro
8: kstest
9: compare_df_columns
10: linked_key
11: display_key_columns
12: interconnected_outliers
13: grouped_summary
14: calc_stats
15: iqr_trimmed_mean
16: mad
17: comp_cat_analysis
18: comp_num_analysis
19: detect_mixed_data_types
20: missing_inf_values
21: columns_info
22: cat_high_cardinality
23: analyze_data
24: num_summary
25: cat_summary
26: calculate_skewness_kurtosis
27: detect_outliers
28: show_missing
29: kde_batches
30: box_plot_batches
31: qq_plot_batches
32: num_vs_num_scatterplot_pair_batch
33: cat_vs_cat_pair_batch
34: num_vs_cat_box_violin_pair_batch
35: cat_bar_batches
36: cat_pie_chart_batches
37: num_analysis_and_plot
38: cat_analyze_and_plot
39: chi2_contingency
40: fisher_exact
41: pearsonr
42: spearmanr
43: ttest_ind
44: mannwhitneyu
45: linkage
46: dendrogram
47: leaves_list
48: variance_inflation_factor
49: train_test_split
50: cross_val_score
51: learning_curve
52: resample
53: compute_class_weight
54: accuracy_score
55: precision_score
56: recall_score
57: f1_score
58: roc_auc_score
59: confusion_matrix
60: precision_recall_curve
61: roc_curve
62: auc
63: calibration_curve
64: classification_report
</pre>
