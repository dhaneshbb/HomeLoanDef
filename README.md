
# Home Loan Default Prediction (PROP-1006-HomeLoanDef)

## Project Overview
This project aims to predict loan defaults by analyzing historical customer data from Home Credit. The goal is to identify key factors influencing loan repayment behavior and build a predictive model to classify eligible vs. ineligible borrowers. The project includes:
- **Data Analysis**: Comprehensive exploration, cleaning, and feature engineering across 7 datasets.
- **Predictive Modeling**: Development and comparison of multiple machine learning models (XGBoost, LightGBM, CatBoost, Random Forest).
- **Model Deployment**: Selection of the best-performing model for production use.

### Key Tasks
1. **Data Analysis Report**: Assess data quality, handle missing values/outliers, and derive actionable insights.
2. **Predictive Model**: Identify customer segments eligible for loans using engineered features.
3. **Model Comparison**: Evaluate performance metrics (accuracy, recall, F1-score) to select the optimal model.
4. **Challenges Report**: Document technical hurdles and solutions during data processing and modeling.

---

## Dataset Description
The data comprises 7 interconnected CSV files with loan application history, credit bureau records, and transactional behavior:
| Dataset | Description | Rows | Key Features |
|---------|-------------|------|--------------|
| `application_train` | Main dataset with loan status (Target: `1`=Default, `0`=Non-Default) | 307,511 | `DAYS_BIRTH`, `AMT_CREDIT`, `EXT_SOURCE_1/2/3` |
| `bureau` | Clients' previous credits from other institutions | 1,716,428 | `DAYS_CREDIT`, `AMT_CREDIT_SUM` |
| `bureau_balance` | Monthly balances of previous credits | 27,299,925 | `MONTHS_BALANCE`, `STATUS` |
| `POS_CASH_balance` | Monthly POS/cash loan snapshots | 10,001,358 | `SK_DPD`, `CNT_INSTALMENT` |
| `credit_card_balance` | Monthly credit card balance history | 3,840,312 | `AMT_BALANCE`, `CREDIT_UTILIZATION_RATIO` |
| `previous_application` | Clients' prior loan applications | 1,670,214 | `AMT_ANNUITY`, `NAME_CONTRACT_TYPE` |
| `installments_payments` | Repayment history for previous credits | 13,605,401 | `AMT_INSTALMENT`, `DAYS_ENTRY_PAYMENT` |

**Download Link**: [PRCP-1006-HomeLoanDef.zip](https://d3libtxj3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip)

---

## Project Structure
```
├── data/                        # Raw and processed data
│   ├── 1.1 raw/                # Original CSV files
│   └── 1.2 processed/          # Cleaned and aggregated data (final_data.csv)
├── docs/                       # Problem statement and dataset documentation
├── notebooks/                  # Jupyter notebooks for analysis and modeling
│   └── PRCP-1006-HomeLoanDef.ipynb  # Main workflow
├── reports/                    # Final reports (Markdown/PDF)
├── results/                    # Analysis outputs, figures, and saved models
│   ├── 365csv pre-anlysis/     # Statistical summaries and visualizations
│   ├── figures/                # Charts and graphs
│   └── models/                 # Serialized models (e.g., final_xgb_model.pkl)
├── scripts/                    # Utility functions and helper scripts
└── requirements.txt            # Python dependencies
```

---

## Installation
1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/dhaneshbb/HomeLoanDef](https://github.com/dhaneshbb/HomeLoanDef).git
   cd HomeLoanDef
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Data**:  
   Extract [PRCP-1006-HomeLoanDef.zip](https://d3libtxj3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip) into `data/1.1 raw/`.

---

## Usage
1. **Run the Jupyter Notebook**:
   - Execute `notebooks/PRCP-1006-HomeLoanDef.ipynb` for end-to-end analysis and modeling.
   - Key steps:  
     - Data preprocessing (missing value imputation, outlier handling).  
     - Feature engineering (e.g., `CREDIT_UTILIZATION_RATIO`, `WEIGHTED_EXT_SOURCE`).  
     - Model training/evaluation and threshold optimization.  

2. **Generate Reports**:  
   - The notebook exports:
     - `reports/Final Report.md`: Detailed analysis and model comparison.  
     - `results/figures/`: Visualizations (e.g., ROC curves, feature importance plots).  

3. **Use the Trained Model**:  
   Load the saved XGBoost model for predictions:
   ```python
   import joblib
   model = joblib.load('results/models/final_xgb_model.pkl')
   ```

---

## Results
### Model Performance

| Model          | Accuracy | Recall | ROC AUC | Training Time (s) |
|----------------|----------|--------|---------|-------------------|
| **XGBoost**    | 83%      | 53%    | 0.785   | 98.10             |
| LightGBM       | 74%      | 68%    | 0.787   | 88.67             |
| CatBoost       | 79%      | 59%    | 0.776   | 147.94            |

**Optimal Threshold**: 0.60 (balances risk detection vs. false positives).

### Key Insights
- **Top Features**: External credit scores (`EXT_SOURCE`), demographic factors, and credit utilization ratios.  
- **Business Impact**: At a threshold of 0.60, the model detects **53% of defaulters** while minimizing false positives.  
- Full details: [Final Report](reports/Final%20Report.md).

---

## Challenges Faced

| Challenge               | Solution                          |
|-------------------------|-----------------------------------|
| High Memory Usage       | Downcasted numerical dtypes      |
| Class Imbalance (8% defaults) | Adjusted `scale_pos_weight` in XGBoost |
| Data Leakage            | Removed post-application features (e.g., `DAYS_LAST_PHONE_CHANGE`) |
| Multicollinearity       | VIF analysis + dropped correlated pairs (e.g., `AMT_CREDIT` vs `AMT_GOODS_PRICE`) |

[Full Challenges Report](reports/Final%20Report.md#challenges-faced-report)

---

## References
1. Dataset Source: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)  
2. Custom Library: [insightfulpy](https://github.com/dhaneshbb/insightfulpy)  
3. Technical Guides: Memory optimization, feature engineering, and model tuning best practices.  

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


This README provides a structured, overview of the project, linking to key files and highlighting critical insights. It balances technical detail with accessibility for collaborators.
