import pandas as pd
at = pd.read_csv(r'D:\datamites\HomeLoanDef\1. data\1.1 raw\application_train.csv')
bu = pd.read_csv(r'D:\datamites\HomeLoanDef\1. data\1.1 raw\bureau.csv')
bub = pd.read_csv(r'D:\datamites\HomeLoanDef\1. data\1.1 raw\bureau_balance.csv')
pc = pd.read_csv(r'D:\datamites\HomeLoanDef\1. data\1.1 raw\POS_CASH_balance.csv')
ccb = pd.read_csv(r'D:\datamites\HomeLoanDef\1. data\1.1 raw\credit_card_balance.csv')
pa = pd.read_csv(r'D:\datamites\HomeLoanDef\1. data\1.1 raw\previous_application.csv')
ip = pd.read_csv(r'D:\datamites\HomeLoanDef\1. data\1.1 raw\installments_payments.csv')

import pandas as pd
from tabulate import tabulate
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', None)

from insightfulpy.eda import *

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
#import dtale
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, pearsonr, spearmanr, ttest_ind, mannwhitneyu, shapiro
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, learning_curve
)
from sklearn.preprocessing import (
     LabelEncoder, OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample, compute_class_weight

import time
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, 
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import scikitplot as skplt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

import psutil
import os
import gc

def memory_usage():
    """Prints the current memory usage of the Python process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

def dataframe_memory_usage():
    """Prints the memory usage of all loaded Pandas DataFrames."""
    datasets = {'application_train': at, 'bureau': bu, 'bureau_balance': bub,
                'POS_CASH_balance': pc, 'credit_card_balance': ccb,
                'previous_application': pa, 'installments_payments': ip}
    for name, df in datasets.items():
        mem_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
        print(f"{name} Memory Usage: {mem_usage:.2f} MB")

def garbage_collection():
    """Performs garbage collection to free up memory."""
    gc.collect()
    memory_usage()

if __name__ == "__main__":
    memory_usage()

dataframe_memory_usage()

def single_value_columns(df):
    total_entries = df.shape[0]
    single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
    if not single_value_cols:
        print("No columns with a single unique value found.")
        return None
    single_value_summary = pd.DataFrame({
        'Index': range(len(single_value_cols)),
        'Column Name': single_value_cols,
        'Data Type': [df[col].dtype for col in single_value_cols],
        'Missing Percentage': [(df[col].isna().sum() / total_entries) * 100 for col in single_value_cols]
    })
    return single_value_summary

sns.set(style="whitegrid")
def plot_histograms(columns, data):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data[col], kde=True, bins=30, color='skyblue', stat='density')
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
def plot_boxplots(columns, data):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columns, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=data[col], color='lightgreen')
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()
def plot_scatter(columns_x, columns_y, data):
    plt.figure(figsize=(15, 10))
    for i, col_x in enumerate(columns_x):
        for j, col_y in enumerate(columns_y):
            plt.subplot(len(columns_x), len(columns_y), i*len(columns_y)+j+1)
            sns.scatterplot(x=data[col_x], y=data[col_y])
            plt.title(f'{col_x} vs {col_y}')
    plt.tight_layout()
    plt.show()
def plot_bar_plots(columns, data):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columns, 1):
        plt.subplot(3, 3, i)
        sns.countplot(x=data[col], palette='Set2')
        plt.title(f'Bar plot of {col}')
    plt.tight_layout()
    plt.show()
def plot_scatter_df(income_and_loan_cols, age_and_employment_cols, at):
    num_income_cols = len(income_and_loan_cols)
    num_age_cols = len(age_and_employment_cols)
    total_plots = num_income_cols * num_age_cols
    fig, axes = plt.subplots(num_income_cols, num_age_cols, figsize=(5 * num_age_cols, 5 * num_income_cols))
    if total_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    plot_index = 0
    for income_col in income_and_loan_cols:
        for age_col in age_and_employment_cols:
            ax = axes[plot_index]
            sns.scatterplot(x=at[age_col], y=at[income_col], alpha=0.5, ax=ax)
            ax.set_xlabel(age_col.replace('_', ' '))
            ax.set_ylabel(income_col.replace('_', ' '))
            ax.set_title(f"{income_col} vs {age_col}")
            plot_index += 1
    
    plt.tight_layout()
    plt.show()

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)  
        else: 
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
def reduce_mem_usagewithout_causing_cat(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe before optimization: {start_mem:.2f} MB")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in ['int64', 'int32', 'int16', 'int8']: 
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        elif col_type in ['float64', 'float32', 'float16']:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif col_type.name == 'category': 
            continue 
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB")
    print(f"Memory reduction: {100 * (start_mem - end_mem) / start_mem:.2f}%")
    return df
def optimize_data_types(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def chi_square_test(data, target_col='y'):
    chi_square_results = {}
    categorical_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()    
    for col in categorical_vars:
        if col == target_col:
            continue        
        contingency_table = pd.crosstab(data[col], data[target_col])        
        # Apply Chi-Square only if the table is larger than 2x2
        if contingency_table.shape != (2, 2):
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            if (expected >= 5).all():
                chi_square_results[col] = p_value
                print(f"- {col} (Chi-Square Test): p-value = {p_value:.4f}")    
    return chi_square_results

def fisher_exact_test(data, target_col='y'):
    fisher_exact_results = {}
    categorical_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()   
    for col in categorical_vars:
        if col == target_col:
            continue        
        contingency_table = pd.crosstab(data[col], data[target_col])        
        # Apply Fisherâ€™s Exact Test only for 2x2 tables
        if contingency_table.shape == (2, 2):
            _, p_value = fisher_exact(contingency_table)
            fisher_exact_results[col] = p_value
            print(f"- {col} (Fisher's Exact Test): p-value = {p_value:.4f}")    
    return fisher_exact_results

def spearman_correlation_with_target(data, non_normal_cols, target_col='TARGET', plot=True, table=True):
    if not pd.api.types.is_numeric_dtype(data[target_col]):
        raise ValueError(f"Target column '{target_col}' must be numeric. Please encode it before running this test.")
    correlation_results = {}
    for col in non_normal_cols:
        if col not in data.columns:
            continue 
        coef, p_value = spearmanr(data[col], data[target_col], nan_policy='omit')
        correlation_results[col] = {'Spearman Coefficient': coef, 'p-value': p_value}
    correlation_data = pd.DataFrame(correlation_results).T.dropna()
    correlation_data = correlation_data.sort_values('Spearman Coefficient', ascending=False)
    if target_col in correlation_data.index:
        correlation_data = correlation_data.drop(target_col)
    positive_corr = correlation_data[correlation_data['Spearman Coefficient'] > 0]
    negative_corr = correlation_data[correlation_data['Spearman Coefficient'] < 0]
    if table:
        print(f"\nPositive Spearman Correlations with Target ('{target_col}'):\n")
        for feature, stats in positive_corr.iterrows():
            print(f"- {feature}: Correlation={stats['Spearman Coefficient']:.4f}, p-value={stats['p-value']:.4f}")
        print(f"\nNegative Spearman Correlations with Target ('{target_col}'):\n")
        for feature, stats in negative_corr.iterrows():
            print(f"- {feature}: Correlation={stats['Spearman Coefficient']:.4f}, p-value={stats['p-value']:.4f}")
    if plot:
        plt.figure(figsize=(20, 8))  # Increase figure width to prevent label overlap
        sns.barplot(x=correlation_data.index, y='Spearman Coefficient', data=correlation_data, palette='coolwarm')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f"Spearman Correlation with Target ('{target_col}')", fontsize=16)
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Spearman Coefficient", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels for clarity
        plt.subplots_adjust(bottom=0.3)  # Add space below the plot for labels
        plt.tight_layout()
        plt.show()
    return correlation_data

def normality_test_with_skew_kurt(df):
    normal_cols = []
    not_normal_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if len(col_data) >= 3:
            if len(col_data) <= 5000:
                stat, p_value = shapiro(col_data)
                test_used = 'Shapiro-Wilk'
            else:
                stat, p_value = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                test_used = 'Kolmogorov-Smirnov'
            col_skewness = skew(col_data)
            col_kurtosis = kurtosis(col_data)
            result = {
                'Column': col,
                'Test': test_used,
                'Statistic': stat,
                'p_value': p_value,
                'Skewness': col_skewness,
                'Kurtosis': col_kurtosis
            }
            if p_value > 0.05:
                normal_cols.append(result)
            else:
                not_normal_cols.append(result)
    normal_df = (
        pd.DataFrame(normal_cols)
        .sort_values(by='Column') 
        if normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    not_normal_df = (
        pd.DataFrame(not_normal_cols)
        .sort_values(by='p_value', ascending=False)  # Sort by p-value descending (near normal to not normal)
        if not_normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    print("\nNormal Columns (p > 0.05):")
    display(normal_df)
    print("\nNot Normal Columns (p ≤ 0.05) - Sorted from Near Normal to Not Normal:")
    display(not_normal_df)
    return normal_df, not_normal_df

def spearman_correlation(data, non_normal_cols, exclude_target=None, multicollinearity_threshold=0.8):
    if non_normal_cols.empty:
        print("\nNo non-normally distributed numerical columns found. Exiting Spearman Correlation.")
        return
    selected_columns = non_normal_cols['Column'].tolist()
    if exclude_target and exclude_target in selected_columns and pd.api.types.is_numeric_dtype(data[exclude_target]):
        selected_columns.remove(exclude_target)
    spearman_corr_matrix = data[selected_columns].corr(method='spearman')
    multicollinear_pairs = []
    for i, col1 in enumerate(selected_columns):
        for col2 in selected_columns[i+1:]:
            coef = spearman_corr_matrix.loc[col1, col2]
            if abs(coef) > multicollinearity_threshold:
                multicollinear_pairs.append((col1, col2, coef))
    print("\nVariables Exhibiting Multicollinearity (|Correlation| > {:.2f}):".format(multicollinearity_threshold))
    if multicollinear_pairs:
        for col1, col2, coef in multicollinear_pairs:
            print(f"- {col1} & {col2}: Correlation={coef:.4f}")
    else:
        print("No multicollinear pairs found.")
    annot_matrix = spearman_corr_matrix.round(2).astype(str)
    num_vars = len(selected_columns)
    fig_size = max(min(24, num_vars * 1.2), 10)  # Keep reasonable bounds
    annot_font_size = max(min(10, 200 / num_vars), 6)  # Smaller font for more variables
    plt.figure(figsize=(fig_size, fig_size * 0.75))
    sns.heatmap(
        spearman_corr_matrix,
        annot=annot_matrix,
        fmt='',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": annot_font_size},
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Spearman Correlation Matrix', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()
    
def calculate_vif(data, exclude_target='TARGET', multicollinearity_threshold=5.0):
    # Select only numeric columns, exclude target, and drop rows with missing values
    numeric_data = data.select_dtypes(include=[np.number]).drop(columns=[exclude_target], errors='ignore').dropna()
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_data.columns
    vif_data['VIF'] = [variance_inflation_factor(numeric_data.values, i) 
                       for i in range(numeric_data.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    high_vif = vif_data[vif_data['VIF'] > multicollinearity_threshold]
    low_vif = vif_data[vif_data['VIF'] <= multicollinearity_threshold]
    print(f"\nVariance Inflation Factor (VIF) Scores (multicollinearity_threshold = {multicollinearity_threshold}):")
    print("\nFeatures with VIF > threshold (High Multicollinearity):")
    if not high_vif.empty:
        print(high_vif.to_string(index=False))
    else:
        print("None. No features exceed the VIF threshold.")
    print("\nFeatures with VIF <= threshold (Low/No Multicollinearity):")
    if not low_vif.empty:
        print(low_vif.to_string(index=False))
    else:
        print("None. All features exceed the VIF threshold.")
    return vif_data, high_vif['Feature'].tolist()

dataframes0 = {
    'at': at,
    'bu': bu,
    "bub": bub,
    'pc': pc,
    'ccb': ccb,
    'pa': pa,
    'ip': ip
}

linked_key(dataframes0)

relation_keys = ['SK_ID_CURR','NAME_CONTRACT_TYPE','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','NAME_TYPE_SUITE',
                'WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','SK_ID_BUREAU','MONTHS_BALANCE','SK_ID_PREV',
                'NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']

at = reduce_mem_usage(at)
pc = reduce_mem_usage(pc)
ccb = reduce_mem_usage(ccb)
pa = reduce_mem_usage(pa)
ip = reduce_mem_usage(ip)

bu = reduce_mem_usage(bu)
bub = reduce_mem_usage(bub)
bu_bub = pd.merge(bu, bub, on='SK_ID_BUREAU', how='left')

dataframes = {
    'at': at,
    "bu_bub": bu_bub,
    'pc': pc,
    'ccb': ccb,
    'pa': pa,
    'ip': ip
}

base_profile_df, linked_profiles_df = compare_df_columns('at', dataframes)
display_key_columns('at', dataframes)

base_profile_df

linked_profiles_df

at_linked_cols = [
    'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'AMT_ANNUITY', 
    'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'WEEKDAY_APPR_PROCESS_START', 
    'HOUR_APPR_PROCESS_START'
]

print(at.shape)
for idx, col in enumerate(at.columns):
        print(f"{idx}: {col}")

detect_mixed_data_types(at)

cat_high_cardinality(at)

at.dtypes.value_counts()

missing_inf_values(at)
print(f"\nNumber of duplicate rows: {at.duplicated().sum()}\n")
duplicates = at[at.duplicated()]
duplicates

columns_info("Dataset Overview", at)

analyze_data(at)

cat_analyze_and_plot(at, 'TARGET', subplot=False)

at_normal_df, at_not_normal_df = normality_test_with_skew_kurt(at)

num_summary(at)

cat_summary(at)

at_outlier_summary, at_non_outlier_summary = comp_num_analysis(at, outlier_df=True)
at_missing_summary, at_non_missing_summary = comp_num_analysis(at, missing_df=True)
at_cat_missing_summary, at_cat_non_missing_summary = comp_cat_analysis(at, missing_df=True)
at_cat_missing_summary = at_cat_missing_summary.sort_values(by="Unique_Count", ascending=False)
at_cat_non_missing_summary = at_cat_non_missing_summary.sort_values(by="Unique_Count", ascending=False)
print(at_cat_missing_summary.shape)
print(at_missing_summary.shape)
print(at_outlier_summary.shape)

at_outlier_summary

at_missing_summary

at_cat_missing_summary

at_negative_values = at.select_dtypes(include=[np.number]).lt(0).sum()
print("Columns with Negative Values:\n", at_negative_values[at_negative_values > 0])

income_and_loan_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
age_and_employment_cols = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'OWN_CAR_AGE']
family_and_social_cols = ['CNT_CHILDREN', 'CNT_FAM_MEMBERS']
credit_history_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
asset_and_contact_cols = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMAIL', 'FLAG_WORK_PHONE']
region_and_city_cols = ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REGION_POPULATION_RELATIVE']
loan_application_cols = ['HOUR_APPR_PROCESS_START', 'WEEKDAY_APPR_PROCESS_START']
building_and_housing_cols = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG']
social_circle_cols = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']

plot_histograms(income_and_loan_cols, at)
plot_boxplots(income_and_loan_cols, at)

plot_histograms(age_and_employment_cols, at)
plot_boxplots(age_and_employment_cols, at)

plot_histograms(family_and_social_cols, at)

plot_histograms(credit_history_cols, at)
plot_boxplots(credit_history_cols, at)

plot_bar_plots(asset_and_contact_cols, at)

region_and_city_cols = ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REGION_POPULATION_RELATIVE']
for col in region_and_city_cols:
    if at[col].dtype == 'float16':
        at[col] = at[col].astype('float32')

plot_bar_plots(region_and_city_cols, at)

plot_bar_plots(loan_application_cols, at)

plot_histograms(building_and_housing_cols, at)
plot_boxplots(building_and_housing_cols, at)

plot_histograms(social_circle_cols, at)
plot_boxplots(social_circle_cols, at)

plot_scatter_df(income_and_loan_cols, age_and_employment_cols, at)

plot_scatter_df(credit_history_cols, building_and_housing_cols, at)

leakage_cols = [
    'DAYS_LAST_PHONE_CHANGE',
    'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
    'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT'
]
credit_bureau_cols = [col for col in at.columns if col.startswith('AMT_REQ_CREDIT_BUREAU_')]
flag_document_cols = [col for col in at.columns if col.startswith('FLAG_DOCUMENT_')]
leakage_candidates = leakage_cols + credit_bureau_cols + flag_document_cols
leakage_candidates = [col for col in leakage_candidates if col in at.columns]
correlations = at[leakage_candidates + ['TARGET']].corr()['TARGET'].drop('TARGET')
print("\n **Potential Data Leakage Features (Correlation with TARGET)** \n")
print(correlations.abs().sort_values(ascending=False).to_markdown(tablefmt="pipe"))

high_leakage_cols = [
    'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'DAYS_LAST_PHONE_CHANGE',
    'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_DOCUMENT_3'
]
weak_leakage_cols = [col for col in at.columns if col.startswith('AMT_REQ_CREDIT_BUREAU_') or col.startswith('FLAG_DOCUMENT_')]
threshold = 0.02  
conditional_drops = correlations[correlations.abs() > threshold].index.tolist()
drop_cols = list(set(high_leakage_cols + conditional_drops))
at.drop(columns=drop_cols, inplace=True)
print(f" Dropped {len(drop_cols)} potential leakage columns.\n")
print(f" Removed: {drop_cols}")

categorical_columns = at.select_dtypes(include='category').columns
for col in categorical_columns:
    at[col] = at[col].replace(['XNA', 'Unknown'], np.nan)

at['CNT_CHILDREN'] = at['CNT_CHILDREN'].apply(lambda x: np.nan if x < 0 else x)

at['DAYS_EMPLOYED'] = at['DAYS_EMPLOYED'].replace(365243, np.nan).abs()

at['DAYS_BIRTH'] = at['DAYS_BIRTH'].abs()
at['DAYS_REGISTRATION'] = at['DAYS_REGISTRATION'].abs()
at['DAYS_ID_PUBLISH'] = at['DAYS_ID_PUBLISH'].abs()

at['DAYS_EMPLOYED'].fillna(0, inplace=True) 

columns_to_cap = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'OWN_CAR_AGE']
for col in columns_to_cap:
    lower_bound = at[col].quantile(0.01)
    upper_bound = at[col].quantile(0.99)
    at[col] = at[col].clip(lower=lower_bound, upper=upper_bound)

# Employment & Age Features
at['EMPLOYMENT_LENGTH'] = at['DAYS_EMPLOYED'] / -365
at['AGE'] = at['DAYS_BIRTH'] / -365
# Income Ratios (Fixed)
at['INCOME_TO_CREDIT_RATIO'] = at['AMT_INCOME_TOTAL'] / at['AMT_CREDIT']
at['INCOME_TO_ANNUITY_RATIO'] = at['AMT_INCOME_TOTAL'] / at['AMT_ANNUITY']
at['INCOME_TO_GOODS_RATIO'] = at['AMT_INCOME_TOTAL'] / at['AMT_GOODS_PRICE']
# Additional Ratios & Features (Fixed)
at['DAYS_EMPLOYED_PERC'] = at['DAYS_EMPLOYED'] / at['DAYS_BIRTH']
at['INCOME_PER_PERSON'] = at['AMT_INCOME_TOTAL'] / at['CNT_FAM_MEMBERS']
at['PAYMENT_RATE'] = at['AMT_ANNUITY'] / at['AMT_CREDIT']
at['DIFF_PRICE_CREDIT'] = (at['AMT_GOODS_PRICE'] != at['AMT_CREDIT']).astype(int)
# Weighted Average of External Sources
at['WEIGHTED_EXT_SOURCE'] = at[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mul([0.5, 0.3, 0.2]).sum(axis=1)
# Summation of FLAG_DOCUMENT Features
at['DOCUMENT_COUNT'] = at.filter(like='FLAG_DOCUMENT').sum(axis=1)

cat_missing_cols = [
    "NAME_FAMILY_STATUS", "CODE_GENDER", "NAME_TYPE_SUITE", "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE", "EMERGENCYSTATE_MODE", "HOUSETYPE_MODE", 
    "WALLSMATERIAL_MODE", "FONDKAPREMONT_MODE"
]
for col in cat_missing_cols:
    if at[col].dtype.name == "category":  
        at[col] = at[col].cat.add_categories("Missing") 
        at[col].fillna("Missing", inplace=True)  

chi= chi_square_test(at, target_col='TARGET')
fish= fisher_exact_test(at, target_col='TARGET')

binary_features = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
label_encoded_features = ["CODE_GENDER", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "NAME_EDUCATION_TYPE","EMERGENCYSTATE_MODE", "NAME_CONTRACT_TYPE"]
onehot_encoded_features = ["NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START", "NAME_INCOME_TYPE"]
frequency_encoded_features = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE", "WALLSMATERIAL_MODE"]
for col in binary_features:
    at[col] = at[col].map({'Y': 1, 'N': 0}) 
label_encoder = LabelEncoder()
for col in label_encoded_features:
    at[col] = label_encoder.fit_transform(at[col])
at = pd.get_dummies(at, columns=onehot_encoded_features, drop_first=True)
for col in frequency_encoded_features:
    freq_map = at[col].value_counts(normalize=True)  
    at[col] = at[col].map(freq_map) 

for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
    at[col] = at[col].astype(int)  
for col in ["OCCUPATION_TYPE", "ORGANIZATION_TYPE", "WALLSMATERIAL_MODE"]:
    at[col] = at[col].astype(float) 

at.replace([np.inf, -np.inf], np.nan, inplace=True)

at=optimize_data_types(at)

missing_inf_values(at)

at_normal_df, at_not_normal_df = normality_test_with_skew_kurt(at)
spearman_correlation(at, at_not_normal_df, exclude_target='TARGET', multicollinearity_threshold=0.8)

at_above_threshold, at_below_threshold = calculate_vif(at, exclude_target='TARGET', multicollinearity_threshold=8.0)

bu_bub = optimize_data_types(bu_bub)

base_profile_df, linked_profiles_df = compare_df_columns('bu_bub', dataframes)
display_key_columns('bu_bub', dataframes)

base_profile_df

linked_profiles_df

bu_linked_cols = ['SK_ID_CURR', 'AMT_ANNUITY', 'MONTHS_BALANCE']

print(bu_bub.shape)
for idx, col in enumerate(bu_bub.columns):
    print(f"{idx}: {col}")

detect_mixed_data_types(bu_bub)

cat_high_cardinality(bu_bub)

bu_bub.dtypes.value_counts()

missing_inf_values(bu_bub)
print(f"\nNumber of duplicate rows: {bu_bub.duplicated().sum()}\n")
duplicates = bu_bub[bu_bub.duplicated()]
duplicates

columns_info("Dataset Overview", bu_bub)

analyze_data(bu_bub)

credit_status_columns = ['CREDIT_ACTIVE', 'CREDIT_TYPE', 'STATUS']
credit_amount_columns = ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY']
credit_duration_columns = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG', 'DAYS_CREDIT_ENDDATE', 'DAYS_CREDIT_UPDATE']

# plot_bar_plots(credit_status_columns, bu_bub)

# plot_histograms(credit_amount_columns, bu_bub)
# plot_boxplots(credit_amount_columns, bu_bub)

# plot_histograms(credit_duration_columns, bu_bub)
# plot_boxplots(credit_duration_columns, bu_bub)

bu_bub_outlier_summary, bu_bub_non_outlier_summary = comp_num_analysis(bu_bub, outlier_df=True)
bu_bub_missing_summary, bu_bub_non_missing_summary = comp_num_analysis(bu_bub, missing_df=True)
bu_bub_cat_missing_summary, bu_bub_cat_non_missing_summary = comp_cat_analysis(bu_bub, missing_df=True)
bu_bub_cat_missing_summary = bu_bub_cat_missing_summary.sort_values(by="Unique_Count", ascending=False)
bu_bub_cat_non_missing_summary = bu_bub_cat_non_missing_summary.sort_values(by="Unique_Count", ascending=False)
print(bu_bub_cat_missing_summary.shape)
print(bu_bub_missing_summary.shape)
print(bu_bub_outlier_summary.shape)

bu_bub_negative_values = bu_bub.select_dtypes(include=[np.number]).lt(0).sum()
print("Columns with Negative Values:\n", bu_bub_negative_values[bu_bub_negative_values > 0])

num_summary(bu_bub)

cat_summary(bu_bub)

bu_bub_outlier_summary

bu_bub_missing_summary

bu_bub_cat_missing_summary

bu_bub_cat_non_missing_summary

high_leakage_cols = [
    'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM_DEBT',
    'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_DAY_OVERDUE'
]
bu_bub.drop(columns=high_leakage_cols, inplace=True)

bu_bub["AMT_CREDIT_SUM_LIMIT"] = bu_bub["AMT_CREDIT_SUM_LIMIT"].clip(lower=0)

bu_bub['CREDIT_DURATION'] = bu_bub['DAYS_CREDIT'] / 365  # Convert days to years
bu_bub['TOTAL_MONTHS'] = bu_bub.groupby('SK_ID_CURR')['MONTHS_BALANCE'].transform('count').replace(0, 1)

for col in ['CREDIT_ACTIVE', 'CREDIT_CURRENCY']:
    if bu_bub[col].dtype.name == 'category':  
        if 'missing' not in bu_bub[col].cat.categories:  
            bu_bub[col] = bu_bub[col].cat.add_categories('missing')
        bu_bub[col].fillna('missing', inplace=True)  

bu_bub['STATUS'].fillna('X', inplace=True)  
if bu_bub['CREDIT_TYPE'].dtype.name == 'category':  
    if 'Other' not in bu_bub['CREDIT_TYPE'].cat.categories: 
        bu_bub['CREDIT_TYPE'] = bu_bub['CREDIT_TYPE'].cat.add_categories('Other')
    bu_bub['CREDIT_TYPE'].fillna('Other', inplace=True) 

bu_bub = pd.get_dummies(bu_bub, columns=['CREDIT_ACTIVE', 'CREDIT_CURRENCY'], drop_first=True)

status_order = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'C': 6, 'X': 7}  

bu_bub['STATUS'] = bu_bub['STATUS'].map(status_order).astype('int8')  

frequency = bu_bub['CREDIT_TYPE'].value_counts(normalize=True)

bu_bub['CREDIT_TYPE'] = bu_bub['CREDIT_TYPE'].map(frequency)
bu_bub['CREDIT_TYPE'].fillna(frequency.min(), inplace=True)
bu_bub['CREDIT_TYPE'] = bu_bub['CREDIT_TYPE'].astype('float32')

bu_bub.replace([np.inf, -np.inf], np.nan, inplace=True)

bu_bub = optimize_data_types(bu_bub)

missing_inf_values(bu_bub)

# bu_normal_df, bu_not_normal_df = normality_test_with_skew_kurt(bu_bub)
# spearman_correlation(bu_bub, bu_not_normal_df, exclude_target='TARGET', multicollinearity_threshold=0.8)

available_columns = bu_bub.columns.tolist()
num_aggregations = {
 # Credit Utilization & Debt Exposure
    'AMT_CREDIT_SUM': 'sum',
    'DEBT_TO_CREDIT_RATIO': 'mean',
 # Payment Behavior
    'CNT_CREDIT_PROLONG': 'sum',
    'CREDIT_DURATION': 'mean',
    'TOTAL_MONTHS': 'mean',
# Time-Based Features
    'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
    'DAYS_CREDIT_UPDATE': ['mean'],
    'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
    'AMT_ANNUITY': ['max', 'mean'],
# Payment Behavior Over Time
    'STATUS': lambda x: (x == '1').sum(),  
}
num_aggregations = {k: v for k, v in num_aggregations.items() if k in available_columns}
cat_aggregations = {}
for col in [c for c in available_columns if 'CREDIT_ACTIVE_' in c]:
    cat_aggregations[col] = ['mean']
for col in [c for c in available_columns if 'CREDIT_CURRENCY_' in c]:
    cat_aggregations[col] = ['mean']
if 'STATUS' in available_columns:
    cat_aggregations['STATUS'] = ['mean']
if 'CREDIT_TYPE' in available_columns:
    cat_aggregations['CREDIT_TYPE'] = ['mean']
bu_bub_agg = bu_bub.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
bu_bub_agg.columns = ['BURO_' + '_'.join(col).upper() for col in bu_bub_agg.columns]
bu_bub_agg.reset_index(inplace=True)
gc.collect()
print("Final Aggregated Data Shape:", bu_bub_agg.shape)

at_bu_bub = at.merge(bu_bub_agg, on='SK_ID_CURR', how='left')
print("Final Merged Data Shape:", at_bu_bub.shape)

base_profile_df, linked_profiles_df = compare_df_columns('pa', dataframes)
display_key_columns('pa', dataframes)

base_profile_df

linked_profiles_df

linked_cols = [
    'SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_ANNUITY', 
     'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'NAME_CONTRACT_STATUS', 'NAME_TYPE_SUITE'
]

print(pa.shape)
for idx, col in enumerate(pa.columns):
    print(f"{idx}: {col}")

detect_mixed_data_types(pa)

cat_high_cardinality(pa)

pa.dtypes.value_counts()

missing_inf_values(pa)
show_missing(pa)
print(f"\nNumber of duplicate rows: {pa.duplicated().sum()}\n")
duplicates = pa[pa.duplicated()]
duplicates

columns_info("Dataset Overview", pa)

analyze_data(pa)

continuous_columns = ['AMT_CREDIT', 'AMT_APPLICATION', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
                      'RATE_DOWN_PAYMENT', 'DAYS_DECISION']
continuous_columns_for_boxplot = ['AMT_CREDIT', 'AMT_APPLICATION', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
x_columns = ['AMT_CREDIT', 'AMT_APPLICATION']
y_columns = ['AMT_ANNUITY', 'AMT_GOODS_PRICE']
categorical_columns = ['NAME_CONTRACT_TYPE', 'NAME_CLIENT_TYPE', 'NAME_PAYMENT_TYPE', 'NAME_GOODS_CATEGORY']

# plot_histograms(continuous_columns, pa)
# plot_boxplots(continuous_columns_for_boxplot, pa)
# plot_bar_plots(categorical_columns, pa)

plot_scatter(x_columns, y_columns, pa)

pa_normal_df, pa_not_normal_df = normality_test_with_skew_kurt(pa)

pa_negative_values = pa.select_dtypes(include=[np.number]).lt(0).sum()
print("Columns with Negative Values:\n", pa_negative_values[pa_negative_values > 0])

num_summary(pa)

cat_summary(pa)

pa_outlier_summary, pa_non_outlier_summary = comp_num_analysis(pa, outlier_df=True)
pa_missing_summary, pa_non_missing_summary = comp_num_analysis(pa, missing_df=True)
pa_cat_missing_summary, pa_cat_non_missing_summary = comp_cat_analysis(pa, missing_df=True)
pa_cat_missing_summary = pa_cat_missing_summary.sort_values(by="Unique_Count", ascending=False)
pa_cat_non_missing_summary = pa_cat_non_missing_summary.sort_values(by="Unique_Count", ascending=False)
print(pa_cat_missing_summary.shape)
print(pa_missing_summary.shape)
print(pa_outlier_summary.shape)

pa_outlier_summary

pa_missing_summary

pa_cat_missing_summary

pa_cat_non_missing_summary

high_leakage_cols = [
    'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
    'DAYS_LAST_DUE', 'DAYS_TERMINATION', 'NFLAG_INSURED_ON_APPROVAL'
]
pa.drop(columns=high_leakage_cols, inplace=True)

pa.replace(['XNA', 'XAP', 365243], np.nan, inplace=True)

pa['AMT_DOWN_PAYMENT'] = pa['AMT_DOWN_PAYMENT'].apply(lambda x: max(x, 0))
pa['RATE_DOWN_PAYMENT'] = pa['RATE_DOWN_PAYMENT'].apply(lambda x: max(x, 0))

columns_to_drop = ['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
                   'NAME_CASH_LOAN_PURPOSE','CODE_REJECT_REASON']
pa = pa.drop(columns=columns_to_drop)

def cap_extreme_values(df, columns):
    for col in columns:
        percentile_99 = df[col].quantile(0.99)
        df[col] = np.minimum(df[col], percentile_99)
    return df
columns_to_cap = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_APPLICATION', 'AMT_ANNUITY']
pa = cap_extreme_values(pa, columns_to_cap)

# Loan Amount Ratios
pa['INCOME_TO_CREDIT_RATIO'] = pa['AMT_CREDIT'] / pa['AMT_APPLICATION']
pa['DOWN_PAYMENT_RATIO'] = pa['AMT_DOWN_PAYMENT'] / pa['AMT_CREDIT']
pa['LOAN_TO_GOODS_PRICE'] = pa['AMT_CREDIT'] / pa['AMT_GOODS_PRICE']
# Interaction Features
pa['CREDIT_TO_ANNUITY_RATIO'] = pa['AMT_CREDIT'] / pa['AMT_ANNUITY']
pa['DOWN_PAYMENT_TO_CREDIT'] = pa['AMT_DOWN_PAYMENT'] / pa['AMT_CREDIT']
# Create flag features
pa['LAST_APPL_IN_DAY'] = pa['NFLAG_LAST_APPL_IN_DAY']
pa['PAYMENT_FREQUENCY'] = pa['CNT_PAYMENT']

from category_encoders import BinaryEncoder

categorical_cols = ['NAME_GOODS_CATEGORY', 'PRODUCT_COMBINATION', 'NAME_SELLER_INDUSTRY',
                    'NAME_TYPE_SUITE', 'NAME_PORTFOLIO', 'NAME_YIELD_GROUP',
                    'NAME_CONTRACT_TYPE', 'NAME_CLIENT_TYPE', 'NAME_PAYMENT_TYPE',
                    'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'WEEKDAY_APPR_PROCESS_START',
                    'NAME_CONTRACT_STATUS', 'FLAG_LAST_APPL_PER_CONTRACT']

for col in categorical_cols:
    if col in pa.columns:
        pa[col] = pa[col].astype(str).fillna('Missing')

one_hot_cols = ['NAME_CONTRACT_TYPE', 'NAME_CLIENT_TYPE', 'NAME_PAYMENT_TYPE',
                'NAME_PRODUCT_TYPE', 'NAME_CONTRACT_STATUS']
pa = pd.get_dummies(pa, columns=one_hot_cols, drop_first=True)

label_cols = ['NAME_PORTFOLIO', 'NAME_YIELD_GROUP', 'WEEKDAY_APPR_PROCESS_START']
for col in label_cols:
    if col in pa.columns:
        le = LabelEncoder()
        pa[col] = le.fit_transform(pa[col])
frequency_cols = ['NAME_GOODS_CATEGORY', 'PRODUCT_COMBINATION', 'NAME_SELLER_INDUSTRY',
                  'NAME_TYPE_SUITE', 'CHANNEL_TYPE']
for col in frequency_cols:
    freq = pa[col].value_counts(normalize=True)
    pa[col + '_FREQ'] = pa[col].map(freq)

binary_encoder = BinaryEncoder()
pa_binary = binary_encoder.fit_transform(pa[['FLAG_LAST_APPL_PER_CONTRACT']])
pa_binary.columns = [f'FLAG_LAST_APPL_PER_CONTRACT_{c}' for c in pa_binary.columns]
pa = pd.concat([pa, pa_binary], axis=1)
pa.drop(['FLAG_LAST_APPL_PER_CONTRACT'], axis=1, inplace=True)

pa.replace([np.inf, -np.inf], np.nan, inplace=True)

pa=optimize_data_types(pa)

missing_inf_values(pa)

pa_normal_df, pa_not_normal_df = normality_test_with_skew_kurt(pa)
spearman_correlation(pa, pa_not_normal_df, exclude_target='TARGET', multicollinearity_threshold=0.8)

available_columns = pa.columns.tolist()
num_aggregations = {}
for col in [
    'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_DOWN_PAYMENT',
    'CREDIT_TO_ANNUITY_RATIO', 'DOWN_PAYMENT_TO_CREDIT', 'LOAN_TO_GOODS_PRICE', 
    'DOWN_PAYMENT_RATIO'
]:
    if col in available_columns:
        num_aggregations[col] = ['min', 'max', 'mean', 'sum', 'var'] 
cat_aggregations = {}
for col in [c for c in available_columns if '_FREQ' in c]:  
    cat_aggregations[col] = ['mean']
for col in [c for c in available_columns if 'FLAG_LAST_APPL_PER_CONTRACT_' in c]:  
    cat_aggregations[col] = ['mean']
pa_agg = pa.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
pa_agg.columns = ['PA_' + '_'.join(col).upper() for col in pa_agg.columns]
pa_agg.reset_index(inplace=True)
print("Final Aggregated Data Shape:", pa_agg.shape)

final_data = at_bu_bub.merge(pa_agg, on='SK_ID_CURR', how='left')
print("Final Merged Data Shape:", final_data.shape)

base_profile_df, linked_profiles_df = compare_df_columns('pc', dataframes)
display_key_columns('pc', dataframes)

base_profile_df

linked_profiles_df

linked_cols = [
    'SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'NAME_CONTRACT_STATUS', 
    'SK_DPD', 'SK_DPD_DEF'
]

print(pc.shape)
for idx, col in enumerate(pc.columns):
    print(f"{idx}: {col}")

detect_mixed_data_types(pc)

cat_high_cardinality(pc)

pc.dtypes.value_counts()

missing_inf_values(pc)
show_missing(pc)
print(f"\nNumber of duplicate rows: {pc.duplicated().sum()}\n")
duplicates = pc[pc.duplicated()]
duplicates

columns_info("Dataset Overview", pc)

analyze_data(pc)

# kde_batches(pc, batch_num=1)
# box_plot_batches(pc, batch_num=1)

# plt.figure(figsize=(12, 6))
# sns.lineplot(x=pc['MONTHS_BALANCE'], y=pc['SK_DPD'], marker='o')
# plt.title('Overdue Days (SK_DPD) Over Time (Months Balance)')
# plt.xlabel('Months Balance')
# plt.ylabel('Overdue Days (SK_DPD)')
# plt.show()

cat_analyze_and_plot(pc, "NAME_CONTRACT_STATUS")

pc_normal_df, pc_not_normal_df = normality_test_with_skew_kurt(pc)

num_summary(pc)

cat_summary(pc)

pc_negative_values = pc.select_dtypes(include=[np.number]).lt(0).sum()
print("Columns with Negative Values:\n", pc_negative_values[pc_negative_values > 0])

pc_outlier_summary, pc_non_outlier_summary = comp_num_analysis(pc, outlier_df=True)
pc_missing_summary, pc_non_missing_summary = comp_num_analysis(pc, missing_df=True)
pc_cat_missing_summary, pc_cat_non_missing_summary = comp_cat_analysis(pc, missing_df=True)
pc_cat_missing_summary = pc_cat_missing_summary.sort_values(by="Unique_Count", ascending=False)
pc_cat_non_missing_summary = pc_cat_non_missing_summary.sort_values(by="Unique_Count", ascending=False)
print(pc_cat_missing_summary.shape)
print(pc_missing_summary.shape)
print(pc_outlier_summary.shape)

pc_outlier_summary

pc_missing_summary

pc_cat_missing_summary

pc_cat_non_missing_summary

pc['CNT_INSTALMENT'] = pc['CNT_INSTALMENT'].apply(lambda x: np.nan if x < 0 else x)
pc['CNT_INSTALMENT_FUTURE'] = pc['CNT_INSTALMENT_FUTURE'].apply(lambda x: np.nan if x < 0 else x)

pc['IS_OVERDUE'] = (pc['SK_DPD'] > 0).astype(int)  
pc['IS_DEFAULT'] = (pc['SK_DPD_DEF'] > 0).astype(int)  
pc['DAYS_TO_DEFAULT'] = pc['SK_DPD_DEF'] - pc['SK_DPD']
pc['INSTALLMENT_PROGRESS'] = pc['CNT_INSTALMENT'] / (pc['CNT_INSTALMENT'] + pc['CNT_INSTALMENT_FUTURE'])
pc['MONTHS_TO_NEXT_PAYMENT'] = pc['CNT_INSTALMENT_FUTURE']

if 'NAME_CONTRACT_STATUS' in pc.columns:
    pc['NAME_CONTRACT_STATUS_FREQ'] = pc['NAME_CONTRACT_STATUS'].map(
        pc['NAME_CONTRACT_STATUS'].value_counts(normalize=True)
    ).astype(float) 

bool_cols = ['IS_OVERDUE', 'IS_DEFAULT']
for col in bool_cols:
    pc[col] = pc[col].astype(int)

pc.replace([np.inf, -np.inf], np.nan, inplace=True)

pc=optimize_data_types(pc)

missing_inf_values(pc)

pc_normal_df, pc_not_normal_df = normality_test_with_skew_kurt(pc)
spearman_correlation(pc, pc_not_normal_df, exclude_target='TARGET', multicollinearity_threshold=0.8)

available_columns = pc.columns.tolist()
num_aggregations = {
    'SK_DPD': ['mean'], 
    'CNT_INSTALMENT': ['sum'],
    'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
    'INSTALLMENT_PROGRESS': ['min', 'max', 'mean'],
    'MONTHS_TO_NEXT_PAYMENT': ['min', 'max', 'mean'],
    'SK_DPD_DEF': ['min', 'max', 'mean', 'sum'],
    'DAYS_TO_DEFAULT': ['min', 'max', 'mean'],
    'NAME_CONTRACT_STATUS_FREQ': ['mean']
}
num_aggregations = {k: v for k, v in num_aggregations.items() if k in available_columns}
bool_aggregations = {col: ['mean'] for col in bool_cols if col in available_columns}
pc_agg = pc.groupby('SK_ID_CURR').agg({**num_aggregations, **bool_aggregations})
pc_agg.columns = ['PC_' + '_'.join(col).upper() for col in pc_agg.columns]
pc_agg.reset_index(inplace=True)
gc.collect()
print("Final Aggregated Data Shape:", pc_agg.shape)

final_data = final_data.merge(pc_agg, on='SK_ID_CURR', how='left')
print("Final Merged Data Shape:", final_data.shape)

base_profile_df, linked_profiles_df = compare_df_columns('ccb', dataframes)
display_key_columns('ccb', dataframes)

base_profile_df

linked_profiles_df

linked_cols = [
    'SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'NAME_CONTRACT_STATUS', 
    'SK_DPD', 'SK_DPD_DEF'
]

print(ccb.shape)
for idx, col in enumerate(ccb.columns):
    print(f"{idx}: {col}")

detect_mixed_data_types(ccb)

cat_high_cardinality(ccb)

ccb.dtypes.value_counts()

missing_inf_values(ccb)
show_missing(ccb)
print(f"\nNumber of duplicate rows: {ccb.duplicated().sum()}\n")
duplicates = ccb[ccb.duplicated()]
duplicates

columns_info("Dataset Overview", ccb)

analyze_data(ccb)

# payment and balance 
financial_columns = ['AMT_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']
# plot_histograms(financial_columns, ccb)
# plot_boxplots(financial_columns, ccb)

# Credit Utilization and Payment Ratios
ccb['CREDIT_UTILIZATION_RATIO'] = ccb['AMT_BALANCE'] / ccb['AMT_CREDIT_LIMIT_ACTUAL']
ccb['PAYMENT_RATIO'] = ccb['AMT_PAYMENT_TOTAL_CURRENT'] / ccb['AMT_BALANCE']
plot_scatter(['CREDIT_UTILIZATION_RATIO'], ['PAYMENT_RATIO'], ccb)

plot_scatter(['SK_DPD'], ['SK_DPD_DEF'], ccb)

plt.figure(figsize=(12, 6))
sns.lineplot(x=ccb['MONTHS_BALANCE'], y=ccb['SK_DPD'], marker='o')
plt.title('Overdue Days (SK_DPD) Over Time (Months Balance)')
plt.xlabel('Months Balance')
plt.ylabel('Overdue Days (SK_DPD)')
plt.show()

cat_analyze_and_plot(ccb, "NAME_CONTRACT_STATUS")

ccb_normal_df, ccb_not_normal_df = normality_test_with_skew_kurt(ccb)

num_summary(ccb)

cat_summary(ccb)

ccb_negative_values = ccb.select_dtypes(include=[np.number]).lt(0).sum()
print("Columns with Negative Values:\n", ccb_negative_values[ccb_negative_values > 0])

ccb_outlier_summary, ccb_non_outlier_summary = comp_num_analysis(ccb, outlier_df=True)
ccb_missing_summary, ccb_non_missing_summary = comp_num_analysis(ccb, missing_df=True)
ccb_cat_missing_summary, ccb_cat_non_missing_summary = comp_cat_analysis(ccb, missing_df=True)
ccb_cat_missing_summary = ccb_cat_missing_summary.sort_values(by="Unique_Count", ascending=False)
ccb_cat_non_missing_summary = ccb_cat_non_missing_summary.sort_values(by="Unique_Count", ascending=False)
print(ccb_cat_missing_summary.shape)
print(ccb_missing_summary.shape)
print(ccb_outlier_summary.shape)

ccb_outlier_summary

ccb_missing_summary

ccb_cat_missing_summary

ccb_cat_non_missing_summary

def cap_outliers(df, columns, cap_value):
    for col in columns:
        df[col] = df[col].apply(lambda x: min(x, cap_value))  # Cap values at the given threshold
    return df
columns_to_cap = ['CNT_INSTALMENT_MATURE_CUM', 'CNT_DRAWINGS_POS_CURRENT', 'AMT_BALANCE']
cap_values = {'CNT_INSTALMENT_MATURE_CUM': 120, 'CNT_DRAWINGS_POS_CURRENT': 100, 'AMT_BALANCE': 500000}
for col in columns_to_cap:
    ccb = cap_outliers(ccb, [col], cap_values[col])

columns_to_clean = [
    'AMT_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT',
    'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 
    'CNT_INSTALMENT_MATURE_CUM', 'CNT_DRAWINGS_POS_CURRENT'
]
for col in columns_to_clean:
    ccb[col] = ccb[col].apply(lambda x: np.nan if x < 0 else x)

if 'NAME_CONTRACT_STATUS' in ccb.columns:
    ccb['NAME_CONTRACT_STATUS_FREQ'] = ccb['NAME_CONTRACT_STATUS'].map(
        ccb['NAME_CONTRACT_STATUS'].value_counts(normalize=True)
    ).astype(float) 

ccb.replace([np.inf, -np.inf], np.nan, inplace=True)

ccb=optimize_data_types(ccb)

missing_inf_values(ccb)

ccb_normal_df, ccb_not_normal_df = normality_test_with_skew_kurt(ccb)
spearman_correlation(ccb, ccb_not_normal_df, exclude_target='TARGET', multicollinearity_threshold=0.8)

available_columns = ccb.columns.tolist()
num_aggregations = {
    'AMT_BALANCE': ['min', 'max', 'mean', 'sum'],
    'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum'],
    'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
    'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
    'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
    'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
    'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean', 'sum'],
    'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
    'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
    'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'sum'],
    'AMT_RECIVABLE': ['min', 'max', 'mean', 'sum'],
    'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
    'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
    'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
    'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
    'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'sum'],
    'CREDIT_UTILIZATION_RATIO': ['min', 'max', 'mean', 'sum'],
    'PAYMENT_RATIO': ['min', 'max', 'mean', 'sum'],
    'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
    'SK_DPD': ['min', 'max', 'mean', 'sum'],
    'SK_DPD_DEF': ['min', 'max', 'mean', 'sum'],
    'NAME_CONTRACT_STATUS_FREQ': ['mean']  
}
num_aggregations = {k: v for k, v in num_aggregations.items() if k in available_columns}
ccb_agg = ccb.groupby('SK_ID_CURR').agg(num_aggregations)
ccb_agg.columns = ['CCB_' + '_'.join(col).upper() for col in ccb_agg.columns]
ccb_agg.reset_index(inplace=True)
gc.collect()
print("Final Aggregated Data Shape:", ccb_agg.shape)

final_data = final_data.merge(ccb_agg, on='SK_ID_CURR', how='left')
print("Final Merged Data Shape:", final_data.shape)

base_profile_df, linked_profiles_df = compare_df_columns('ip', dataframes)
display_key_columns('ip', dataframes)

base_profile_df

linked_profiles_df

print(ip.shape)
for idx, col in enumerate(ip.columns):
    print(f"{idx}: {col}")

detect_mixed_data_types(ip)

cat_high_cardinality(ip)

ip.dtypes.value_counts()

missing_inf_values(ip)
show_missing(ip)
print(f"\nNumber of duplicate rows: {ip.duplicated().sum()}\n")
duplicates = ip[ip.duplicated()]
duplicates

ip = ip.drop_duplicates()

columns_info("Dataset Overview", ip)

analyze_data(ip)

# kde_batches(ip, batch_num=1)
# box_plot_batches(ip, batch_num=1)

ip_negative_values = ip.select_dtypes(include=[np.number]).lt(0).sum()
print("Columns with Negative Values:\n", ip_negative_values[ip_negative_values > 0])

num_summary(ip)

ip_outlier_summary, ip_non_outlier_summary = comp_num_analysis(ip, outlier_df=True)
ip_outlier_summary

ip_missing_summary, ip_non_missing_summary = comp_num_analysis(ip, missing_df=True)
ip_missing_summary

def cap_outliers(df, columns, percentile=0.99):
    for col in columns:
        cap_value = df[col].quantile(percentile)
        df[col] = np.minimum(df[col], cap_value)
    return df

columns_to_cap = ['AMT_INSTALMENT', 'AMT_PAYMENT']
ip = cap_outliers(ip, columns_to_cap)

ip['PAYMENT_TO_INSTALMENT_RATIO'] = ip['AMT_PAYMENT'] / ip['AMT_INSTALMENT']
ip['TIME_TO_PAYMENT'] = ip['DAYS_ENTRY_PAYMENT'] - ip['DAYS_INSTALMENT']
ip['IS_LATE_PAYMENT'] = ip['TIME_TO_PAYMENT'] > 0
ip['IS_LATE_PAYMENT'] = ip['IS_LATE_PAYMENT'].astype(int) 

ip.replace([np.inf, -np.inf], np.nan, inplace=True)

ip=optimize_data_types(ip)

missing_inf_values(ip)

ip_normal_df, ip_not_normal_df = normality_test_with_skew_kurt(ip)
spearman_correlation(ip, ip_not_normal_df, exclude_target='TARGET', multicollinearity_threshold=0.8)

available_columns = ip.columns.tolist()
num_aggregations = {}
for col in [
    'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 
    'DAYS_ENTRY_PAYMENT', 'TIME_TO_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT', 
    'PAYMENT_TO_INSTALMENT_RATIO'
]:
    if col in available_columns:
        num_aggregations[col] = ['min', 'max', 'mean', 'sum']
bool_aggregations = {'IS_LATE_PAYMENT': ['mean']}
ip_agg = ip.groupby('SK_ID_CURR').agg({**num_aggregations, **bool_aggregations})
ip_agg.columns = ['IP_' + '_'.join(col).upper() for col in ip_agg.columns]
ip_agg.reset_index(inplace=True)
gc.collect()
print("Final Aggregated Data Shape:", ip_agg.shape)

final_data = final_data.merge(ip_agg, on='SK_ID_CURR', how='left')
print("Final Merged Data Shape:", final_data.shape)

final_data.to_csv("final_data.csv", index=False)

data = reduce_mem_usagewithout_causing_cat(final_data)

data_duplicates = data.duplicated(subset=['SK_ID_CURR'], keep=False)
if data_duplicates.any():
    print(f"Removing {data_duplicates.sum()} duplicate rows...")
    data = data.loc[~data_duplicates]

detect_mixed_data_types(data)

data.shape

data.dtypes.value_counts()

linked_key_features = [
    'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'NAME_TYPE_SUITE', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 
    'SK_ID_BUREAU', 'MONTHS_BALANCE', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS', 'SK_DPD', 'SK_DPD_DEF'
]
available_linked_keys = [col for col in linked_key_features if col in data.columns]
print("Linked Key Features Found in data:")
print(available_linked_keys)

drop_keys = {'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'WEEKDAY_APPR_PROCESS_START', 
             'HOUR_APPR_PROCESS_START', 'MONTHS_BALANCE'}
data = data.drop(columns=drop_keys, errors='ignore')
print("Remaining Features Shape:", data.shape)

# data_outlier_summary, data_non_outlier_summary = comp_num_analysis(data, outlier_df=True)
# data_outlier_summary.to_csv("data_outlier_summary.csv", index=False)
# data_missing_summary, data_non_missing_summary = comp_num_analysis(data, missing_df=True)
# data_missing_summary.to_csv("data_missing_summary.csv", index=False)
# data_cat_missing_summary, data_cat_non_missing_summary = comp_cat_analysis(data, missing_df=True)
# data_cat_missing_summary.to_csv("data_cat_missing_summary.csv", index=False)
# print(data_cat_missing_summary.shape)
# print(data_missing_summary.shape)
# print(data_outlier_summary.shape)

results = missing_inf_values(data, missing=True,df_table=True)
print(results.sort_values(by="Missing Percentage", ascending=False).to_markdown(tablefmt="pipe"))

drop_cols = data.columns[data.isnull().mean() >= 0.75]
data.drop(columns=drop_cols, inplace=True, errors='ignore')
print(f"Removed {len(drop_cols)} columns with ≥75% missing values.")
print("Final Data Shape:", data.shape)

outliers_ckeck = [
    'AGE', 'EMPLOYMENT_LENGTH', 'DAYS_EMPLOYED', 'DAYS_EMPLOYED_PERC',
    'CREDIT_UTILIZATION_RATIO', 'PAYMENT_RATIO', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
    'INCOME_TO_CREDIT_RATIO', 'AMT_GOODS_PRICE', 'AMT_ANNUITY'
]
existing_cols = [col for col in outliers_ckeck if col in data.columns]
outlier_counts = {}
total_rows = len(data)
for col in existing_cols:
    lower, upper = data[col].quantile([0.01, 0.99])
    outlier_count = ((data[col] < lower) | (data[col] > upper)).sum()
    outlier_counts[col] = [outlier_count, (outlier_count / total_rows) * 100]
outlier_df = pd.DataFrame(outlier_counts, index=['Outlier Count', 'Outlier Percentage']).T
print(outlier_df)

cap_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'INCOME_TO_CREDIT_RATIO', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']
for col in cap_cols:
    lower, upper = data[col].quantile([0.01, 0.99])
    data[col] = data[col].clip(lower, upper)

cat_high_cardinality(data)

columns_info("Dataset Overview", data)

single_value_df = single_value_columns(data)
display(single_value_df)

single_value_cols = [col for col in data.columns if data[col].nunique() == 1]
data.drop(columns=single_value_cols, inplace=True)
print("Dropped Columns:", single_value_cols)

data_negative_values = final_data.select_dtypes(include=[np.number]).lt(0).sum()
data_negative_values = data_negative_values[data_negative_values > 0].sort_values(ascending=False)
print("Columns with Negative Values (Sorted):\n", data_negative_values)

data['AGE'] = data['AGE'].abs()
data['EMPLOYMENT_LENGTH'] = data['EMPLOYMENT_LENGTH'].abs()
for col in [
    'CCB_CREDIT_UTILIZATION_RATIO_MIN', 'CCB_CREDIT_UTILIZATION_RATIO_MEAN', 'CCB_CREDIT_UTILIZATION_RATIO_SUM',
    'CCB_PAYMENT_RATIO_SUM'
]:
    data[col] = data[col].clip(lower=0)  
data['BURO_CREDIT_DURATION_MEAN'] = data['BURO_CREDIT_DURATION_MEAN'].abs()

inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
print(f"Total Inf values: {inf_counts}")

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data=optimize_data_types(data)
missing=missing_inf_values(data,missing=True,df_table=True)
print(missing.sort_values(by="Missing Percentage", ascending=False).to_markdown(tablefmt="pipe"))

correlation_matrix = data.drop(columns=['TARGET'], errors='ignore').corr()
threshold = 0.85
high_corr_pairs = (
    correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    .stack() 
    .reset_index()
)
high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
high_corr_pairs = high_corr_pairs[high_corr_pairs['Correlation'].abs() > threshold]
print(high_corr_pairs.sort_values(by="Correlation", ascending=False).to_markdown(tablefmt="pipe"))

cat_analyze_and_plot(data, 'TARGET', subplot=False)

X = data.drop(columns=["TARGET"])
y = data["TARGET"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print("scale_pos_weight:", scale_pos_weight)

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance.sort_values(by="Importance", ascending=False).to_markdown(tablefmt="pipe"))

zero_importance_features = feature_importance[feature_importance["Importance"] == 0]["Feature"].tolist()
print(f"Before dropping, X_train shape: {X_train.shape}")
print(f"Before dropping, X_test shape: {X_test.shape}")
X_train.drop(columns=zero_importance_features, errors="ignore", inplace=True)
X_test.drop(columns=zero_importance_features, errors="ignore", inplace=True)
print("Final X_train shape after feature selection:", X_train.shape)
print("Final X_test shape after feature selection:", X_test.shape)

low_importance_threshold = 0.002  
low_importance_features = feature_importance[feature_importance["Importance"] <= low_importance_threshold]
print(low_importance_features.sort_values(by="Importance", ascending=False).to_markdown(tablefmt="pipe"))

low_importance_features = feature_importance[feature_importance["Importance"] < 0.002]["Feature"].tolist()
print(f"Before dropping, X_train shape: {X_train.shape}")
print(f"Before dropping, X_test shape: {X_test.shape}")
X_train.drop(columns=low_importance_features, errors='ignore', inplace=True)
X_test.drop(columns=low_importance_features, errors='ignore', inplace=True)
print(f"Final X_train shape after dropping low-importance features: {X_train.shape}")
print(f"Final X_test shape after dropping low-importance features: {X_test.shape}")

correlation_threshold = 0.9  
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [(col, idx, upper.loc[col, idx]) 
                      for col in upper.columns 
                      for idx in upper.index 
                      if upper.loc[col, idx] > correlation_threshold]
high_corr_df = pd.DataFrame(high_corr_features, columns=["Feature 1", "Feature 2", "Correlation"])
print(high_corr_df.sort_values(by="Correlation", ascending=False).to_markdown(tablefmt="pipe"))

def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None  
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1_metric = f1_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)
    cv_acc = np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy"))
    overfit = acc_train - acc
    return {
        "Model Name": type(model).__name__,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1_metric,
        "ROC AUC Score": roc_auc if roc_auc else "Not Available",
        "Cross-Validation Accuracy": cv_acc,
        "Training Accuracy": acc_train,
        "Overfit": overfit,
        "Training Time (seconds)": round(training_time, 4)
    }

def cross_validation_analysis_table(model, X_train, y_train, cv_folds=5, scoring_metric="f1"):
    strat_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=strat_kfold, scoring=scoring_metric)
    cv_results_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(cv_folds)],
        "F1-Score": scores
    })
    cv_results_df.loc["Mean"] = ["Mean", np.mean(scores)]
    cv_results_df.loc["Std"] = ["Standard Deviation", np.std(scores)]
    return cv_results_df
    
def threshold_analysis(model, X_test, y_test, thresholds=np.arange(0.1, 1.0, 0.1)):
    y_probs = model.predict_proba(X_test)[:, 1]  
    results = []
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results.append({
            "Threshold": round(threshold, 2),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Accuracy": round(accuracy, 4),
            "True Negatives (TN)": tn,
            "False Positives (FP)": fp,
            "False Negatives (FN)": fn,
            "True Positives (TP)": tp
        })
    df_results = pd.DataFrame(results)
    best_threshold = df_results.loc[df_results["F1-Score"].idxmax(), "Threshold"]
    print(f" Best Decision Threshold (Max F1-Score): {best_threshold:.2f}")
    return df_results, best_threshold

def plot_all_evaluation_metrics(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
    y_pred_default = (y_probs >= 0.6).astype(int)
    cm = confusion_matrix(y_test, y_pred_default)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes[0, 0].plot(prob_pred, prob_true, marker="o", label="Calibration")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    axes[0, 0].set_title("Calibration Curve")
    axes[0, 0].set_xlabel("Predicted Probability")
    axes[0, 0].set_ylabel("Actual Probability")
    axes[0, 0].legend()
    axes[0, 0].grid()
    skplt.metrics.plot_cumulative_gain(y_test, model.predict_proba(X_test), ax=axes[0, 1])
    axes[0, 1].set_title("Cumulative Gains Curve")
    y_probs_1 = y_probs[y_test == 1]  # Positive class
    y_probs_0 = y_probs[y_test == 0]  # Negative class
    axes[0, 2].hist(y_probs_1, bins=50, alpha=0.5, label="y=1")
    axes[0, 2].hist(y_probs_0, bins=50, alpha=0.5, label="y=0")
    axes[0, 2].set_title("Kolmogorov-Smirnov (KS) Statistic")
    axes[0, 2].set_xlabel("Predicted Probability")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].legend()
    axes[0, 2].grid()
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = np.linspace(0.6, 0.9, 10)
    val_scores = np.linspace(0.55, 0.85, 10)
    axes[1, 0].plot(train_sizes, train_scores, label="Train Score")
    axes[1, 0].plot(train_sizes, val_scores, label="Validation Score")
    axes[1, 0].set_title("Learning Curve (Simulated)")
    axes[1, 0].set_xlabel("Training Size")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid()
    skplt.metrics.plot_lift_curve(y_test, model.predict_proba(X_test), ax=axes[1, 1])
    axes[1, 1].set_title("Lift Curve")
    axes[1, 2].plot(thresholds, precision[:-1], "b--", label="Precision")
    axes[1, 2].plot(thresholds, recall[:-1], "r-", label="Recall")
    axes[1, 2].set_title("Precision-Recall Curve")
    axes[1, 2].set_xlabel("Threshold")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].legend()
    axes[1, 2].grid()
    axes[2, 0].plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    axes[2, 0].plot([0, 1], [0, 1], linestyle="--", color="black")
    axes[2, 0].set_title("ROC Curve")
    axes[2, 0].set_xlabel("False Positive Rate")
    axes[2, 0].set_ylabel("True Positive Rate")
    axes[2, 0].legend()
    axes[2, 0].grid()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[2, 1], cmap="Blues")
    axes[2, 1].set_title("Confusion Matrix")
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
    disp_norm.plot(ax=axes[2, 2], cmap="Blues")
    axes[2, 2].set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.show()

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print("scale_pos_weight:", scale_pos_weight)

print("X_train Data Types:\n", X_train.dtypes.value_counts()) 
print("X_test Data Types:\n", X_test.dtypes.value_counts())    

if isinstance(y_train, pd.Series):
    y_train = y_train.values
if isinstance(y_test, pd.Series):
    y_test = y_test.values
    
print("y_train shape:", y_train.shape) 
print("y_test shape:", y_test.shape)  

# Remove special characters from column names for lightgbm con't support for json
X_train.columns = X_train.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
X_test.columns = X_test.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

base_models = {
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        eval_metric="logloss",
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        random_state=42
    ),
    "CatBoost": CatBoostClassifier(
        iterations=100,
        scale_pos_weight=scale_pos_weight, 
        verbose=0, 
        random_seed=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
}

results = []
for name, model in base_models.items():
    print(f"Training {name} model...")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    metrics["Model Name"] = name
    results.append(metrics)
base_results = pd.DataFrame(results)
base_results.sort_values(by="ROC AUC Score", ascending=False, inplace=True)

base_results

if isinstance(y_train, np.ndarray):
    y_train = pd.Series(y_train)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print("scale_pos_weight:", scale_pos_weight)

# import optuna
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# def optimize_hyperparameters(model_name, X_train, y_train, n_trials=10):
#     scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
#     def objective(trial):
#         if model_name == 'XGBoost':
#             model = xgb.XGBClassifier(
#                 n_estimators=trial.suggest_int('n_estimators', 100, 1000),
#                 max_depth=trial.suggest_int('max_depth', 3, 12),
#                 learning_rate=trial.suggest_loguniform('learning_rate', 0.01, 0.3),
#                 subsample=trial.suggest_float('subsample', 0.5, 1.0),
#                 colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
#                 reg_lambda=trial.suggest_loguniform('reg_lambda', 1e-3, 10),
#                 reg_alpha=trial.suggest_loguniform('reg_alpha', 1e-3, 10),
#                 scale_pos_weight=scale_pos_weight,
#                 use_label_encoder=False,
#                 verbosity=0
#             )
#         elif model_name == 'LightGBM':
#             model = lgb.LGBMClassifier(
#                 n_estimators=trial.suggest_int('n_estimators', 100, 1000),
#                 max_depth=trial.suggest_int('max_depth', 3, 12),
#                 learning_rate=trial.suggest_loguniform('learning_rate', 0.01, 0.3),
#                 num_leaves=trial.suggest_int('num_leaves', 20, 150),
#                 subsample=trial.suggest_float('subsample', 0.5, 1.0),
#                 colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
#                 reg_lambda=trial.suggest_loguniform('reg_lambda', 1e-3, 10),
#                 scale_pos_weight=scale_pos_weight
#             )
#         elif model_name == 'CatBoost':
#             model = cb.CatBoostClassifier(
#                 iterations=trial.suggest_int('iterations', 100, 1000),
#                 depth=trial.suggest_int('depth', 3, 8),
#                 learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
#                 l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10),
#                 border_count=trial.suggest_int('border_count', 32, 128),
#                 random_strength=trial.suggest_float('random_strength', 0.1, 5.0),
#                 scale_pos_weight=scale_pos_weight,
#                 verbose=0,
#                 eval_metric='AUC'
#             )
#         cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#         auc_score = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
#         return auc_score
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=n_trials)
#     return study.best_params

# best_params_xgb = optimize_hyperparameters('XGBoost', X_train, y_train, n_trials=10)
# best_params_lgb = optimize_hyperparameters('LightGBM', X_train, y_train, n_trials=10)
# best_params_cb = optimize_hyperparameters('CatBoost', X_train, y_train, n_trials=10)
# print("Best Hyperparameters for XGBoost:", best_params_xgb)
# print("Best Hyperparameters for LightGBM:", best_params_lgb)
# print("Best Hyperparameters for CatBoost:", best_params_cb)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
optimized_params = {
    "XGBoost": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.03209810701885085,
        "subsample": 0.7090197575717343,
        "colsample_bytree": 0.6728971506385499,
        "reg_lambda": 6.535290526211357,
        "reg_alpha": 0.30199872992709925,
        "scale_pos_weight": scale_pos_weight, 
        "random_state": 42
    },
    "LightGBM": {
        "n_estimators": 700,
        "max_depth": 11,
        "learning_rate": 0.023541696374895275,
        "subsample": 0.632474929751148,
        "colsample_bytree": 0.6090953955288042,
        "reg_lambda": 0.0010042814914133617,
        "reg_alpha": 0.004264944346604339,
        "scale_pos_weight": scale_pos_weight, 
        "random_state": 42
    },
    "CatBoost": {
        "iterations": 452,
        "depth": 8,
        "learning_rate": 0.12609528877384452,
        "l2_leaf_reg": 4.412360416869782,
        "border_count": 77,
        "random_strength": 3.3697320457883957,
        "bagging_temperature": 0.10482861514016462,
        "class_weights": [1, scale_pos_weight],  
        "verbose": 0,
        "random_seed": 42
    }
}

tune_models = {
    "XGBoost": xgb.XGBClassifier(**optimized_params["XGBoost"]),
    "LightGBM": lgb.LGBMClassifier(**optimized_params["LightGBM"]),
    "CatBoost": cb.CatBoostClassifier(**optimized_params["CatBoost"])
}

results = []
for name, model in tune_models.items():
    print(f"Training {name}...")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    results.append(metrics)
tune_result = pd.DataFrame(results)

tune_result

base_results

tune_result

def show_default_feature_importance(models, X_train):
    feature_importance_df = pd.DataFrame({"Feature": X_train.columns})
    for model_name, model in tune_models.items():
        if hasattr(model, "feature_importances_"):
            feature_importance_df[model_name] = model.feature_importances_
        else:
            print(f" Feature importance not available for {model_name}")
    feature_importance_df = feature_importance_df.sort_values(by="XGBoost", ascending=False)
    print("\n Default Feature Importance Across Models:")
    print(feature_importance_df.to_markdown(tablefmt="pipe", index=False))
for name, model in tune_models.items():
    model.fit(X_train, y_train)
show_default_feature_importance(tune_models, X_train)

final_model = XGBClassifier(
    n_estimators= 400,
    max_depth= 6,
    learning_rate= 0.03209810701885085,
    subsample= 0.7090197575717343,
    colsample_bytree= 0.6728971506385499,
    reg_lambda= 6.535290526211357,
    reg_alpha= 0.30199872992709925,
    scale_pos_weight= scale_pos_weight, 
    random_state= 42
)
final_model.fit(X_train, y_train)

cv_results_table = cross_validation_analysis_table(final_model, X_train, y_train, cv_folds=5, scoring_metric="f1")
cv_results_table

df_threshold_results, best_threshold = threshold_analysis(final_model, X_test, y_test)
df_threshold_results = df_threshold_results.sort_values(by="F1-Score", ascending=False)
print("\n=== Threshold Comparison Table ===")
df_threshold_results

from sklearn.metrics import classification_report, confusion_matrix
if not hasattr(final_model, "predict_proba"):
    raise ValueError(" Model does not support probability predictions.")
y_probs = final_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_probs >= 0.60).astype(int)
print("\n xgb Classification Report (Final Adjusted Threshold - 0.60):")
print(classification_report(y_test, y_pred_final))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
print(f"\nConfusion Matrix at Threshold 0.60:")
print(f"True Negatives (TN): {tn}, False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}, True Positives (TP): {tp}")

plot_all_evaluation_metrics(final_model, X_test, y_test)

import joblib
joblib.dump(final_model, "final_xgb_model.pkl")
print("Model saved successfully!")

final_model = joblib.load("final_xgb_model.pkl")
y_probs = final_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_probs >= 0.60).astype(int)
print("\n xgb Classification Report (Final Adjusted Threshold - 0.60):")
print(classification_report(y_test, y_pred_final))

import types
import inspect
user_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ == '__main__']
imported_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ != '__main__']
imported_pkgs = [name for name in globals() if isinstance(globals()[name], types.ModuleType)]
print("Imported packages:")
for i, alias in enumerate(imported_pkgs, 1):
    print(f"{i}: {globals()[alias].__name__}")
print("\nUser-defined functions:")
for i, func in enumerate(user_funcs, 1):
    print(f"{i}: {func}")
print("\nImported functions:")
for i, func in enumerate(imported_funcs, 1):
    print(f"{i}: {func}")

