# Extracted utility functions from the Jupyter Notebook

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

def cap_extreme_values(df, columns):
    for col in columns:
        percentile_99 = df[col].quantile(0.99)
        df[col] = np.minimum(df[col], percentile_99)
    return df
columns_to_cap = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_APPLICATION', 'AMT_ANNUITY']
pa = cap_extreme_values(pa, columns_to_cap)

def cap_outliers(df, columns, cap_value):
    for col in columns:
        df[col] = df[col].apply(lambda x: min(x, cap_value))  # Cap values at the given threshold
    return df
columns_to_cap = ['CNT_INSTALMENT_MATURE_CUM', 'CNT_DRAWINGS_POS_CURRENT', 'AMT_BALANCE']
cap_values = {'CNT_INSTALMENT_MATURE_CUM': 120, 'CNT_DRAWINGS_POS_CURRENT': 100, 'AMT_BALANCE': 500000}
for col in columns_to_cap:
    ccb = cap_outliers(ccb, [col], cap_values[col])

def cap_outliers(df, columns, percentile=0.99):
    for col in columns:
        cap_value = df[col].quantile(percentile)
        df[col] = np.minimum(df[col], cap_value)
    return df

columns_to_cap = ['AMT_INSTALMENT', 'AMT_PAYMENT']
ip = cap_outliers(ip, columns_to_cap)

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