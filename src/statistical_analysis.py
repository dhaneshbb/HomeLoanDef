import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy.stats import chi2_contingency, fisher_exact, kurtosis, shapiro, skew, kstest, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor


def chi_square_test(data, target_col="y"):
    chi_square_results = {}
    categorical_vars = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
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


def fisher_exact_test(data, target_col="y"):
    fisher_exact_results = {}
    categorical_vars = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    for col in categorical_vars:
        if col == target_col:
            continue
        contingency_table = pd.crosstab(data[col], data[target_col])
        # Apply Fisher's Exact Test only for 2x2 tables
        if contingency_table.shape == (2, 2):
            _, p_value = fisher_exact(contingency_table)
            fisher_exact_results[col] = p_value
            print(f"- {col} (Fisher's Exact Test): p-value = {p_value:.4f}")
    return fisher_exact_results


def spearman_correlation_with_target(
    data, non_normal_cols, target_col="TARGET", plot=True, table=True
):
    if not pd.api.types.is_numeric_dtype(data[target_col]):
        raise ValueError(
            f"Target column '{target_col}' must be numeric. Please encode it before running this test."
        )
    correlation_results = {}
    for col in non_normal_cols:
        if col not in data.columns:
            continue
        coef, p_value = spearmanr(data[col], data[target_col], nan_policy="omit")
        correlation_results[col] = {"Spearman Coefficient": coef, "p-value": p_value}
    correlation_data = pd.DataFrame(correlation_results).T.dropna()
    correlation_data = correlation_data.sort_values(
        "Spearman Coefficient", ascending=False
    )
    if target_col in correlation_data.index:
        correlation_data = correlation_data.drop(target_col)
    positive_corr = correlation_data[correlation_data["Spearman Coefficient"] > 0]
    negative_corr = correlation_data[correlation_data["Spearman Coefficient"] < 0]
    if table:
        print(f"\nPositive Spearman Correlations with Target ('{target_col}'):\n")
        for feature, stats in positive_corr.iterrows():
            print(
                f"- {feature}: Correlation={stats['Spearman Coefficient']:.4f}, p-value={stats['p-value']:.4f}"
            )
        print(f"\nNegative Spearman Correlations with Target ('{target_col}'):\n")
        for feature, stats in negative_corr.iterrows():
            print(
                f"- {feature}: Correlation={stats['Spearman Coefficient']:.4f}, p-value={stats['p-value']:.4f}"
            )
    if plot:
        plt.figure(figsize=(20, 8))  # Increase figure width to prevent label overlap
        sns.barplot(
            x=correlation_data.index,
            y="Spearman Coefficient",
            data=correlation_data,
            palette="coolwarm",
        )
        plt.axhline(0, color="black", linestyle="--")
        plt.title(f"Spearman Correlation with Target ('{target_col}')", fontsize=16)
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Spearman Coefficient", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate labels for clarity
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
                test_used = "Shapiro-Wilk"
            else:
                stat, p_value = kstest(
                    col_data, "norm", args=(col_data.mean(), col_data.std())
                )
                test_used = "Kolmogorov-Smirnov"
            col_skewness = skew(col_data)
            col_kurtosis = kurtosis(col_data)
            result = {
                "Column": col,
                "Test": test_used,
                "Statistic": stat,
                "p_value": p_value,
                "Skewness": col_skewness,
                "Kurtosis": col_kurtosis,
            }
            if p_value > 0.05:
                normal_cols.append(result)
            else:
                not_normal_cols.append(result)
    normal_df = (
        pd.DataFrame(normal_cols).sort_values(by="Column")
        if normal_cols
        else pd.DataFrame(
            columns=["Column", "Test", "Statistic", "p_value", "Skewness", "Kurtosis"]
        )
    )
    not_normal_df = (
        pd.DataFrame(not_normal_cols).sort_values(
            by="p_value", ascending=False
        )  # Sort by p-value descending (near normal to not normal)
        if not_normal_cols
        else pd.DataFrame(
            columns=["Column", "Test", "Statistic", "p_value", "Skewness", "Kurtosis"]
        )
    )
    print("\nNormal Columns (p > 0.05):")
    display(normal_df)
    print("\nNot Normal Columns (p d 0.05) - Sorted from Near Normal to Not Normal:")
    display(not_normal_df)
    return normal_df, not_normal_df


def spearman_correlation(
    data, non_normal_cols, exclude_target=None, multicollinearity_threshold=0.8
):
    if non_normal_cols.empty:
        print(
            "\nNo non-normally distributed numerical columns found. Exiting Spearman Correlation."
        )
        return
    selected_columns = non_normal_cols["Column"].tolist()
    if (
        exclude_target
        and exclude_target in selected_columns
        and pd.api.types.is_numeric_dtype(data[exclude_target])
    ):
        selected_columns.remove(exclude_target)
    spearman_corr_matrix = data[selected_columns].corr(method="spearman")
    multicollinear_pairs = []
    for i, col1 in enumerate(selected_columns):
        for col2 in selected_columns[i + 1 :]:
            coef = spearman_corr_matrix.loc[col1, col2]
            if abs(coef) > multicollinearity_threshold:
                multicollinear_pairs.append((col1, col2, coef))
    print(
        "\nVariables Exhibiting Multicollinearity (|Correlation| > {:.2f}):".format(
            multicollinearity_threshold
        )
    )
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
        fmt="",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": annot_font_size},
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Spearman Correlation Matrix", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()


def calculate_vif(data, exclude_target="TARGET", multicollinearity_threshold=5.0):
    # Select only numeric columns, exclude target, and drop rows with missing values
    numeric_data = (
        data.select_dtypes(include=[np.number])
        .drop(columns=[exclude_target], errors="ignore")
        .dropna()
    )
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_data.columns
    vif_data["VIF"] = [
        variance_inflation_factor(numeric_data.values, i)
        for i in range(numeric_data.shape[1])
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    high_vif = vif_data[vif_data["VIF"] > multicollinearity_threshold]
    low_vif = vif_data[vif_data["VIF"] <= multicollinearity_threshold]
    print(
        f"\nVariance Inflation Factor (VIF) Scores (multicollinearity_threshold = {multicollinearity_threshold}):"
    )
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
    return vif_data, high_vif["Feature"].tolist()
