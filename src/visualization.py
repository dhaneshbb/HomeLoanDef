import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

sns.set(style="whitegrid")


def plot_histograms(columns, data):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data[col], kde=True, bins=30, color="skyblue", stat="density")
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()


def plot_boxplots(columns, data):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columns, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=data[col], color="lightgreen")
        plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()


def plot_scatter(columns_x, columns_y, data):
    plt.figure(figsize=(15, 10))
    for i, col_x in enumerate(columns_x):
        for j, col_y in enumerate(columns_y):
            plt.subplot(len(columns_x), len(columns_y), i * len(columns_y) + j + 1)
            sns.scatterplot(x=data[col_x], y=data[col_y])
            plt.title(f"{col_x} vs {col_y}")
    plt.tight_layout()
    plt.show()


def plot_bar_plots(columns, data):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columns, 1):
        plt.subplot(3, 3, i)
        sns.countplot(x=data[col], palette="Set2")
        plt.title(f"Bar plot of {col}")
    plt.tight_layout()
    plt.show()


def plot_scatter_df(income_and_loan_cols, age_and_employment_cols, at):
    num_income_cols = len(income_and_loan_cols)
    num_age_cols = len(age_and_employment_cols)
    total_plots = num_income_cols * num_age_cols
    fig, axes = plt.subplots(
        num_income_cols, num_age_cols, figsize=(5 * num_age_cols, 5 * num_income_cols)
    )
    if total_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    plot_index = 0
    for income_col in income_and_loan_cols:
        for age_col in age_and_employment_cols:
            ax = axes[plot_index]
            sns.scatterplot(x=at[age_col], y=at[income_col], alpha=0.5, ax=ax)
            ax.set_xlabel(age_col.replace("_", " "))
            ax.set_ylabel(income_col.replace("_", " "))
            ax.set_title(f"{income_col} vs {age_col}")
            plot_index += 1

    plt.tight_layout()
    plt.show()


def plot_all_evaluation_metrics(model, X_test, y_test):
    import scikitplot as skplt

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
    skplt.metrics.plot_cumulative_gain(
        y_test, model.predict_proba(X_test), ax=axes[0, 1]
    )
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
