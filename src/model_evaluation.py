import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


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
        "Training Time (seconds)": round(training_time, 4),
    }


def cross_validation_analysis_table(
    model, X_train, y_train, cv_folds=5, scoring_metric="f1"
):
    strat_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train, cv=strat_kfold, scoring=scoring_metric
    )
    cv_results_df = pd.DataFrame(
        {"Fold": [f"Fold {i+1}" for i in range(cv_folds)], "F1-Score": scores}
    )
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
        results.append(
            {
                "Threshold": round(threshold, 2),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4),
                "Accuracy": round(accuracy, 4),
                "True Negatives (TN)": tn,
                "False Positives (FP)": fp,
                "False Negatives (FN)": fn,
                "True Positives (TP)": tp,
            }
        )
    df_results = pd.DataFrame(results)
    best_threshold = df_results.loc[df_results["F1-Score"].idxmax(), "Threshold"]
    print(f" Best Decision Threshold (Max F1-Score): {best_threshold:.2f}")
    return df_results, best_threshold


def show_default_feature_importance(models, X_train):
    feature_importance_df = pd.DataFrame({"Feature": X_train.columns})
    for model_name, model in tune_models.items():
        if hasattr(model, "feature_importances_"):
            feature_importance_df[model_name] = model.feature_importances_
        else:
            print(f" Feature importance not available for {model_name}")
    feature_importance_df = feature_importance_df.sort_values(
        by="XGBoost", ascending=False
    )
    print("\n Default Feature Importance Across Models:")
    print(feature_importance_df.to_markdown(tablefmt="pipe", index=False))
