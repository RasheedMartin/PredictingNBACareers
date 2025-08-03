import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import matplotlib.pyplot as plt

class NumericalCustomCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # If input is Pandas, convert to Polars
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        X_cleaned = X.with_columns(
            *[pl.when((pl.col(col) == float("inf")) | (pl.col(col) == float("-inf")) | (pl.col(col).is_nan()))
              .then(0).otherwise(pl.col(col)).alias(col) for col in X.columns]
        )

        return X_cleaned.to_numpy()

def pipeline():
    numeric_features = [
        'PF',
        '2PA_third',
        'STL_third',
        'WS/48_third',
        'BPM_third',
        'PTS',
        'FGA_third',
        'DRB_third',
        'FT_third',
        '2PA',
        'GS',
        'OBPM_third',
        'FT',
        'DWS_third',
        'FTA',
        'FG_third',
        'FG',
        'DRB',
        'WS_third',
        'USG%_third',
        'TRB_third',
        'AST_third',
        'TOV',
        'PER',
        'MP',
        'PF_third',
        'FTA_third',
        'PER_third',
        '2P_third',
        'PTS_third',
        '2P',
        'FGA',
        'VORP_third',
        'TOV_third',
        'GS_third',
        'MP_third'
    ]

    # Numeric preprocessing: fill missing values, scale
    numeric_transformer = Pipeline([
        ('numerical_cleaner', NumericalCustomCleaner()),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    # Combine into one preprocessor
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
    ])

    return preprocessor


def print_metrics(y_true, y_pred, target_names):
    print(classification_report(y_true, y_pred, target_names=target_names))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision (macro):", precision)
    print("Recall (macro):", recall)
    print("F1 Score (macro):", f1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='viridis')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.show()


def print_roc_curve(y_test, model, X_test_processed, classes=None):


    # Define your classes if not provided
    if classes is None:
        classes = ["Bust", "Role", "Star"]
    n_classes = len(classes)

    # Binarize the labels
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))  # y_test must be 0–(n-1)
    y_score = model.predict_proba(X_test_processed)  # shape (n_samples, n_classes)

    # --- ROC Curves ---
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"ROC curve for class {classes[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def print_precision_recall_curve(y_test, model, X_test_processed, classes=None):

    # Define your classes if not provided
    if classes is None:
        classes = ["Bust", "Role", "Star"]
    n_classes = len(classes)

    # Binarize the labels
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))  # y_test must be 0–(n-1)
    y_score = model.predict_proba(X_test_processed)  # shape (n_samples, n_classes)


    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    # Plot all PR curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f"PR curve for class {classes[i]} (AP = {average_precision[i]:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()