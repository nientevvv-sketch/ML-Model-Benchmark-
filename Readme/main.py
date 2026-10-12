import os
import time

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_data():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y, dataset.target_names


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def build_cv_scorer():
    return {
        "accuracy": "accuracy",
        "precision_malignant": make_scorer(
            precision_score, pos_label=0, zero_division=0
        ),
        "recall_malignant": make_scorer(
            recall_score, pos_label=0, zero_division=0
        ),
        "f1_malignant": make_scorer(
            f1_score, pos_label=0, zero_division=0
        ),
    }


def run_cross_validation(X_train, y_train, models):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = build_cv_scorer()

    rows = []
    for name, model in models.items():
        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        row = {"Model": name}
        for metric in scoring:
            row[f"{metric} mean"] = cv_results[f"test_{metric}"].mean()
            row[f"{metric} std"] = cv_results[f"test_{metric}"].std()
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    results = []
    predictions = {}

    for name, model in models.items():
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start_time

        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision (malignant)": precision_score(
                    y_test, y_pred, pos_label=0, zero_division=0
                ),
                "Recall (malignant)": recall_score(
                    y_test, y_pred, pos_label=0, zero_division=0
                ),
                "F1 (malignant)": f1_score(
                    y_test, y_pred, pos_label=0, zero_division=0
                ),
                "Training time (sec)": elapsed,
            }
        )

    return pd.DataFrame(results), predictions


def save_results(results_df, cv_df):
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/model_comparison.csv", index=False)
    cv_df.to_csv("results/cross_validation.csv", index=False)


def plot_confusion_matrices(predictions, y_test, target_names):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, y_pred) in zip(axes.ravel(), predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=[0, 1],
            display_labels=target_names,
            values_format="d",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(name)

    plt.tight_layout()
    plt.show()


def main():
    X, y, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    cv_df = run_cross_validation(X_train, y_train, models)
    print("\n=== 5-Fold Cross-Validation ===")
    print(cv_df.round(4).to_string(index=False))

    results_df, predictions = evaluate_models(X_train, y_train, X_test, y_test, models)
    print("\n=== Test Set Model Comparison ===")
    print(results_df.round(4).to_string(index=False))

    save_results(results_df, cv_df)

    for name, y_pred in predictions.items():
        print(f"\n=== {name}: Classification Report ===")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=target_names,
                zero_division=0,
            )
        )

    plot_confusion_matrices(predictions, y_test, target_names)


if __name__ == "__main__":
    main()
