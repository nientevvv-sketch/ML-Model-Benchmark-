"""
ML Model Benchmark for Breast Cancer Wisconsin dataset
-----------------------------------------------------
Educational portfolio project: comparison of several classifiers
(LogisticRegression, DecisionTree, RandomForest, GradientBoosting) on the
breast cancer dataset (sklearn.datasets.load_breast_cancer). Includes
hyperparameter tuning via GridSearchCV, nested cross-validation for robust
model comparison, evaluation on a held-out test set, and saving metrics and
visualizations.

Methodology:
1. Split data into train/test (stratified). The test set is not touched until the end.
2. Hyperparameters are tuned with GridSearchCV only on the training set.
3. Models are compared using nested CV (on train), not using the test set.
4. The test set is used once—for the final report on already chosen models.

How to use:
- Install dependencies: scikit-learn, pandas, matplotlib
- Run: python ml_model_benchmark_commented.py
- Results will be saved in the "results" folder
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ------------------------------------------------------------------
# Project configuration constants
# ------------------------------------------------------------------
RANDOM_STATE = 42  # fix seed for reproducibility
TEST_SIZE = 0.2  # fraction of the held-out test set
N_SPLITS = 5  # number of folds for CV during hyperparameter search
OUTER_SPLITS = 5  # folds in the outer loop of nested CV
OUTER_REPEATS = 3  # number of repeats for the outer loop
INNER_SPLITS = 3  # folds in the inner loop of nested CV (hyperparameter search)
RESULTS_DIR = Path("results")  # directory for metrics and plots output


def make_malignant_scorer():
    """F1 for the malignant class (in this dataset class 0 = malignant).

    F1 is chosen because we care about the balance between precision and recall
    for detecting malignant tumors. zero_division=0 prevents division errors.
    """
    return make_scorer(f1_score, pos_label=0, zero_division=0)


def load_data():
    """Load the breast cancer dataset.

    Returns:
    - X: pandas.DataFrame with features
    - y: pandas.Series with labels (0/1)
    - target_names: class names (e.g., ['malignant', 'benign'])
    """
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names

    return X, y, target_names


def build_models():
    """Create baseline machine learning models.

    Models are prepared in raw form (no tuned hyperparameters).
    LogisticRegression is wrapped in a pipeline with StandardScaler — LR is
    sensitive to feature scaling.
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def build_param_grids():
    """Define hyperparameter search spaces for each model.

    Grids are intentionally small: on ~450 training samples very wide grids
    rarely pay off and only slow down search while increasing the risk of
    overfitting to CV folds.
    """
    return {
        "Logistic Regression": {
            # In the pipeline the step name becomes "logisticregression" (sklearn)
            "logisticregression__C": [0.01, 0.1, 1, 10],
            "logisticregression__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "n_estimators": [100, 300],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
        },
    }


def tune_hyperparameters(X_train, y_train, models):
    """Tune models using training data only.

    Notes:
    - We use StratifiedKFold to preserve class proportions in folds.
    - The tuning metric is F1 for the malignant class.
    - GridSearchCV with n_jobs=-1 uses all cores to speed up search.
    - refit_time_ is the time of the final fit of the best model on the whole
      training set; we use it as the training time instead of manual re-training.
    """
    param_grids = build_param_grids()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scorer = make_malignant_scorer()

    tuned_models = {}
    best_params = {}
    best_cv_scores = {}
    refit_times = {}

    for name, model in models.items():
        print(f"\nTuning {name}...")

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

        # Tune on the training data (GridSearchCV includes internal CV)
        search.fit(X_train, y_train)

        # Store the best estimator, parameters, CV score and refit time
        tuned_models[name] = search.best_estimator_
        best_params[name] = search.best_params_
        best_cv_scores[name] = search.best_score_
        refit_times[name] = search.refit_time_

        print(f"Best CV F1 (malignant): {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    return tuned_models, best_params, best_cv_scores, refit_times


def nested_cv_scores(X_train, y_train, models):
    """Estimate model quality with repeated nested cross-validation.

    The outer loop estimates generalization, the inner loop (GridSearchCV)
    tunes hyperparameters. This way final scores do not depend on whether the
    hyperparameters happened to fit particular folds. The test set is not used here.

    Returns a dict {model_name: (mean F1, std)} across outer folds.
    """
    param_grids = build_param_grids()
    scorer = make_malignant_scorer()

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )

    results = {}

    for name, model in models.items():
        print(f"Nested CV: {name}...")

        # Parallelize only the outer loop to avoid nested process spawning
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring=scorer,
            cv=inner_cv,
            n_jobs=1,
        )

        scores = cross_val_score(
            search,
            X_train,
            y_train,
            cv=outer_cv,
            scoring=scorer,
            n_jobs=-1,
        )

        results[name] = (scores.mean(), scores.std())
        print(f"  F1 (malignant) = {scores.mean():.4f} ± {scores.std():.4f}")

    return results


def evaluate_models(
    X_test,
    y_test,
    tuned_models,
    best_cv_scores,
    nested_scores,
    refit_times,
):
    """Evaluate tuned models on the held-out test set.

    Steps:
    - Predictions on the test set, compute accuracy, precision/recall/F1 for malignant.
    - Compute ROC-AUC for the probability of the malignant class (class 0).
    - Models are sorted by mean F1 from nested CV (not by test), so the test is
      not involved in selecting the best model.
    """
    rows = []
    predictions = {}

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        # In this dataset class 0 corresponds to malignant — take probability for that class.
        # predict_proba returns shape=(n_samples, n_classes); column 0 is class 0.
        malignant_scores = model.predict_proba(X_test)[:, 0]
        malignant_auc = roc_auc_score(
            (y_test == 0).astype(int),
            malignant_scores,
        )

        nested_mean, nested_std = nested_scores[name]

        rows.append(
            {
                "Model": name,
                "Nested CV F1 mean (malignant)": nested_mean,
                "Nested CV F1 std (malignant)": nested_std,
                "Best CV F1 (malignant)": best_cv_scores[name],
                "Test Accuracy": accuracy_score(y_test, y_pred),
                "Test Precision (malignant)": precision_score(
                    y_test,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                "Test Recall (malignant)": recall_score(
                    y_test,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                "Test F1 (malignant)": f1_score(
                    y_test,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                "Test ROC-AUC (malignant)": malignant_auc,
                "Training Time (sec)": refit_times[name],
            }
        )

    results_df = pd.DataFrame(rows)

    # Sort by nested CV mean, not by test
    results_df = results_df.sort_values(
        by="Nested CV F1 mean (malignant)",
        ascending=False,
    ).reset_index(drop=True)

    return results_df, predictions


def save_results(results_df, best_params):
    """Save metrics and selected hyperparameters to CSV files.

    Outputs:
    - results/final_report.csv — comparison table of metrics
    - results/best_hyperparameters.csv — chosen parameters (JSON in a cell)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        RESULTS_DIR / "final_report.csv",
        index=False,
    )

    params_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Best Parameters": json.dumps(
                    params,
                    sort_keys=True,
                ),
            }
            for name, params in best_params.items()
        ]
    )

    params_df.to_csv(
        RESULTS_DIR / "best_hyperparameters.csv",
        index=False,
    )


def save_classification_reports(
    predictions,
    y_test,
    target_names,
):
    """Save detailed classification reports for each model.

    Format: CSV with one row per model+class including precision/recall/f1/support.
    Also print reports to stdout — convenient for quick checks.
    """
    report_rows = []

    for name, y_pred in predictions.items():
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

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

        for class_name in target_names:
            report_rows.append(
                {
                    "Model": name,
                    "Class": class_name,
                    "Precision": report[class_name]["precision"],
                    "Recall": report[class_name]["recall"],
                    "F1-score": report[class_name]["f1-score"],
                    "Support": report[class_name]["support"],
                }
            )

    pd.DataFrame(report_rows).to_csv(
        RESULTS_DIR / "classification_reports.csv",
        index=False,
    )


def plot_confusion_matrices(
    predictions,
    y_test,
    target_names,
):
    """Plot and save confusion matrices for all models.

    Create a 2x2 grid (for 4 models). For more models the grid size can be
    adjusted automatically.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, y_pred) in zip(
        axes.ravel(),
        predictions.items(),
    ):
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

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_roc_curves(X_test, y_test, tuned_models):
    """Plot and save ROC curves for all tuned models.

    ROC is computed for distinguishing the malignant class (0) from the rest.
    AUC values are shown in the legend.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    # binary target for ROC: 1 = malignant, 0 = benign
    y_true = (y_test == 0).astype(int)

    for name, model in tuned_models.items():
        malignant_scores = model.predict_proba(X_test)[:, 0]

        fpr, tpr, _ = roc_curve(y_true, malignant_scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path = RESULTS_DIR / "roc_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"ROC curves saved to: {output_path}")


def main():
    # Create the results directory ahead of time
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, target_names = load_data()

    # Stratified split preserves class balance between train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=== Dataset Information ===")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {list(target_names)}")

    models = build_models()

    tuned_models, best_params, best_cv_scores, refit_times = tune_hyperparameters(
        X_train,
        y_train,
        models,
    )

    print("\n=== Hyperparameter Search Results ===")
    for name, score in best_cv_scores.items():
        print(f"{name}: best CV F1 (malignant) = {score:.4f}")

    # Nested CV — only on train, test is not used
    print("\n=== Nested Cross-Validation (train only) ===")
    nested_scores = nested_cv_scores(X_train, y_train, models)

    results_df, predictions = evaluate_models(
        X_test,
        y_test,
        tuned_models,
        best_cv_scores,
        nested_scores,
        refit_times,
    )

    print("\n=== Final Comparison (sorted by nested CV, test metrics for reference) ===")
    print(results_df.round(4).to_string(index=False))

    save_results(results_df, best_params)

    save_classification_reports(
        predictions,
        y_test,
        target_names,
    )

    plot_confusion_matrices(
        predictions,
        y_test,
        target_names,
    )

    plot_roc_curves(
        X_test,
        y_test,
        tuned_models,
    )

    print("\n=== Results Saved ===")
    print(f"Output directory: {RESULTS_DIR.resolve()}")
    print("- final_report.csv")
    print("- best_hyperparameters.csv")
    print("- classification_reports.csv")
    print("- confusion_matrices.png")
    print("- roc_curves.png")


if __name__ == "__main__":
    main()
