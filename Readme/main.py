"""
ML Model Benchmark for Breast Cancer Wisconsin dataset
-----------------------------------------------------
Образовательный проект для портфолио: сравнение нескольких классификаторов
(LogisticRegression, DecisionTree, RandomForest, GradientBoosting) на датасете
breast cancer (sklearn.datasets.load_breast_cancer). Включает подбор гиперпараметров
через GridSearchCV, оценку на отложенной тестовой выборке, сохранение метрик и
визуализаций.

Как использовать:
- Установите зависимости: scikit-learn, pandas, matplotlib
- Запустите: python ml_model_benchmark_commented.py
- Результаты сохранятся в папке "results"
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ------------------------------------------------------------------
# Константы конфигурации проекта
# ------------------------------------------------------------------
RANDOM_STATE = 42  # фиксируем сид для воспроизводимости
TEST_SIZE = 0.2  # доля отложенной тестовой выборки
N_SPLITS = 5  # число фолдов для кросс-валидации при поиске гиперпараметров
RESULTS_DIR = Path("results")  # директория для вывода метрик и графиков


def load_data():
    """Load the breast cancer dataset.

    Возвращает:
    - X: pandas.DataFrame с признаками
    - y: pandas.Series с метками (0/1)
    - target_names: имена классов (например, ['malignant', 'benign'])
    """
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names

    return X, y, target_names


def build_models():
    """Create baseline machine learning models.

    Здесь модели подготовлены в "сыром" виде (без настройки гиперпараметров).
    LogisticRegression обёрнут в pipeline с StandardScaler — это необходимо,
    так как LR чувствительна к масштабу признаков.
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

    Подбор параметров — грубая сетка для демонстрации и сравнения моделей.
    Для большого проекта имеет смысл сузить и/или использовать RandomizedSearch.
    """
    return {
        "Logistic Regression": {
            # В pipeline имя шага стало "logisticregression" (сконструировано sklearn)
            "logisticregression__C": [0.01, 0.1, 1, 10],
            "logisticregression__solver": ["liblinear", "lbfgs"],
        },
        "Decision Tree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "max_depth": [1, 2, 3],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    }


def tune_hyperparameters(X_train, y_train, models):
    """Tune models using training data only.

    Пояснения:
    - Используем StratifiedKFold, чтобы сохранить соотношение классов в фолдах.
    - Метрика подбора: F1 для класса malignant (в этом датасете класс 0 — malignant).
      F1 выбран, потому что важен баланс precision и recall для обнаружения
      злокачественных опухолей. zero_division=0 предотвращает ошибки при делении.
    - GridSearchCV с n_jobs=-1 использует все ядра для ускорения поиска.
    """
    param_grids = build_param_grids()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Внимание: pos_label=0 — мы считаем важным класс "malignant" (0)
    scorer = make_scorer(
        f1_score,
        pos_label=0,
        zero_division=0,
    )

    tuned_models = {}
    best_params = {}
    best_cv_scores = {}

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

        # Подбираем на тренировочной выборке (внутри GridSearchCV уже кросс-валидация)
        search.fit(X_train, y_train)

        # Сохраняем лучший обученный объект, параметры и CV-скор
        tuned_models[name] = search.best_estimator_
        best_params[name] = search.best_params_
        best_cv_scores[name] = search.best_score_

        print(f"Best CV F1 (malignant): {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    return tuned_models, best_params, best_cv_scores


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    tuned_models,
    best_cv_scores,
):
    """Evaluate tuned models on the held-out test set.

    Проводится:
    - Повторная тренировка (clone + fit) для честного измерения времени обучения.
      clone делает глубокую копию модели, чтобы не переиспользовать один объект.
    - Предсказания на тесте, расчет accuracy, precision/recall/F1 для класса malignant.
    - ROC-AUC рассчитывается для вероятности предсказания класса malignant (класс 0).
    """
    rows = []
    predictions = {}

    for name, tuned_model in tuned_models.items():
        # Create a fresh model and measure training time.
        model = clone(tuned_model)

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        # Generate predictions and probabilities.
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        # В этом датасете класс 0 соответствует malignant — берем вероятность этого класса.
        # predict_proba возвращает массив shape=(n_samples, n_classes), индексируем колонку для класса 0.
        malignant_scores = model.predict_proba(X_test)[:, 0]
        malignant_auc = roc_auc_score(
            (y_test == 0).astype(int),
            malignant_scores,
        )

        rows.append(
            {
                "Model": name,
                # Сохраняем лучший CV-скор, найденный при подборе гиперпараметров
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
                "Training Time (sec)": training_time,
            }
        )

    results_df = pd.DataFrame(rows)
    # Сортируем по F1 для класса malignant — это целевая метрика
    results_df = results_df.sort_values(
        by="Test F1 (malignant)",
        ascending=False,
    ).reset_index(drop=True)

    return results_df, predictions


def save_results(results_df, best_params):
    """Save metrics and selected hyperparameters to CSV files.

    Результаты:
    - results/final_report.csv — сравнительная таблица метрик на тесте
    - results/best_hyperparameters.csv — выбранные параметры (JSON в ячейке)
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

    Формат: CSV с отдельной строкой на модель + класс с precision/recall/f1/support.
    Также печатаем отчёты в stdout — удобно для быстрой проверки.
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

    Создаём сетку 2x2 (под 4 модели). Для большего числа моделей можно
    автоматически подбирать размеры сетки.
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

    ROC строится для способности отделять класс malignant (0) от остальных.
    Отмечаем AUC в легенде.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    # бинарная цель для ROC: 1 — malignant, 0 — benign (мы хотим положительный маркер для malignant)
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
    # Создаём директорию результатов заранее
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, target_names = load_data()

    # Стратифицированное разбиение избегает дисбаланса классов между train/test
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

    tuned_models, best_params, best_cv_scores = (
        tune_hyperparameters(
            X_train,
            y_train,
            models,
        )
    )

    print("\n=== Hyperparameter Search Results ===")
    for name, score in best_cv_scores.items():
        print(f"{name}: best CV F1 (malignant) = {score:.4f}")

    results_df, predictions = evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        tuned_models,
        best_cv_scores,
    )

    print("\n=== Final Test Set Comparison ===")
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
