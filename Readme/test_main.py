import pandas as pd
from sklearn.model_selection import train_test_split

import main


def test_load_data_returns_expected_shapes():
    X, y, target_names = main.load_data()

    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 30
    assert len(target_names) == 2
    assert set(y.unique()) == {0, 1}


def test_build_models_returns_four_models():
    models = main.build_models()

    assert len(models) == 4
    assert set(models.keys()) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
    }


def test_hyperparameter_tuning_returns_all_models():
    X, y, _ = main.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = main.build_models()
    tuned_models, best_params, best_cv_scores = main.tune_hyperparameters(
        X_train,
        y_train,
        models,
    )

    assert len(tuned_models) == 4
    assert len(best_params) == 4
    assert len(best_cv_scores) == 4
    assert set(tuned_models.keys()) == set(models.keys())
    assert all(isinstance(value, float) for value in best_cv_scores.values())


def test_evaluate_models_returns_valid_metrics():
    X, y, _ = main.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = main.build_models()
    tuned_models, _, best_cv_scores = main.tune_hyperparameters(
        X_train,
        y_train,
        models,
    )
    results_df, predictions = main.evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        tuned_models,
        best_cv_scores,
    )

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 4
    assert len(predictions) == 4

    required_columns = {
        "Model",
        "Best CV F1 (malignant)",
        "Test Accuracy",
        "Test Precision (malignant)",
        "Test Recall (malignant)",
        "Test F1 (malignant)",
    }
    assert required_columns.issubset(set(results_df.columns))

    for metric in [
        "Test Accuracy",
        "Test Precision (malignant)",
        "Test Recall (malignant)",
        "Test F1 (malignant)",
    ]:
        assert results_df[metric].between(0, 1).all(), (
            f"{metric} has values outside [0, 1]"
        )

    assert set(results_df["Model"]) == set(models.keys())


def test_save_results_creates_csv_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    X, y, _ = main.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = main.build_models()
    tuned_models, best_params, best_cv_scores = main.tune_hyperparameters(
        X_train,
        y_train,
        models,
    )
    results_df, predictions = main.evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        tuned_models,
        best_cv_scores,
    )

    main.save_results(results_df, best_params)

    assert (tmp_path / "results" / "final_report.csv").exists()
    assert (tmp_path / "results" / "best_hyperparameters.csv").exists()

    comparison_df = pd.read_csv(tmp_path / "results" / "final_report.csv")
    validation_df = pd.read_csv(tmp_path / "results" / "best_hyperparameters.csv")

    assert not comparison_df.empty
    assert not validation_df.empty
    assert len(comparison_df) == 4
    assert len(validation_df) == 4
