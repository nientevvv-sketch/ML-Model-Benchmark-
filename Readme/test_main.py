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


def test_cross_validation_has_expected_columns():
    X, y, _ = main.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = main.build_models()
    cv_df = main.run_cross_validation(X_train, y_train, models)

    assert isinstance(cv_df, pd.DataFrame)
    assert len(cv_df) == 4

    required_columns = {
        "Model",
        "accuracy mean",
        "accuracy std",
        "precision_malignant mean",
        "precision_malignant std",
        "recall_malignant mean",
        "recall_malignant std",
        "f1_malignant mean",
        "f1_malignant std",
    }

    assert required_columns.issubset(set(cv_df.columns))


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
    results_df, predictions = main.evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        models,
    )

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 4
    assert len(predictions) == 4

    for metric in ["Accuracy", "Precision (malignant)", "Recall (malignant)", "F1 (malignant)"]:
        assert results_df[metric].between(0, 1).all(), f"{metric} has values outside [0, 1]"

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
    cv_df = main.run_cross_validation(X_train, y_train, models)
    results_df, predictions = main.evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        models,
    )

    main.save_results(results_df, cv_df)

    assert (tmp_path / "results" / "model_comparison.csv").exists()
    assert (tmp_path / "results" / "cross_validation.csv").exists()

    comparison_df = pd.read_csv(tmp_path / "results" / "model_comparison.csv")
    validation_df = pd.read_csv(tmp_path / "results" / "cross_validation.csv")

    assert not comparison_df.empty
    assert not validation_df.empty
    assert len(comparison_df) == 4
    assert len(validation_df) == 4
