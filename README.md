# ML Model Benchmark

Educational project comparing machine-learning models on two datasets:

1. **Breast Cancer Wisconsin (Diagnostic)** — scikit-learn classification benchmark.
2. **CIFAR-10** — computer-vision benchmark comparing a small CNN with ResNet18.

The repository also includes an interactive Streamlit dashboard for the breast-cancer benchmark. This project is for education and research only; it is **not intended for medical diagnosis or clinical decision-making**.

## Project at a glance

| Experiment | Entry point | Dataset | Results location |
|---|---|---|---|
| Tabular classification | `Readme/main.py` | Breast Cancer Wisconsin, from scikit-learn | `Readme/results/` when run from `Readme/` |
| Computer vision | `Readme/cv_benchmark.py` | CIFAR-10, downloaded by torchvision | `Readme/results/computer_vision/` by default |
| Interactive dashboard | `Readme/app,py` | Breast Cancer Wisconsin, from scikit-learn | Displays results in the browser; does not save benchmark reports |

## Repository structure

```text
ML-Model-Benchmark-/
├── README.md                         # Project overview and usage guide
├── LICENSE
├── .gitignore
└── Readme/                           # Python project files and generated artifacts
    ├── main.py                       # Breast-cancer model training and evaluation
    ├── cv_benchmark.py               # CIFAR-10 CNN vs. ResNet18 benchmark
    ├── app,py                        # Streamlit dashboard (filename includes a comma)
    ├── requirements.txt              # Python dependencies
    ├── test_main.py                  # Tests for the tabular benchmark
    ├── test_cv_benchmark.py          # Tests for the computer-vision benchmark
    ├── python-tests.yml              # CI workflow definition (see CI note below)
    ├── results/                      # Tracked tabular benchmark reports and plots
    │   ├── best_hyperparameters.csv
    │   ├── classification_reports.csv
    │   ├── confusion_matrices.png
    │   ├── cross_validation.csv
    │   ├── final_report.csv
    │   ├── model_comparison.csv
    │   ├── roc_curves.png
    │   └── computer_vision/          # Created when the CIFAR-10 benchmark runs
    └── data/                         # CIFAR-10 download/cache, created at runtime
```

### Where to find things

- **Python source:** `Readme/`
- **Tests:** `Readme/test_main.py` and `Readme/test_cv_benchmark.py`
- **Existing breast-cancer outputs:** `Readme/results/`
- **CIFAR-10 outputs:** `Readme/results/computer_vision/` (created when that benchmark runs)
- **Downloaded CIFAR-10 data:** `Readme/data/cifar10/` (created automatically; not committed)

## Setup

Python 3.10 or newer is recommended.

From the repository root, create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r Readme/requirements.txt
```

## Run the benchmarks

Run commands from the `Readme/` directory so generated reports are saved under `Readme/results/`:

```bash
cd Readme
python main.py
```

This runs hyperparameter tuning, nested cross-validation, and final evaluation for Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. The held-out test set is reserved for final evaluation.

Run the CIFAR-10 experiment from the same directory:

```bash
python cv_benchmark.py
```

The first run downloads CIFAR-10. Default settings use a CPU-friendly subset and two training epochs per model. To adjust the run, for example:

```bash
python cv_benchmark.py --train-samples 20000 --test-samples 5000 --epochs 5
```

The CIFAR-10 benchmark writes its reports and visualizations to `results/computer_vision/` and caches downloaded data under `data/cifar10/` by default.

## Run the interactive dashboard

The dashboard is in `Readme/app,py` (the current filename contains a comma). From the repository root, run:

```bash
python -m streamlit run "Readme/app,py"
```

Streamlit will print a local URL to open in your browser. The dashboard compares the breast-cancer models and allows an editable sample prediction. Its predictions are educational demonstrations, not medical advice.

## Results guide

### Breast-cancer benchmark (`Readme/results/`)

| File | What it contains |
|---|---|
| `final_report.csv` | Test metrics and cross-validation summary for each model |
| `best_hyperparameters.csv` | Best parameter configuration found for each model |
| `classification_reports.csv` | Per-class precision, recall, F1-score, and support |
| `confusion_matrices.png` | Confusion matrices for the four classifiers |
| `roc_curves.png` | ROC curves for the four classifiers |
| `cross_validation.csv`, `model_comparison.csv` | Additional saved comparison outputs |

### CIFAR-10 benchmark (`Readme/results/computer_vision/`)

The computer-vision script generates `final_report.csv`, `training_history.csv`, `classification_reports.csv`, `confusion_matrices.png`, `training_curves.png`, and `predictions.png`.

## Run tests

From the `Readme/` directory:

```bash
python -m pytest -q
```

The CIFAR-10 tests use synthetic test data for their smoke checks, avoiding a full CIFAR-10 download during those tests.

> **CI note:** `python-tests.yml` is currently stored in `Readme/`. GitHub Actions only discovers workflow files under `.github/workflows/`, so this file will not run as a GitHub Actions workflow until it is moved there.

## Methodology and limitations

- The breast-cancer experiment uses an 80/20 stratified train/test split. Hyperparameter tuning and nested cross-validation use the training split; the held-out test split is used for final evaluation.
- The CIFAR-10 experiment reserves part of its selected training subset for validation and keeps its selected test subset separate for final evaluation.
- The datasets and experimental settings are limited. Results are educational comparisons and should not be treated as evidence of clinical or production readiness.

## Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, PyTorch, torchvision, Streamlit, and pytest.
