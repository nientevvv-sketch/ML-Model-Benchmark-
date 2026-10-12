# ML Model Benchmark

A beginner-friendly machine learning project that compares four classification algorithms on the Breast Cancer Wisconsin dataset from scikit-learn.

This project is intended as an educational benchmark and portfolio example. It demonstrates core machine learning workflow steps: data loading, train/test splitting, model comparison, cross-validation, metric evaluation, and confusion matrix visualization.

Important: this project is for learning and portfolio purposes only. It is not intended for clinical diagnosis or real-world medical decision-making.

## Project goal

The project compares the following classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

These models are evaluated on the Breast Cancer Wisconsin dataset using classification metrics and cross-validation.

## Dataset

The dataset used is the scikit-learn `load_breast_cancer` dataset.

- Samples: 569
- Features: 30 numerical attributes
- Target classes:
  - 0 = malignant
  - 1 = benign

The data is split into training and test sets using an 80/20 split with:
- `random_state=42`
- `stratify=y`

## Methodology

The project follows a standard supervised learning workflow:

1. Load the dataset
2. Split into training and test sets
3. Train the selected models
4. Evaluate models on the held-out test set
5. Run 5-fold cross-validation on the training set
6. Compare model performance
7. Visualize confusion matrices

## Current experiment results

The values below reflect the current experiment in this repository and should be interpreted as results from this project, not as universal performance guarantees.

### Test set performance

| Model | Accuracy | Precision (malignant) | Recall (malignant) | F1 (malignant) |
|---|---:|---:|---:|---:|
| Logistic Regression | 98.25% | 97.62% | 97.62% | 97.62% |
| Decision Tree | 91.23% | 84.78% | 92.86% | 88.64% |
| Random Forest | 95.61% | 95.12% | 92.86% | 93.98% |
| Gradient Boosting | 95.61% | 97.44% | 90.48% | 93.83% |

### 5-fold cross-validation on the training set

| Model | Mean Accuracy | Mean Recall (malignant) | Mean F1 (malignant) |
|---|---:|---:|---:|
| Logistic Regression | 97.80% | 96.47% | 97.03% |
| Decision Tree | 91.65% | 92.35% | 89.19% |
| Random Forest | 96.26% | 95.88% | 95.06% |
| Gradient Boosting | 95.16% | 93.53% | 93.53% |

## Repository structure

```text
ML-Model-Benchmark/
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
├── results/
│   ├── model_comparison.csv
│   └── cross_validation.csv
└── .venv/                  # optional local virtual environment[README.md](https://github.com/user-attachments/files/33327025/README.md)
