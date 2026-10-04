# Custom Decision Tree Classifier

A from-scratch **Decision Tree** implementation using **information gain** (entropy-based splitting), validated against scikit-learn's `DecisionTreeClassifier` on the **Titanic** survival dataset.

## Executive Summary

This project builds a complete decision tree from first principles: recursive splitting via entropy minimization, support for both numeric and categorical features, configurable maximum depth, and tree visualization. On Titanic survival prediction with `max_depth=3`, the custom tree matches scikit-learn's training accuracy exactly (**82.66%**) and slightly outperforms on the test set (**74.83%** vs. **74.13%**).

## Technical Overview

### Algorithm

1. **Entropy** H(S) = −Σ pᵢ · log₂(pᵢ) measures label impurity (0 = pure, 1 = maximum chaos).
2. **Information Gain** IG = H(parent) − weighted_avg(H(left), H(right)) selects the best split.
3. **Splitting:**
   - Numeric features: threshold at midpoints between sorted unique values (`feature <= threshold`).
   - Categorical features: equality split (`feature == category`).
4. **Stopping criteria:** pure node, `max_depth` reached, or no gain-improving split found.
5. **Leaf prediction:** majority class vote.

### Architecture

```
Node                    → Tree node (split or leaf)
DecisionTree            → Orchestrates fit/predict
  ├── _build_recursive  → Top-down tree construction
  ├── _find_best_split  → Exhaustive split search
  └── _predict_recursive→ Single-row tree traversal
```

### API

```python
from DecisionTree import DecisionTree

tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
print(tree)  # ASCII tree visualization
```

## Installation & Setup

### Requirements

- Python 3.10+
- NumPy
- Pandas
- scikit-learn (benchmark comparison)
- Jupyter Notebook

### Setup

```bash
cd custom-decision-tree/ai-s3-decision-tree-misha-lrs2-main
pip install numpy pandas scikit-learn jupyter seaborn
```

## Usage

### Run the Validation Notebook

```bash
jupyter notebook tests.ipynb
```

The notebook:
1. Loads Titanic passenger data (`sex`, `age`, `class`, `fare`, `embarked` → survived)
2. Performs an 80/20 train-test split (`random_state=42`)
3. Trains custom and scikit-learn trees at `max_depth=3`
4. Compares accuracy and prints the tree structure

### Programmatic Example

```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

df = sns.load_dataset("titanic")
df = df[["survived", "pclass", "sex", "age", "fare", "embarked"]].dropna()

X = pd.get_dummies(df.drop("survived", axis=1))
y = df["survived"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)
print(tree)
```

## Results & Interpretation

### Titanic Survival — max_depth=3

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| **Custom DecisionTree** | 82.66% | **74.83%** |
| scikit-learn DecisionTreeClassifier | 82.66% | 74.13% |

### Interpretation

- **Identical training accuracy** confirms the split-finding logic matches scikit-learn's CART algorithm for this configuration.
- **Marginally higher test accuracy** (+0.70 pp) suggests equivalent generalization with no overfitting penalty from the custom implementation.
- The **max_depth=3** constraint produces an interpretable tree: top splits typically involve sex and passenger class, consistent with known Titanic survival patterns.

### Example Tree Structure

```
Node: 'sex_male' == True
├── L: Node: 'pclass' <= 1.50
│   ├── L: Leaf: prediction=1
│   └── R: Leaf: prediction=0
└── R: Node: 'age' <= 2.50
    ├── L: Leaf: prediction=1
    └── R: Leaf: prediction=0
```

## Project Structure

```
ai-s3-decision-tree-misha-lrs2-main/
├── DecisionTree.py   # Node and DecisionTree classes
├── tests.ipynb       # Titanic validation notebook
└── README.md
```

## Author

**Misha Leenders** — AI & Machine Learning portfolio project (2025)
