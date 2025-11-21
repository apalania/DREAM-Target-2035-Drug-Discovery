import numpy as np
import time
import argparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score, f1_score, average_precision_score, balanced_accuracy_score

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--x_path', required=True, help='Path to features .npy file')
parser.add_argument('--y_path', required=True, help='Path to labels .npy file')
parser.add_argument('--name', required=True, help='Name of the dataset for logging')
args = parser.parse_args()

X_path = args.x_path
y_path = args.y_path
name = args.name

print(f"\n===== Running HPO for: {name} =====")
print(f"Loading data:\n - X: {X_path}\n - y: {y_path}")

X = np.load(X_path)
y = np.load(y_path)
print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

# ---------------------- Sampling and Model Setup ----------------------

RANDOM_STATE = 42

# Use 25% of the dataset for HPO
_, X_hpo_sample, _, y_hpo_sample = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

print(f"Using {X_hpo_sample.shape[0]} samples for HPO")

estimator_hgb = HistGradientBoostingClassifier(
    class_weight='balanced',
    early_stopping=True,
    tol=1e-5,
    learning_rate=0.05,
    max_bins=25,
    n_iter_no_change=5,
    validation_fraction=0.05,
    random_state=RANDOM_STATE
)

param_dist = {
    'max_iter': [500, 700, 1000],
    'max_depth': [10, 20, 30],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# ---------------------- Run HPO for Multiple Metrics ----------------------

scorers = {
    "Balanced Accuracy": make_scorer(balanced_accuracy_score),
    "F1 Score": make_scorer(f1_score, average='macro'),
    "Average Precision": make_scorer(average_precision_score, average='macro'),
    "Recall Score": make_scorer(recall_score, average='macro'),
}

for scorer_name, scorer in scorers.items():
    print(f"\n[{name}] Running HPO optimizing: {scorer_name}")
    random_search = RandomizedSearchCV(
        estimator=estimator_hgb,
        param_distributions=param_dist,
        n_iter=10,
        scoring=scorer,
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    start_time = time.time()
    random_search.fit(X_hpo_sample, y_hpo_sample)
    end_time = time.time()

    print(f"[{name}] {scorer_name} - HPO completed in {end_time - start_time:.2f} seconds")
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best {scorer_name} (CV): {random_search.best_score_}")
