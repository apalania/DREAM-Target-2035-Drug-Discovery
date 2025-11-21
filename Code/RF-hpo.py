import numpy as np
import time
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, average_precision_score, balanced_accuracy_score

# ---------------------- Parse Command Line Arguments ----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--x_path', required=True, help='Path to features .npy file')
parser.add_argument('--y_path', required=True, help='Path to labels .npy file')
parser.add_argument('--name', required=True, help='Name for the dataset')

args = parser.parse_args()
X_path = args.x_path
y_path = args.y_path
name = args.name

# ---------------------- Run HPO Function ----------------------

def run_hpo(X_path, y_path, name):
    print(f"\n===== Running HPO for {name} =====")
    print(f"Loading X from: {X_path}")
    print(f"Loading y from: {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    RANDOM_STATE = 42

    _, X_hpo_sample, _, y_hpo_sample = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE
    )

    estimator_rf = RandomForestClassifier(
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.2,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )

    param_dist = {
        'n_estimators': [700, 1000, 1200],
        'max_depth': [30, 50]
    }

    

    scorers = {
        'F1 Score': make_scorer(f1_score, average='macro'),
        'Recall': make_scorer(recall_score, average='macro'),
        'Average Precision': make_scorer(average_precision_score, average='macro'),
        'Balanced Accuracy': make_scorer(balanced_accuracy_score),
    }

    for metric_name, scorer in scorers.items():
        print(f"\n[{name}] Running HPO optimizing: {metric_name}")
        random_search = RandomizedSearchCV(
            estimator=estimator_rf,
            param_distributions=param_dist,
            n_iter=10,
            scoring=scorer,
            cv=3,
            verbose=2,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        start_time = time.time()
        random_search.fit(X_hpo_sample, y_hpo_sample)
        end_time = time.time()

        print(f"[{name}] {metric_name} - HPO completed in {end_time - start_time:.2f} seconds")
        print(f"Best Parameters ({metric_name}): {random_search.best_params_}")
        print(f"Best {metric_name} (CV): {random_search.best_score_:.4f}")

# ---------------------- Execute ----------------------

run_hpo(X_path, y_path, name)
