import numpy as np
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    recall_score, precision_score, average_precision_score,
    balanced_accuracy_score
)

# Set random state
RANDOM_STATE = 42

# Model parameters
model_params = {
    "n_estimators": 700,
    "max_depth": 30,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": 'sqrt',
    "class_weight": 'balanced_subsample',
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

def run_training(datasets):
    for x_path, y_path, description in datasets:
        print(f"\n=== Training on: {description} ===")

        X = np.load(x_path)
        y = np.load(y_path)

        # Ensure y is 1D (convert from one-hot if needed)
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = y.argmax(axis=1)

        model = RandomForestClassifier(**model_params)
        model.fit(X, y)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)

        model_filename = f"rf_model_init_{description.lower().replace('%', '').replace(' ', '_').replace('__', '_')}.pkl"
        joblib.dump(model, model_filename)
        print(f"Model saved to: {model_filename}")
        print("Classification Report:")
        print(classification_report(y, y_pred, digits=4))

        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

        print(f"Balanced Accuracy: {balanced_accuracy_score(y, y_pred):.4f}")
        print(f"F1 Score (macro): {f1_score(y, y_pred, average='macro'):.4f}")
        print(f"Precision (macro): {precision_score(y, y_pred, average='macro'):.4f}")
        print(f"Recall (macro): {recall_score(y, y_pred, average='macro'):.4f}")

        # Use only positive class probabilities for binary classification
        if y_prob.shape[1] == 2:
            y_score = y_prob[:, 1]
        else:
            y_score = y_prob  # For multiclass, y_score remains as is

        print(f"Average Precision Score (macro): {average_precision_score(y, y_score, average='macro'):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest on multiple datasets")
    parser.add_argument(
        '--datasets', 
        type=str, 
        required=True, 
        help=(
            "Datasets list as: x_path,y_path,description; x_path,y_path,description; ..."
            "Example: ./X_pca_95.npy,./y.npy,PCA95;./X_scaled.npy,./y.npy,std_scaled"
        )
    )
    args = parser.parse_args()

    # Parse datasets argument into list of tuples
    datasets = []
    for dataset_str in args.datasets.split(';'):
        parts = dataset_str.strip().split(',')
        if len(parts) != 3:
            raise ValueError(f"Each dataset must have 3 parts: x_path,y_path,description. Got: {dataset_str}")
        datasets.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))

    run_training(datasets)
