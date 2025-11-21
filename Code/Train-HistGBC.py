import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    classification_report
)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

# Full dataset configuration with fixed parameters
dataset_config = {
    'dataset1': {
        'evalset': 'TestX_with_binary.npy',
        'testset': 'TestSet4Prediction_with_binary.npy',
        'features': 'X_features_with_MW_ALOGP_binary.npy',
        'labels_eval': '14_Testy.npy',
        'max_depth': 6,
        'max_iter': 600,
        'learning_rate': 0.03
    },
    'dataset2': {
        'evalset': 'TestX_with_cut_bins.npy',
        'testset': 'TestSet4Prediction_with_cut_bins.npy',
        'features': 'X_features_with_MW_ALOGP_binned.npy',
        'labels_eval': '14_Testy.npy',
        'max_depth': 10,
        'max_iter': 1000,
        'learning_rate': 0.05
    },
    'dataset3': {
        'evalset': 'Test1_PCA99.npy',
        'testset': 'Test2_PCA99.npy',
        'features': 'X_pca_standard_99.npy',
        'labels_eval': '14_Testy.npy',
        'max_depth': 7,
        'max_iter': 750,
        'learning_rate': 0.07
    },
    'dataset4': {
        'evalset': 'Test1_PCA95.npy',
        'testset': 'Test2_PCA95.npy',
        'features': 'X_pca_standard_95.npy',
        'labels_eval': '14_Testy.npy',
        'max_depth': 8,
        'max_iter': 900,
        'learning_rate': 0.04
    },
    'dataset5': {
        'evalset': 'TestX_scaled.npy',
        'testset': 'TestSet4Prediction_f_scaled.npy',
        'features': 'X_standard_scaled.npy',
        'labels_eval': '14_Testy.npy',
        'max_depth': 5,
        'max_iter': 850,
        'learning_rate': 0.06
    }
}


def evaluate_model(y_true, y_pred, y_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'AUC-PR': average_precision_score(y_true, y_proba),
        'Confusion Matrix': confusion_matrix(y_true, y_pred)
    }

def train_and_evaluate(dataset_name, config):
    print(f"\n=== Processing {dataset_name} ===")
    
    # Load data
    X = np.load(config['features'], allow_pickle=True)
    y = np.load('y_labels.npy', allow_pickle=True)
    X_eval = np.load(config['evalset'], allow_pickle=True)
    y_eval = np.load(config['labels_eval'], allow_pickle=True)

   

    # Initialize CatBoost model with fixed parameters
    model = HistGradientBoostingClassifier(
    class_weight='balanced',
    early_stopping=True,
    tol=1e-5,
    learning_rate=config['learning_rate'],
    max_bins=25,
    n_iter_no_change=5,
    validation_fraction=0.05,
    max_iter=config['max_iter'],
    max_depth=config['max_depth'],
    random_state=42
)



    # 5-Fold Cross-Validation with AUC-PR
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = {metric: [] for metric in [
        'Accuracy', 'Balanced Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC-PR'
    ]}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/5 ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        fold_metrics = evaluate_model(y_val, y_pred, y_proba)
        
        for metric in cv_metrics:
            cv_metrics[metric].append(fold_metrics[metric])

    # Print CV results
    print("\n=== 5-Fold Cross-Validation Metrics ===")
    for metric, values in cv_metrics.items():
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Full dataset training
    print("\nTraining on full dataset...")
    model.fit(X, y)
    y_pred_full = model.predict(X)
    y_proba_full = model.predict_proba(X)[:, 1]
    
    # Full dataset evaluation
    full_metrics = evaluate_model(y, y_pred_full, y_proba_full)
    print("\n=== Full Dataset Metrics ===")
    for metric, value in full_metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value:.4f}")
    print(f"Confusion Matrix:\n{full_metrics['Confusion Matrix']}")

    # Evaluation set metrics
    y_eval_pred = model.predict(X_eval)
    y_eval_proba = model.predict_proba(X_eval)[:, 1]
    eval_metrics = evaluate_model(y_eval, y_eval_pred, y_eval_proba)
    print("\n=== Evaluation Set Metrics ===")
    for metric, value in eval_metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value:.4f}")
    print(f"Confusion Matrix:\n{eval_metrics['Confusion Matrix']}")

    # Save predictions as .npy with index, probability, and class
    X_test = np.load(config['testset'], allow_pickle=True)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    df_pred = pd.DataFrame({
        'Index': np.arange(1, len(test_proba) + 1),
        'Predict_Proba': test_proba,
        'Predict_Class': test_pred
    })
    np.save(f"hist_{dataset_name}_predictions.npy", df_pred.to_numpy())
    print(f"\nPredictions saved to {dataset_name}_predictions.npy")

if __name__ == "__main__":
    for dataset_name, config in dataset_config.items():
        train_and_evaluate(dataset_name, config)