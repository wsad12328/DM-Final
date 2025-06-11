import os
import joblib
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils.io_utils import load_preprocessed_data
from train_xgboost import train_xgboost
from utils.weights import compute_per_instance_weights
from train_catboost import train_catboost

def train_and_evaluate_kfold(X, Y, categorical_columns, model_name, n_splits=10, model_save_dir="../models", encoding_method='label'):
    os.makedirs(model_save_dir, exist_ok=True)
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    best_map3 = -1
    best_model_save_path = os.path.join(model_save_dir, f"{model_name}_{encoding_method}.pkl")

    for fold_index, (train_indices, valid_indices) in enumerate(stratified_kfold.split(X, Y), 1):
        print(f"\n================ Fold {fold_index} ================")
        X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
        Y_train, Y_valid = Y.iloc[train_indices], Y.iloc[valid_indices]

        if model_name == 'xgboost':
            sample_weights = compute_per_instance_weights(Y_train)
            accuracy, map3_score_fold = train_xgboost(
                X_train, Y_train, X_valid, Y_valid, sample_weights, best_model_save_path
            )
        elif model_name == 'catboost':
            accuracy, map3_score_fold = train_catboost(
                X_train, Y_train, X_valid, Y_valid, categorical_columns, best_model_save_path
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        fold_accuracies.append(accuracy)
        print(f"✅ Fold {fold_index} Accuracy: {accuracy:.4f}")
        print(f"📊 Fold {fold_index} MAP@3: {map3_score_fold:.5f}")

        if map3_score_fold > best_map3:
            best_map3 = map3_score_fold
            print(f"🌟 Best model updated and saved to {best_model_save_path}")

    print(f"\n🏆 Best model saved to {best_model_save_path} (Fold with highest accuracy: {best_map3:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label', help="Encoding method")
    parser.add_argument('--model', choices=['xgboost', 'random_forest', 'lightgbm', 'catboost'], default='xgboost', help="Model name")
    parser.add_argument('--kfold', action='store_true', help="Use KFold cross-validation", default=False)
    parser.add_argument('--n_splits', type=int, default=10, help="Number of KFold splits")
    parser.add_argument('--model_save_dir', type=str, default='models', help="Directory to save trained models")
    args = parser.parse_args()

    encoding_method = args.encoding
    model_name = args.model

    df = load_preprocessed_data(data_type='train', encoding_method=encoding_method)
    X = df.drop(columns=['id', 'Fertilizer Name'])
    Y = df['Fertilizer Name']
    num_classes = Y.nunique()
    print(f"Number of classes: {num_classes}")
    assert len(Y) > 0, "No samples found in labels. Please check your preprocessed data."
    categorical_columns = ['Soil Type', 'Crop Type'] if 'Soil Type' in X.columns else []

    if model_name in ['xgboost', 'lightgbm', 'catboost']:
        train_and_evaluate_kfold(
            X, Y, categorical_columns, model_name,
            n_splits=args.n_splits, model_save_dir=args.model_save_dir, encoding_method=encoding_method
        )
    else:
        print("Other models not implemented in this snippet.")

if __name__ == '__main__':
    main()