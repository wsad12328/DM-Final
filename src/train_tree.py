import os
import argparse
from sklearn.model_selection import StratifiedKFold
from utils.io_utils import load_preprocessed_data
from train_xgboost import train_xgboost
from train_catboost import train_catboost
from train_lightgbm import train_lightgbm
from utils.feature_utils import restore_data_types
from train_randomforest import train_random_forest


import pickle

def train_and_evaluate_kfold(X, Y, categorical_columns, model_name, n_splits=10, model_save_dir="../models", encoding_method='label'):
    os.makedirs(model_save_dir, exist_ok=True)
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_index, (train_indices, valid_indices) in enumerate(stratified_kfold.split(X, Y)):
        print(f"\n================ Fold {fold_index} ================")
        X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
        Y_train, Y_valid = Y.iloc[train_indices], Y.iloc[valid_indices]
        # 為每個 fold 創建獨立的模型保存路徑
        fold_model_save_path = os.path.join(model_save_dir, f"{model_name}_{encoding_method}_fold_{fold_index}.pkl")

        if model_name == 'xgboost':
            map3_score_fold = train_xgboost(
                X_train, Y_train, X_valid, Y_valid, fold_model_save_path
            )
        elif model_name == 'catboost':
            map3_score_fold = train_catboost(
                X_train, Y_train, X_valid, Y_valid, categorical_columns, fold_model_save_path
            )
        elif model_name == 'lightgbm':
            map3_score_fold = train_lightgbm(
                X_train, Y_train, X_valid, Y_valid, categorical_columns, fold_model_save_path
            )
        elif model_name == 'random_forest':
            map3_score_fold = train_random_forest(
                X_train, Y_train, X_valid, Y_valid, categorical_columns, fold_model_save_path
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        print(f"📊 Fold {fold_index} MAP@3: {map3_score_fold:.5f}")
        print(f"💾 Fold {fold_index} model saved to: {fold_model_save_path}")

    print(f"\n🏆 All {n_splits} fold models saved in {model_save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label', help="Encoding method")
    parser.add_argument('--model', choices=['xgboost', 'random_forest', 'lightgbm', 'catboost'], default='xgboost', help="Model name")
    parser.add_argument('--kfold', action='store_true', help="Use KFold cross-validation", default=False)
    parser.add_argument('--n_splits', type=int, default=5, help="Number of KFold splits")
    parser.add_argument('--model_save_dir', type=str, default='models', help="Directory to save trained models")
    args = parser.parse_args()

    encoding_method = args.encoding
    model_name = args.model

    df = load_preprocessed_data(data_type='train', encoding_method=encoding_method)

    # 修復數據類型（這是關鍵修復）
    cols_info_path = os.path.join(os.path.dirname(__file__), f'../data/cols_info_train_{args.encoding}.pkl')
    df = restore_data_types(df, cols_info_path)

    X = df.drop(columns=['id', 'Fertilizer Name'])
    Y = df['Fertilizer Name']
    num_classes = Y.nunique()
    print(f"Number of classes: {num_classes}")
    assert len(Y) > 0, "No samples found in labels. Please check your preprocessed data."

    with open(cols_info_path, 'rb') as f:
        cols_info = pickle.load(f)

    categorical_columns = cols_info['cat_cols']

    if model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
        train_and_evaluate_kfold(
            X, Y, categorical_columns, model_name,
            n_splits=args.n_splits, model_save_dir=args.model_save_dir, encoding_method=encoding_method
        )
    else:
        print("Other models not implemented in this snippet.")

if __name__ == '__main__':
    main()