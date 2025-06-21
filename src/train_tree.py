import os
import argparse
import time
from datetime import datetime
import numpy as np
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
    
    fold_times = []  # 記錄每個 fold 的時間
    fold_scores = []  # 記錄每個 fold 的分數

    for fold_index, (train_indices, valid_indices) in enumerate(stratified_kfold.split(X, Y)):
        fold_start_time = time.time()
        print(f"\n================ Fold {fold_index+1}/{n_splits} ================")
        X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
        Y_train, Y_valid = Y.iloc[train_indices], Y.iloc[valid_indices]
        # 為每個 fold 創建獨立的模型保存路徑
        fold_model_save_path = os.path.join(model_save_dir, f"{model_name}_{encoding_method}_fold_{fold_index+1}.pkl")

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

        # 計算並記錄每個 fold 的時間
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_times.append(fold_duration)
        fold_scores.append(map3_score_fold)

        print(f"📊 Fold {fold_index+1} MAP@3: {map3_score_fold:.5f}")
        print(f"⏱️  Fold {fold_index+1} completed in: {fold_duration/60:.2f} minutes")
        print(f"💾 Fold {fold_index+1} model saved to: {fold_model_save_path}")

    print(f"\n🏆 All {n_splits} fold models saved in {model_save_dir}")
    
    # 返回時間和分數統計
    return fold_times, fold_scores

def main():
    # 記錄開始時間
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
        fold_times, fold_scores = train_and_evaluate_kfold(
            X, Y, categorical_columns, model_name,
            n_splits=args.n_splits, model_save_dir=args.model_save_dir, encoding_method=encoding_method
        )
        
        # 計算總時間和平均時間
        total_time = time.time() - start_time
        end_datetime = datetime.now()
        average_time = sum(fold_times) / args.n_splits
        
        print(f"\n{'='*50}")
        print(f"🏁 Training Summary")
        print(f"{'='*50}")
        print(f"All folds MAP@3: {fold_scores}")
        print(f"Average MAP@3: {np.mean(fold_scores):.4f}")
        print(f"Standard deviation: {np.std(fold_scores):.4f}")
        print(f"\n⏱️  Time Summary:")
        print(f"Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ended at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"Average time per fold: {average_time/60:.2f} minutes")
        print(f"Fold times: {[f'{t/60:.2f}min' for t in fold_times]}")
        
        # 保存時間資訊到 txt 文件
        results_dir = os.path.join(os.path.dirname(__file__), '../result')
        os.makedirs(results_dir, exist_ok=True)
       
        time_file = os.path.join(results_dir, f'training_time_{model_name}_{encoding_method}.txt')
        
        with open(time_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Encoding: {encoding_method}\n")
            f.write(f"Number of Folds: {args.n_splits}\n")
            f.write(f"\nTraining Time Summary:\n")
            f.write(f"Average Time Per Fold: {average_time:.2f} seconds ({average_time/60:.2f} minutes)\n")
            f.write(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
            f.write(f"\nIndividual Fold Times:\n")
            for i, fold_time in enumerate(fold_times):
                f.write(f"Fold {i+1}: {fold_time:.2f} seconds ({fold_time/60:.2f} minutes)\n")
        
        print(f"Training time information saved to: {time_file}")
        
    else:
        print("Other models not implemented in this snippet.")

if __name__ == '__main__':
    main()