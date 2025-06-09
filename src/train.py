import os
import joblib
import argparse
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from evaluate import mapk
from utils.utils import load_preprocessed_data
from collections import Counter

def compute_per_instance_weights(Y):
    """Compute sample weights for each instance to balance classes in the fold."""
    class_counter = Counter(Y)
    max_class_count = max(class_counter.values())
    return Y.map(lambda class_label: max_class_count / class_counter[class_label])

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
        sample_weights = compute_per_instance_weights(Y_train)

        # 模型選擇與訓練
        if model_name == 'xgboost':
            model = XGBClassifier(
                max_depth=12,
                colsample_bytree=0.467,
                subsample=0.86,
                n_estimators=4000,
                learning_rate=0.03,
                gamma=0.26,
                max_delta_step=4,
                reg_alpha=2.7,
                reg_lambda=1.4,
                objective='multi:softprob',
                random_state=13,
                enable_categorical=True,
                tree_method='hist',
                n_jobs=20,
                early_stopping_rounds=150,
            )
            model.fit(
                X_train,
                Y_train,
                sample_weight=sample_weights,
                eval_set=[(X_valid, Y_valid)],

                verbose=200,
            )
        elif model_name == 'lightgbm':
            model = LGBMClassifier(
                objective='multiclass',
                n_estimators=4000,
                learning_rate=0.03,
                max_depth=12,
                subsample=0.86,
                colsample_bytree=0.467,
                reg_alpha=2.7,
                reg_lambda=1.4,
                random_state=13,
                device='gpu'
            )
            model.fit(
                X_train,
                Y_train,
                sample_weight=sample_weights,
                eval_set=[(X_valid, Y_valid)],
                early_stopping_rounds=150,
                verbose=200,
                categorical_feature=categorical_columns
            )
        elif model_name == 'catboost':
            class_counter = Counter(Y_train)
            max_count = max(class_counter.values())
            class_weights_list = [max_count / class_counter[i] for i in sorted(class_counter.keys())]
            cat_features_idx = [X_train.columns.get_loc(col) for col in categorical_columns if col in X_train.columns]
            train_pool = Pool(X_train, Y_train, cat_features=cat_features_idx)
            valid_pool = Pool(X_valid, Y_valid, cat_features=cat_features_idx)
            model = CatBoostClassifier(
                iterations=1440,
                learning_rate=0.07465701859965242,
                depth=7,
                l2_leaf_reg=0.8064965409711105,
                bootstrap_type="Bayesian",
                bagging_temperature=0.34298306326556705,
                random_strength=6.632156583833577,
                class_weights=class_weights_list,
                task_type="GPU",
                devices="0",
                gpu_ram_part=0.8,
                random_seed=42,
                eval_metric="Accuracy",
                early_stopping_rounds=50,
                verbose=200,
            )
            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        Y_valid_pred = model.predict(X_valid)
        Y_valid_proba = model.predict_proba(X_valid)

        accuracy = accuracy_score(Y_valid, Y_valid_pred)
        fold_accuracies.append(accuracy)
        print(f"✅ Fold {fold_index} Accuracy: {accuracy:.4f}")

        # 計算本 fold 的 MAP@3
        top3_predictions_fold = np.argsort(Y_valid_proba, axis=1)[:, ::-1][:, :3]
        map3_score_fold = mapk(Y_valid.values, top3_predictions_fold, k=3)
        print(f"📊 Fold {fold_index} MAP@3: {map3_score_fold:.5f}")

        # Save best model by MAP@3
        if map3_score_fold > best_map3:
            best_map3 = map3_score_fold
            joblib.dump(model, best_model_save_path)
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