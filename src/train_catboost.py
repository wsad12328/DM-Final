import joblib
import numpy as np
from catboost import CatBoostClassifier, Pool
from collections import Counter
from sklearn.metrics import accuracy_score
from src.utils.evaluate import mapk

def train_catboost(X_train, Y_train, X_valid, Y_valid, categorical_columns, model_save_path):
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
    Y_valid_pred = model.predict(X_valid)
    Y_valid_proba = model.predict_proba(X_valid)
    accuracy = accuracy_score(Y_valid, Y_valid_pred)
    top3_predictions_fold = np.argsort(Y_valid_proba, axis=1)[:, ::-1][:, :3]
    map3_score_fold = mapk(Y_valid.values, top3_predictions_fold, k=3)
    joblib.dump(model, model_save_path)
    return accuracy, map3_score_fold