import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utils.evaluate import mapk

def train_xgboost(X_train, Y_train, X_valid, Y_valid, sample_weights, model_save_path):
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
        early_stopping_rounds=100,
        objective='multi:softprob',
        random_state=13,
        enable_categorical=True,
        tree_method='hist', 
        n_jobs=20,
    )
    model.fit(
        X_train,
        Y_train,
        eval_set=[(X_valid, Y_valid)],
        sample_weight=sample_weights,
        verbose=200,
    )
    Y_valid_pred = model.predict(X_valid)
    Y_valid_proba = model.predict_proba(X_valid)
    accuracy = accuracy_score(Y_valid, Y_valid_pred)
    top3_predictions_fold = np.argsort(Y_valid_proba, axis=1)[:, ::-1][:, :3]
    map3_score_fold = mapk(Y_valid.values, top3_predictions_fold, k=3)
    joblib.dump(model, model_save_path)
    return accuracy, map3_score_fold