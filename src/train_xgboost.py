import joblib
import numpy as np
from xgboost import XGBClassifier
from utils.evaluate import mapk

def train_xgboost(X_train, Y_train, X_valid, Y_valid, model_save_path):
    params = {
        'objective': 'multi:softprob',  
        'num_class': 7, 
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'max_bin': 128,
        'colsample_bytree': 0.3, 
        'colsample_bylevel': 1,  
        'colsample_bynode': 1,  
        'tree_method': 'hist',  
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'enable_categorical':True,
        'n_estimators':10000,
        'early_stopping_rounds':50,
        'n_jobs': 25,
    }
    model = XGBClassifier(**params)
    model.fit(
        X_train,
        Y_train,
        eval_set=[(X_valid, Y_valid)],
        verbose=200,
    )
    Y_valid_proba = model.predict_proba(X_valid)
    top3_predictions_fold = np.argsort(Y_valid_proba, axis=1)[:, -3:][:, ::-1] 
    map3_score_fold = mapk(Y_valid.values, top3_predictions_fold, k=3)
    joblib.dump(model, model_save_path)
    return map3_score_fold