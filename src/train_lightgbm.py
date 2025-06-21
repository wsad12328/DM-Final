from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import numpy as np
from utils.evaluate import mapk
import joblib

def train_lightgbm(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    categorical_columns,
    model_save_path
):
    """
    Train LightGBM model on GPU with sklearn API (LGBMClassifier).
    
    Args
    ----
    X_train, Y_train : training data
    X_valid, Y_valid : validation data
    categorical_columns : list[str]  — column names that are categorical
    model_save_path : str           — .pkl path to save the model
    
    Returns
    -------
    map3_score : float  — MAP@3 score on the validation set
    """

    param = {
        # core
        'objective':        'multiclass',
        'num_class':        7,
        'n_estimators':     10000,
        'max_depth':        5,
        'colsample_bytree': 0.5,
        'random_state':     42,
        'importance_type':  'gain',
        'verbose':          1,        # 改為 1 顯示訓練過程
        'verbosity':        1,        # 改為 1 或移除這行
        'n_jobs':           25,       # 使用所有可用的 CPU 核心
        # gpu
        # 'device':           'gpu',
    }

    model = LGBMClassifier(**param)

    model.fit(X_train, Y_train,
              eval_set=(X_valid, Y_valid),
              feature_name=X_train.columns.tolist(),
              categorical_feature=categorical_columns,
              callbacks=[
                  early_stopping(stopping_rounds=100, verbose=True),  # 改為 True
                  log_evaluation(period=200)  # 每 50 輪顯示一次進度
              ])

    Y_valid_proba = model.predict_proba(X_valid)
    top3_predictions_fold = np.argsort(Y_valid_proba, axis=1)[:, -3:][:, ::-1] 
    map3_score_fold = mapk(Y_valid.values, top3_predictions_fold, k=3)
    joblib.dump(model, model_save_path)
    return map3_score_fold