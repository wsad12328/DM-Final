from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from utils.evaluate import mapk

def train_random_forest(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    categorical_columns,
    model_save_path
):
    """
    Train Random Forest model on GPU with sklearn API (RandomForestClassifier).
    
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
        'n_estimators': 150,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': 25, 
    }

    model = RandomForestClassifier(**param)

    model.fit(X_train, Y_train)

    Y_valid_proba = model.predict_proba(X_valid)
    top3_predictions_fold = np.argsort(Y_valid_proba, axis=1)[:, -3:][:, ::-1]
    
    map3_score_fold = mapk(Y_valid.values, top3_predictions_fold, k=3)
    
    joblib.dump(model, model_save_path)
    
    return map3_score_fold