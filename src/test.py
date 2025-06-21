import os
import pandas as pd
import numpy as np
import argparse
import torch
import joblib
import pickle
from utils.io_utils import load_preprocessed_data, load_model, load_label_encoders
from models.MLP import MLP
from utils.feature_utils import get_cat_cardinalities

NUM_CLASS = 7  # Number of fertilizer classes

def load_lightgbm_model(model_path):
    """載入 LightGBM 模型"""
    with open(model_path, 'rb') as f:
        model_data = joblib.load(f)
    
    if isinstance(model_data, dict):
        return model_data['model'], model_data.get('label_encoder', None)
    else:
        return model_data, None

def predict_lightgbm(model, X_test, cat_cols, label_encoder=None):
    """LightGBM 預測（處理分類特徵）"""
    X_test_lgb = X_test.copy()
    for col in cat_cols:
        if col in X_test_lgb.columns:
            X_test_lgb[col] = X_test_lgb[col].astype('category')
    
    try:
        proba = model.predict_proba(X_test_lgb)
        return proba
    except Exception as e:
        print(f"⚠️ LightGBM prediction error: {e}")
        # Fallback: 不使用分類特徵
        try:
            proba = model.predict_proba(X_test)
            return proba
        except Exception as e2:
            print(f"❌ LightGBM prediction failed completely: {e2}")
            raise e2

def predict_mlp(encoding_method, X_test, num_cols, cat_cols, fertilizer_le, n_folds=5):
    """MLP 預測 - 集成多個 fold 模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_x = X_test[num_cols].values.astype(np.float32)
    cat_x = X_test[cat_cols].values.astype(np.int64)
    numeric_num_features = num_x.shape[1]
    cat_cardinalities = get_cat_cardinalities(cat_x)
    output_dim = len(fertilizer_le.classes_)

    pred_prob = np.zeros((len(X_test), output_dim))
    model_count = 0
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for fold in range(n_folds):
        model_path = os.path.join(base_dir, '..', 'models', f'mlp_{encoding_method}_fold{fold+1}.pth')
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            continue
            
        print(f"Loading MLP model from fold {fold+1}: {model_path}")
        
        try:
            # 載入模型
            model = MLP(numeric_num_features, cat_cardinalities, output_dim)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(device)
            model.eval()

            # 預測
            with torch.no_grad():
                num_x_tensor = torch.tensor(num_x, dtype=torch.float32).to(device)
                cat_x_tensor = torch.tensor(cat_x, dtype=torch.long).to(device)
                logits = model(num_x_tensor, cat_x_tensor)
                fold_proba = torch.softmax(logits, dim=1).cpu().numpy()
            
            pred_prob += fold_proba
            model_count += 1
            print(f"✅ Fold {fold+1} predictions added")
            
        except Exception as e:
            print(f"❌ Fold {fold+1} prediction failed: {e}")
            continue
    
    if model_count == 0:
        raise ValueError("No MLP fold models found!")
    
    # 平均預測結果
    pred_prob /= model_count
    print(f"📊 Ensembled MLP predictions from {model_count} models")
    
    return pred_prob

def ensemble_tree_models(model_name, encoding_method, X_test, cat_cols, n_folds, base_dir):
    """集成樹模型預測"""
    pred_prob = np.zeros((len(X_test), NUM_CLASS))
    model_count = 0
    
    for fold in range(n_folds):
        model_path = os.path.join(base_dir, '..', 'models', f'{model_name}_{encoding_method}_fold_{fold}.pkl')
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            continue
            
        print(f"Loading model from fold {fold}: {model_path}")
        
        try:
            if model_name == 'lightgbm':
                model, label_encoder = load_lightgbm_model(model_path)
                fold_proba = predict_lightgbm(model, X_test, cat_cols, label_encoder)
            else:
                model = joblib.load(model_path)
                fold_proba = model.predict_proba(X_test)
            
            pred_prob += fold_proba
            model_count += 1
            print(f"✅ Fold {fold} predictions added")
            
        except Exception as e:
            print(f"❌ Fold {fold} prediction failed: {e}")
            continue
    
    if model_count == 0:
        raise ValueError("No fold models found!")
    
    # 平均預測結果
    pred_prob /= model_count
    print(f"📊 Ensembled predictions from {model_count} models")
    
    return pred_prob

def get_top3_predictions(proba):
    """獲取 top3 預測索引"""
    return np.argsort(proba, axis=1)[:, ::-1][:, :3]

def decode_predictions(top3_idx, model_name, encoding_method, base_dir):
    """解碼預測結果為類別名稱"""
    if model_name == 'lightgbm':
        # LightGBM 特殊處理
        sample_model_path = os.path.join(base_dir, '..', 'models', f'{model_name}_{encoding_method}_fold_0.pkl')
        model, label_encoder = load_lightgbm_model(sample_model_path)
        
        if label_encoder is not None:
            # 使用標籤編碼器解碼
            return label_encoder.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)
        else:
            # 使用模型的 classes_ 屬性
            return model.classes_[top3_idx]
    else:
        # 其他模型
        sample_model_path = os.path.join(base_dir, '..', 'models', f'{model_name}_{encoding_method}_fold_0.pkl')
        model = joblib.load(sample_model_path)
        return model.classes_[top3_idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label', help="Encoding method")
    parser.add_argument('--model', choices=['xgboost', 'random_forest', 'mlp', 'catboost', 'lightgbm'], default='xgboost', help="Model name")
    parser.add_argument('--n_folds', type=int, default=5, help="Number of folds to ensemble")
    args = parser.parse_args()

    # 載入數據和編碼器
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_test = load_preprocessed_data(data_type='test', encoding_method=args.encoding)
    label_encoders = load_label_encoders()
    fertilizer_le = label_encoders['Fertilizer Name']

    # 載入列信息
    cols_info_path = os.path.join(base_dir, f'../data/cols_info_test_{args.encoding}.pkl')
    with open(cols_info_path, 'rb') as f:
        cols = pickle.load(f)
    num_cols = cols['num_cols']
    cat_cols = cols['cat_cols']

    # 準備測試數據
    id_col = df_test['id']
    X_test = df_test.drop(columns=['id'])

    # 根據模型類型進行預測
    if args.model == 'mlp':
        proba = predict_mlp(args.encoding, X_test, num_cols, cat_cols, fertilizer_le)
        top3_idx = get_top3_predictions(proba)
        top3_labels = fertilizer_le.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)
        model_count = 1
    else:
        # 樹模型集成預測
        pred_prob = ensemble_tree_models(args.model, args.encoding, X_test, cat_cols, args.n_folds, base_dir)
        top3_idx = get_top3_predictions(pred_prob)
        top3_preds = decode_predictions(top3_idx, args.model, args.encoding, base_dir)
        
        # 最終解碼為肥料名稱
        if isinstance(top3_preds[0, 0], str):
            # 如果已經是字符串，直接使用
            top3_labels = top3_preds
        else:
            # 否則使用 fertilizer_le 解碼
            top3_labels = fertilizer_le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)
        
        model_count = args.n_folds

    # 生成提交文件
    submission = pd.DataFrame({
        'id': id_col,
        'Fertilizer Name': [' '.join(preds) for preds in top3_labels]
    })

    # 保存結果
    suffix = f"_ensemble_{model_count}folds" if model_count > 1 else ""
    output_path = os.path.join(base_dir, '../result', f'submission_{args.model}_{args.encoding}{suffix}.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"✅ Submission saved to: {output_path}")

if __name__ == '__main__':
    main()