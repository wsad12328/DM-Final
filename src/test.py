import os
import pandas as pd
import numpy as np
import argparse
import torch
from utils.io_utils import load_preprocessed_data, load_model, load_label_encoders
from models.MLP import MLP
from utils.feature_utils import get_cat_cols, get_cat_cardinalities
import pickle

cols = pickle.load(open(os.path.join(os.path.dirname(__file__), '../data/cols_info.pkl'), 'rb'))
num_cols = cols['num_cols']
cat_cols = cols['cat_cols']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label', help="Encoding method")
    parser.add_argument('--model', choices=['xgboost', 'random_forest', 'mlp'], default='xgboost', help="Model name")
    args = parser.parse_args()

    encoding_method = args.encoding
    model_name = args.model

    base_dir = os.path.dirname(os.path.abspath(__file__))

    df_test = load_preprocessed_data(data_type='test', encoding_method=encoding_method)
    label_encoders = load_label_encoders()
    fertilizer_le = label_encoders['Fertilizer Name']

    # === 特徵選擇（排除 id）===
    id_col = df_test['id']
    X_test = df_test.drop(columns=['id'])

    if model_name == 'mlp':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # cat_cols = get_cat_cols(X_test)
        num_x = X_test[num_cols].values.astype(np.float32)
        cat_x = X_test[cat_cols].values.astype(np.int64)
        numeric_num_features = num_x.shape[1]
        cat_cardinalities = get_cat_cardinalities(cat_x)
        output_dim = len(fertilizer_le.classes_)

        # 利用 io_utils 的 load_model 載入 state_dict
        model = MLP(numeric_num_features, cat_cardinalities, output_dim)
        state_dict = load_model(model_name='mlp', encoding_method=encoding_method)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        with torch.no_grad():
            num_x_tensor = torch.tensor(num_x, dtype=torch.float32).to(device)
            cat_x_tensor = torch.tensor(cat_x, dtype=torch.long).to(device)
            logits = model(num_x_tensor, cat_x_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        top3_idx = np.argsort(proba, axis=1)[:, ::-1][:, :3]
        top3_preds = top3_idx
    else:
        model = load_model(model_name=model_name, encoding_method=encoding_method)
        proba = model.predict_proba(X_test)
        top3_idx = np.argsort(proba, axis=1)[:, ::-1][:, :3]
        top3_preds = model.classes_[top3_idx]

    # Decode labels
    top3_labels = fertilizer_le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

    # Build submission dataframe
    submission = pd.DataFrame({
        'id': id_col,
        'Fertilizer Name': [' '.join(preds) for preds in top3_labels]
    })

    # === 輸出 CSV ===
    output_path = os.path.join(base_dir, '../result', f'submission_{args.model}_{args.encoding}.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")