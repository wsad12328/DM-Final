import os
import pandas as pd
import numpy as np
import argparse
from utils.io_utils import load_preprocessed_data, load_model, load_label_encoders

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label', help="Encoding method")
    parser.add_argument('--model', choices=['xgboost', 'random_forest'], default='xgboost', help="Model name")
    args = parser.parse_args()

    encoding_method = args.encoding
    model_name = args.model

    base_dir = os.path.dirname(os.path.abspath(__file__))

    df_test = load_preprocessed_data(data_type='test', encoding_method=encoding_method)
    model = load_model(model_name=model_name, encoding_method=encoding_method)
    label_encoders = load_label_encoders()
    fertilizer_le = label_encoders['Fertilizer Name']

    # === 特徵選擇（排除 id）===
    id_col = df_test['id']
    X_test = df_test.drop(columns=['id'])

    # === 預測 top-3 類別 ===
    proba = model.predict_proba(X_test)
    top3_idx = np.argsort(proba, axis=1)[:, ::-1][:, :3]
    top3_preds = model.classes_[top3_idx]  # top3 是數字編碼

    # Decode labels
    top3_labels = fertilizer_le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

    # Build submission dataframe
    submission = pd.DataFrame({
        'id': df_test['id'],  # if 'id' exists in test_df
        'Fertilizer Name': [' '.join(preds) for preds in top3_labels]
    })

    # === 輸出 CSV ===
    output_path = os.path.join(base_dir, '../result', f'submission_{args.model}_{args.encoding}.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")