import pandas as pd
import os
import argparse
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

TYPE = None
ENCODE = None

def preprocess_data(df, encoding_method='label', is_train=True):    
    numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != 'id']

    for col in numeric_cols:
        df[f'{col}_Binned'] = df[col].astype(str).astype('category')

    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != "Fertilizer Name"]
    label_encoders = {}

    # 類別欄位編碼
    if encoding_method == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    elif encoding_method == 'onehot':
        df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)
    else:
        raise ValueError("encoding_method must be 'label' or 'onehot'")

    # 目標欄位編碼
    if is_train and 'Fertilizer Name' in df.columns:
        target_le = LabelEncoder()
        df['Fertilizer Name'] = target_le.fit_transform(df['Fertilizer Name'])
        label_encoders['Fertilizer Name'] = target_le

    # get new categorical columns if one-hot encoding is used
    if encoding_method == 'onehot':
        cat_cols = [col for col in df.columns if col.startswith(tuple(cat_cols))]

    for col in cat_cols:
        df[col] = df[col].astype("category")

    print(f"Categorical columns after preprocessing: {cat_cols}")
    print(len(cat_cols), "categorical columns found.")

    # save the categorical and numerical columns
    cols_info = {
        'cat_cols': cat_cols,
        'num_cols': numeric_cols
    }
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cols_info_path = os.path.join(base_dir, f'../../data/cols_info_{TYPE}_{ENCODE}.pkl')
    with open(cols_info_path, 'wb') as f:
        pickle.dump(cols_info, f)
    print(f"Column information saved to: {cols_info_path}")

    return df, label_encoders

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['train', 'test'], required=True, help="Specify train or test file")
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label', help="Encoding method")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'train.csv' if args.type == 'train' else 'test.csv'
    file_path = os.path.join(base_dir, '../../data', file_name)

    TYPE = args.type
    ENCODE = args.encoding

    df = pd.read_csv(file_path)
    print("Before preprocessing:")
    print(df.head())

    is_train = args.type == 'train'
    df_processed, label_encoders = preprocess_data(df, encoding_method=args.encoding, is_train=is_train)

    print("\nAfter preprocessing:")
    print(df_processed.head())

    filename = f'preprocessed_{args.type}_{args.encoding}.csv'
    save_path = os.path.join(base_dir, '../../data', filename)
    df_processed.to_csv(save_path, index=False)
    print(f"\nPreprocessed data saved to: {save_path}")

    # 儲存 label encoder
    if is_train:
        encoders_path = os.path.join(base_dir, '../../data', f'label_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved to: {encoders_path}")