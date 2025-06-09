import pandas as pd
import os
import argparse
from sklearn.preprocessing import LabelEncoder
import pickle

def preprocess_data(df, encoding_method='label', is_train=True):
    cat_cols = ['Soil Type', 'Crop Type']
    num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

    label_encoders = {}
    if encoding_method == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        # 標籤單獨編碼
        if is_train and 'Fertilizer Name' in df.columns:
            target_le = LabelEncoder()
            df['Fertilizer Name'] = target_le.fit_transform(df['Fertilizer Name'])
            label_encoders['Fertilizer Name'] = target_le
    elif encoding_method == 'onehot':
        df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)
        # 標籤單獨編碼
        if is_train and 'Fertilizer Name' in df.columns:
            target_le = LabelEncoder()
            df['Fertilizer Name'] = target_le.fit_transform(df['Fertilizer Name'])
            label_encoders['Fertilizer Name'] = target_le
    else:
        raise ValueError("encoding_method must be 'label' or 'onehot'")

    return df, label_encoders

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['train', 'test'], required=True, help="Specify train or test file")
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label', help="Encoding method")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'train.csv' if args.type == 'train' else 'test.csv'
    file_path = os.path.join(base_dir, '../../data', file_name)

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

    if args.encoding == 'label' and is_train:
        encoders_path = os.path.join(base_dir, '../../data', 'label_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved to: {encoders_path}")