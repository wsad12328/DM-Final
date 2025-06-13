import os
import pandas as pd
import argparse
import pickle
from Augmentor import DataAugmentor
from preprocess import preprocess_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label', help='Encoding method')
    parser.add_argument('--corruption_rate', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--type', choices=['train', 'test'], required=True, help="Specify train or test file")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, '../../data', args.type + '.csv')

    # 1. 讀取原始資料
    df = pd.read_csv(input_path)
    print("Original data:")
    print(df.head())

    # 2. Augmentation
    augmentor = DataAugmentor(corruption_rate=args.corruption_rate, seed=args.seed)
    df_aug = augmentor(df)
    print("\nAugmented data:")
    print(df_aug.head())

    # 3. Preprocess original
    df_orig_processed, label_encoders_orig = preprocess_data(df, encoding_method=args.encoding, is_train=True)
    print("\nPreprocessed original:")
    print(df_orig_processed.head())

    # 4. Preprocess augmented
    df_aug_processed, label_encoders_aug = preprocess_data(df_aug, encoding_method=args.encoding, is_train=True)
    print("\nPreprocessed augmented:")
    print(df_aug_processed.head())

    # 5. 儲存
    orig_save_path = os.path.join(base_dir, '../../data', f'original_{args.type}_{args.encoding}.csv')
    aug_save_path = os.path.join(base_dir, '../../data', f'augmented_{args.type}_{args.encoding}.csv')
    df_orig_processed.to_csv(orig_save_path, index=False)
    df_aug_processed.to_csv(aug_save_path, index=False)
    print(f"\nPreprocessed original saved to: {orig_save_path}")
    print(f"Preprocessed augmented saved to: {aug_save_path}")