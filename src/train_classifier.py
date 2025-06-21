import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
from models.SCARF import Encoder
from dataset.SCARFDataset import SCARFDataset
from utils.feature_utils import OneHotEncoderWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def extract_scarf_features(encoder, dataset, device, batch_size=512):
    """從 SCARF encoder 中提取特徵"""
    encoder.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for orig, _, labels in tqdm(dataloader, desc="Extracting SCARF features"):
            orig = orig.float().to(device)
            features = encoder(orig)  # 獲取 encoder 的輸出
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.vstack(all_features)
    labels = np.hstack(all_labels)
    
    return features, labels

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_encoder', type=str, required=True, 
                       help='Path to pretrained SCARF encoder weights')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, '../data/train.csv')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 讀取資料
    df = pd.read_csv(train_csv)
    
    cat_cols = ["Soil Type", "Crop Type","Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
    num_cols = []
    
    # 建立 encoder wrapper
    encoder_wrapper = OneHotEncoderWrapper(cat_cols, num_cols)
    encoder_wrapper.fit(df)
    
    # 建立 dataset
    dataset = SCARFDataset(
        df=df,
        augmentor=None,  # 不需要 augmentation
        encoder=encoder_wrapper,
        cat_cols=cat_cols,
        num_cols=num_cols,
        label_col='Fertilizer Name'
    )
    
    # 載入預訓練的 SCARF encoder
    sample_orig, _, _ = dataset[0]
    input_dim = sample_orig.shape[0]
    
    encoder_path = os.path.join(base_dir, args.pretrained_encoder)

    encoder = Encoder(input_dim=input_dim, hidden_dim=256, dropout=0.2).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    
    print("SCARF encoder loaded successfully!")
    
    # 提取 SCARF 特徵
    scarf_features, labels = extract_scarf_features(encoder, dataset, device)
    
    print(f"SCARF features shape: {scarf_features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 分割資料
    X_train, X_val, y_train, y_val = train_test_split(
        scarf_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # 訓練 XGBoost
    model = xgb.XGBClassifier(
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
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # 預測和評估
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\nXGBoost + SCARF Features Accuracy: {accuracy:.4f}")
    
    # 獲取類別名稱
    label_names = sorted(df['Fertilizer Name'].unique())
    
    # 分類報告
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=label_names))
    
    # 比較：如果你想比較原始特徵的效果
    print("\n" + "="*50)
    print("Comparison with original features:")
    
    # 用原始特徵訓練 XGBoost
    X_orig = df[cat_cols + num_cols].copy()
    y_orig = df['Fertilizer Name'].copy()
    
    # Label encoding for categorical features
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_orig[col] = le.fit_transform(X_orig[col])
        label_encoders[col] = le
    
    # Label encoding for target
    y_encoder = LabelEncoder()
    y_orig_encoded = y_encoder.fit_transform(y_orig)
    
    X_orig_train, X_orig_val, y_orig_train, y_orig_val = train_test_split(
        X_orig, y_orig_encoded, test_size=0.2, random_state=42, stratify=y_orig_encoded
    )
    
    model_orig = xgb.XGBClassifier(
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
    
    model_orig.fit(X_orig_train, y_orig_train)
    y_orig_pred = model_orig.predict(X_orig_val)
    orig_accuracy = accuracy_score(y_orig_val, y_orig_pred)
    
    print(f"XGBoost + Original Features Accuracy: {orig_accuracy:.4f}")
    print(f"SCARF Feature Improvement: {accuracy - orig_accuracy:.4f}")

if __name__ == '__main__':
    main()