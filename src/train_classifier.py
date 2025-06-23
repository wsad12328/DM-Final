import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import os
from models.SCARF import Encoder, ClassifierHead
from utils.feature_utils import OneHotEncoderWrapper, LabelEncoderWrapper
from utils.evaluate import mapk
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse

def get_top_k_predictions(outputs, k):
    """獲取 top-k 預測結果"""
    _, top_k_indices = torch.topk(outputs, k, dim=1)
    return top_k_indices.cpu().numpy()

def encode_all_features(encoder, features, device, batch_size=1024):
    """使用 SCARF encoder 一次性編碼所有特徵"""
    encoder.eval()
    encoded_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(features), batch_size), desc="Encoding features with SCARF"):
            batch = features[i:i+batch_size].float().to(device)
            encoded_batch = encoder(batch)
            encoded_features.append(encoded_batch.cpu())
    
    return torch.cat(encoded_features, dim=0)

def train_classifier(classifier, train_loader, val_loader, device, epochs=50, lr=1e-3, k=3, fold_num=None):
    """訓練分類器 (輸入是已經編碼的特徵)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            div_factor=25.0,
            final_div_factor=1e3,
            pct_start=0.3,
            total_steps=epochs*len(train_loader)
    )
    
    best_val_mapk = 0
    best_model_state = None
    
    fold_desc = f"Fold {fold_num}" if fold_num is not None else "Training"
    
    for epoch in range(epochs):
        # Training
        classifier.train()
        
        train_loss = 0
        train_total = 0
        all_train_actual = []
        all_train_predicted = []
        
        train_pbar = tqdm(train_loader, desc=f"{fold_desc} | Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for encoded_features, labels in train_pbar:
            encoded_features = encoded_features.float().to(device)
            labels = labels.long().to(device)
            
            outputs = classifier(encoded_features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_total += labels.size(0)
            
            # 收集預測結果，但不在這裡計算 MAP@K
            top_k_preds = get_top_k_predictions(outputs, k)
            all_train_actual.extend(labels.cpu().numpy())
            all_train_predicted.extend(top_k_preds)
            
            # 只顯示 loss
            train_pbar.set_postfix({
                'Loss': f'{train_loss/train_total:.4f}'
            })
        
        # Epoch 結束後計算訓練 MAP@K
        train_mapk = mapk(all_train_actual, all_train_predicted, k)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        classifier.eval()
        val_loss = 0
        val_total = 0
        all_val_actual = []
        all_val_predicted = []
        
        val_pbar = tqdm(val_loader, desc=f"{fold_desc} | Epoch {epoch+1}/{epochs} [Valid]", leave=False)
        with torch.no_grad():
            for encoded_features, labels in val_pbar:
                encoded_features = encoded_features.float().to(device)
                labels = labels.long().to(device)
                
                outputs = classifier(encoded_features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_total += labels.size(0)
                
                # 收集預測結果，但不在這裡計算 MAP@K
                top_k_preds = get_top_k_predictions(outputs, k)
                all_val_actual.extend(labels.cpu().numpy())
                all_val_predicted.extend(top_k_preds)
                
                # 只顯示 loss
                val_pbar.set_postfix({
                    'Loss': f'{val_loss/val_total:.4f}'
                })
        
        # Epoch 結束後計算驗證 MAP@K
        val_mapk = mapk(all_val_actual, all_val_predicted, k)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"{fold_desc} | Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train MAP@{k}: {train_mapk:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAP@{k}: {val_mapk:.4f}")
        
        if val_mapk > best_val_mapk:
            best_val_mapk = val_mapk
            best_model_state = classifier.state_dict().copy()
    
    classifier.load_state_dict(best_model_state)
    return best_val_mapk

def evaluate_model(classifier, val_loader, device, k=3, fold_num=None):
    """評估模型"""
    classifier.eval()
    
    all_actual = []
    all_predicted = []
    
    fold_desc = f"Fold {fold_num}" if fold_num is not None else "Evaluation"
    
    eval_pbar = tqdm(val_loader, desc=f"{fold_desc} | Evaluating", leave=False)
    with torch.no_grad():
        for encoded_features, labels in eval_pbar:
            encoded_features = encoded_features.float().to(device)
            labels = labels.long().to(device)
            
            outputs = classifier(encoded_features)
            
            # 收集預測結果
            top_k_preds = get_top_k_predictions(outputs, k)
            all_actual.extend(labels.cpu().numpy())
            all_predicted.extend(top_k_preds)
    
    # 評估結束後計算最終 MAP@K
    final_mapk = mapk(all_actual, all_predicted, k)
    print(f"\n{fold_desc} | Final MAP@{k}: {final_mapk:.4f}")
    
    return final_mapk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_encoder', type=str, required=True, 
                       help='Path to pretrained SCARF encoder weights')
    parser.add_argument('--encoding', type=str, default='onehot', choices=['label', 'onehot'],
                       help='Encoding method used during pretraining')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--k', type=int, default=3, help='k for MAP@K evaluation')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, '../data/train.csv')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using encoding: {args.encoding}")
    print(f"Evaluation metric: MAP@{args.k}")
    
    # 直接讀取資料
    print("Loading data...")
    df = pd.read_csv(train_csv)
    
    cat_cols = ["Soil Type", "Crop Type", "Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
    num_cols = []
    
    # 根據編碼方式選擇不同的編碼器
    print(f"Applying {args.encoding} encoding...")
    if args.encoding == 'onehot':
        feature_encoder = OneHotEncoderWrapper(cat_cols, num_cols)
    else:  # label encoding
        feature_encoder = LabelEncoderWrapper(cat_cols, num_cols)
    
    # 編碼特徵
    feature_encoder.fit(df)
    encoded_features = feature_encoder.transform(df)
    encoded_features = torch.tensor(encoded_features, dtype=torch.float32)
    
    # 編碼標籤
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['Fertilizer Name'])
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)
    
    print(f"Encoded features shape: {encoded_features.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    model_path = os.path.join(base_dir, args.pretrained_encoder)

    # 載入預訓練的 SCARF encoder
    input_dim = encoded_features.shape[1]
    scarf_encoder = Encoder(input_dim=input_dim, hidden_dim=256, dropout=0.2).to(device)
    scarf_encoder.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # 凍結 SCARF encoder 參數
    for param in scarf_encoder.parameters():
        param.requires_grad = False
    scarf_encoder.eval()
    
    print("SCARF encoder loaded and frozen successfully!")
    
    # 用 SCARF encoder 編碼所有特徵
    print("Encoding features with SCARF encoder...")
    scarf_embeddings = encode_all_features(scarf_encoder, encoded_features, device)
    
    print(f"SCARF embeddings shape: {scarf_embeddings.shape}")
    
    # K-Fold Cross Validation
    kfold = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    # 修正儲存路徑 - 應該是 models 目錄，不是預訓練模型的檔案路徑
    save_dir = os.path.join(base_dir, '../models')
    os.makedirs(save_dir, exist_ok=True)
    
    num_classes = len(label_encoder.classes_)
    
    print(f"\nStarting {args.k_folds}-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(scarf_embeddings, encoded_labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{args.k_folds}")
        print(f"{'='*50}")
        
        # 分割 SCARF embeddings
        X_train, X_val = scarf_embeddings[train_idx], scarf_embeddings[val_idx]
        y_train, y_val = encoded_labels[train_idx], encoded_labels[val_idx]
        
        # 建立 DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"Training set size: {X_train.shape}")
        print(f"Validation set size: {X_val.shape}")
        
        # 建立分類器 (輸入維度是 SCARF embedding 維度)
        classifier = ClassifierHead(input_dim=scarf_embeddings.shape[1], num_classes=num_classes).to(device)
        
        # 訓練分類器
        best_mapk = train_classifier(classifier, train_loader, val_loader, device, 
                                    epochs=args.epochs, lr=args.lr, k=args.k, fold_num=fold+1)
        
        # 評估模型
        final_mapk = evaluate_model(classifier, val_loader, device, k=args.k, fold_num=fold+1)
        
        # 儲存分類器
        classifier_path = f"{save_dir}/scarf_classifier_{args.encoding}_fold_{fold+1}.pth"
        torch.save(classifier.state_dict(), classifier_path)
        print(f"Fold {fold+1} classifier saved as: {classifier_path}")
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_mapk': best_mapk,
            'final_test_mapk': final_mapk
        })
        
        print(f"Fold {fold+1} | Best Val MAP@{args.k}: {best_mapk:.4f} | Final Test MAP@{args.k}: {final_mapk:.4f}")
    
    # 計算平均結果
    avg_val_mapk = np.mean([result['best_val_mapk'] for result in fold_results])
    avg_test_mapk = np.mean([result['final_test_mapk'] for result in fold_results])
    std_val_mapk = np.std([result['best_val_mapk'] for result in fold_results])
    std_test_mapk = np.std([result['final_test_mapk'] for result in fold_results])
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS (MAP@{args.k})")
    print(f"{'='*60}")
    for result in fold_results:
        print(f"Fold {result['fold']}: Val MAP@{args.k} = {result['best_val_mapk']:.4f}, Test MAP@{args.k} = {result['final_test_mapk']:.4f}")
    
    print(f"\nAverage Validation MAP@{args.k}: {avg_val_mapk:.4f} ± {std_val_mapk:.4f}")
    print(f"Average Test MAP@{args.k}: {avg_test_mapk:.4f} ± {std_test_mapk:.4f}")
    
    # 儲存結果
    results_summary = {
        'encoding': args.encoding,
        'k_folds': args.k_folds,
        'k': args.k,
        'fold_results': fold_results,
        'avg_val_mapk': avg_val_mapk,
        'avg_test_mapk': avg_test_mapk,
        'std_val_mapk': std_val_mapk,
        'std_test_mapk': std_test_mapk
    }
    
    import json
    with open(f"{save_dir}/cv_results_{args.encoding}_mapk{args.k}.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults summary saved as: {save_dir}/cv_results_{args.encoding}_mapk{args.k}.json")

if __name__ == '__main__':
    main()