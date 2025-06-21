import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.MLP import MLP
from utils.evaluate import mapk
from tqdm import tqdm
import os
from utils.feature_utils import get_cat_cardinalities
import pickle
from sklearn.model_selection import StratifiedKFold

num_cols = None
cat_cols = None

def get_model(model_name, numeric_num_features, output_dim, cat_cardinalities):
    if model_name == 'mlp':
        return MLP(numeric_num_features, cat_cardinalities, output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def load_csv_data(train_path, val_size=0.2, random_state=42):
    train_path = os.path.join(os.path.dirname(__file__), train_path)

    df = pd.read_csv(train_path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    num_x = df[num_cols].values.astype(np.float32)
    cat_x = df[cat_cols].values.astype(np.int32)
    y = df['Fertilizer Name'].values.astype(np.int32)

    num_x_train, num_x_val, cat_x_train, cat_x_val, y_train, y_val = train_test_split(
        num_x, cat_x, y, test_size=val_size, random_state=random_state, stratify=y
    )
    return num_x_train, cat_x_train, y_train, num_x_val, cat_x_val, y_val

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for num_x, cat_x, y_batch in tqdm(train_loader, desc="Training", leave=False):
        num_x, cat_x, y_batch = num_x.to(device), cat_x.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(num_x, cat_x)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * num_x.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, val_loader, device, k=3):
    model.eval()
    correct = 0
    total = 0
    all_actual = []
    all_pred_topk = []
    with torch.no_grad():
        for num_x, cat_x, y_batch in tqdm(val_loader, desc="Validating", leave=False):
            num_x, cat_x, y_batch = num_x.to(device), cat_x.to(device), y_batch.to(device)
            outputs = model(num_x, cat_x)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            topk = torch.topk(outputs, k, dim=1).indices.cpu().numpy()
            all_pred_topk.extend(topk)
            all_actual.extend(y_batch.cpu().numpy())
    accuracy = correct / total
    mapk_score = mapk(all_actual, all_pred_topk, k)
    return accuracy, mapk_score

def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, k):
    best_mapk = 0
    best_state = None
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_acc, mapk_score = evaluate(model, val_loader, device, k=k)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | MAP@{k}: {mapk_score:.4f}")
        if mapk_score > best_mapk:
            best_mapk = mapk_score
            best_state = model.state_dict()
        scheduler.step()
    return best_mapk, best_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='Model type: mlp')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--encoding', type=str, default='label', choices=['label', 'onehot'])
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=3, help='k for MAP@K')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for KFold')
    args = parser.parse_args()
    cols = pickle.load(open(os.path.join(os.path.dirname(__file__), f'../data/cols_info_train_{args.encoding}.pkl'), 'rb'))
    num_cols = cols['num_cols']
    cat_cols = cols['cat_cols']

    if args.encoding == 'label':
        train_csv = '../data/preprocessed_train_label.csv'
    else:
        train_csv = '../data/preprocessed_train_onehot.csv'

    print(train_csv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 讀取全部資料
    train_path = os.path.join(os.path.dirname(__file__), train_csv)
    df = pd.read_csv(train_path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    num_x = df[num_cols].values.astype(np.float32)
    cat_x = df[cat_cols].values.astype(np.int32)
    y = df['Fertilizer Name'].values.astype(np.int64)

    numeric_num_features = num_x.shape[1]
    output_dim = len(np.unique(y))
    cat_cardinalities = get_cat_cardinalities(cat_x)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_mapks = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(num_x, y)):
        print(f"\n=== Fold {fold+1}/{args.n_splits} ===")
        num_x_train, num_x_val = num_x[train_idx], num_x[val_idx]
        cat_x_train, cat_x_val = cat_x[train_idx], cat_x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TensorDataset(
            torch.tensor(num_x_train, dtype=torch.float32),
            torch.tensor(cat_x_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(num_x_val, dtype=torch.float32),
            torch.tensor(cat_x_val, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = get_model(args.model, numeric_num_features, output_dim, cat_cardinalities).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            div_factor=25.0,
            final_div_factor=1e3,
            pct_start=0.3,
            total_steps=total_steps
        )

        best_mapk, best_state = train_fold(
            model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs, args.k
        )
        fold_mapks.append(best_mapk)
        save_dir = os.path.join(os.path.dirname(__file__), '../models')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, f"{save_dir}/{args.model}_{args.encoding}_fold{fold+1}.pth")
        print(f"Fold {fold+1} best MAP@{args.k}: {best_mapk:.4f}")

    print(f"\nAll folds MAP@{args.k}: {fold_mapks}")
    print(f"Average MAP@{args.k}: {np.mean(fold_mapks):.4f}")

if __name__ == '__main__':
    main()