import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from models.SCARF import Encoder, ProjectionHead
from dataset.SCARFDataset import SCARFDataset
from utils.feature_utils import get_cat_cols, get_cat_cardinalities
from tqdm import tqdm
from torch.utils.data import random_split
from dataset.Augmentor import DataAugmentor
from utils.feature_utils import onehot_encode
from utils.feature_utils import OneHotEncoderWrapper

def nt_xent_loss(z1, z2, temperature=0.1):
    batch_size = z1.size(0)
    device = z1.device
    
    # 計算相似度矩陣
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.mm(representations, representations.t()) / temperature
    
    # 創建標籤 - 正樣本對
    labels = torch.arange(batch_size).to(device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    # 創建 mask 來移除自己與自己的相似度
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
    
    # 計算損失
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--encoding', type=str, default='onehot', choices=['label', 'onehot'])
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=3, help='k for MAP@K')
    parser.add_argument('--corruption_rate', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--type', choices=['train', 'test'], required=True, help="Specify train or test file")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    orig_csv = os.path.join(base_dir, f'../data/train.csv')
    print(f"Original: {orig_csv}")

    # 讀取欄位資訊
    # cols = pickle.load(open(os.path.join(os.path.dirname(__file__), '../data/cols_info.pkl'), 'rb'))
    # num_cols = cols['num_cols']
    # cat_cols = cols['cat_cols']
    cat_cols = ["Soil Type","Crop Type","Temperature","Humidity","Moisture","Nitrogen","Potassium","Phosphorous"]
    num_cols = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 讀取整個 csv
    full_df = pd.read_csv(orig_csv)
    encoder = OneHotEncoderWrapper(cat_cols, num_cols)
    encoder.fit(full_df)
    augmentor = DataAugmentor(corruption_rate=args.corruption_rate, seed=args.seed, full_df=full_df)
    dataset = SCARFDataset(
        df=full_df,
        augmentor=augmentor,
        encoder=encoder,
        cat_cols=cat_cols,
        num_cols=num_cols,
        label_col='Fertilizer Name'
    )

    val_size = int(len(dataset) * args.val_size)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20)

    print(f"Total dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    input_dim = train_loader.dataset[0][0].shape[0]

    encoder = Encoder(input_dim=input_dim, hidden_dim=256, dropout=0.2).to(device)
    proj_head = ProjectionHead(input_dim=256, projection_dim=8).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        encoder.train()
        proj_head.train()
        total_loss = 0
        for orig, aug, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            orig = orig.float().to(device)
            aug = aug.float().to(device)
            # print(orig.shape, aug.shape)
            h1 = encoder(orig)
            h2 = encoder(aug)
            z1 = proj_head(h1)
            z2 = proj_head(h2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            # for name, param in encoder.named_parameters():
            #     if param.grad is not None:
            #         print(f'{name}: grad_norm = {param.grad.norm().item():.6f}')
            optimizer.step()
            total_loss += loss.item() * orig.size(0)
        avg_loss = total_loss / len(train_dataset)

        # Validation
        encoder.eval()
        proj_head.eval()
        val_loss = 0
        with torch.no_grad():
            for orig, aug, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                orig = orig.float().to(device)
                aug = aug.float().to(device)
                h1 = encoder(orig)
                h2 = encoder(aug)
                z1 = proj_head(h1)
                z2 = proj_head(h2)
                loss = nt_xent_loss(z1, z2)
                val_loss += loss.item() * orig.size(0)
        avg_val_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_loss:.4f} | Valid Loss: {avg_val_loss:.4f}")


    # 儲存 encoder 預訓練權重
    save_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), f"{save_dir}/scarf_encoder_{args.encoding}.pth")
    print("Pretrained encoder saved.")

if __name__ == '__main__':
    main()