import torch
from torch.utils.data import Dataset

class SCARFDataset(Dataset):
    """
    Each item returns (original_features, augmented_features, label)
    Augmentation is done on-the-fly, before one-hot encoding.
    """
    def __init__(self, df, augmentor, encoder, cat_cols, num_cols, label_col='Fertilizer Name'):
        self.df_orig = df.reset_index(drop=True)
        self.augmentor = augmentor
        self.encoder = encoder
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.label_col = label_col
        self.use_augmentation = augmentor is not None

        if 'id' in self.df_orig.columns:
            self.df_orig = self.df_orig.drop(columns=['id'])

        if self.label_col in self.df_orig.columns:
            self.label_list = sorted(self.df_orig[self.label_col].unique())
            self.label2idx = {name: idx for idx, name in enumerate(self.label_list)}
        else:
            self.label2idx = None

    def __len__(self):
        return len(self.df_orig)

    def __getitem__(self, idx):
        orig_row = self.df_orig.iloc[[idx]].reset_index(drop=True)
        
        if self.use_augmentation:
            # 傳入整個 DataFrame 和指定的 index 給 augmentor
            aug_row = self.augmentor(self.df_orig, idx)
            aug_features = self.encoder.transform(aug_row)
            aug_features = torch.tensor(aug_features, dtype=torch.float32).squeeze(0)
        else:
            # 如果不使用 augmentation，返回原始特徵
            aug_features = None

        orig_features = self.encoder.transform(orig_row)

        if self.label_col in orig_row.columns:
            label = orig_row[self.label_col].values[0]
            label = self.label2idx[label] if self.label2idx is not None else -1
        else:
            label = -1

        orig_features = torch.tensor(orig_features, dtype=torch.float32).squeeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.use_augmentation:
            return orig_features, aug_features, label
        else:
            return orig_features, orig_features, label  # 返回兩次原始特徵以保持一致性