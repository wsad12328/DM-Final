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
        # Drop 'id' and label_col before augmentation if they exist
        drop_cols = [col for col in ['id', self.label_col] if col in orig_row.columns]
        aug_input = orig_row.drop(columns=drop_cols)
        aug_row = self.augmentor(aug_input)

        # Add label_col back to aug_row if needed
        if self.label_col in orig_row.columns:
            aug_row[self.label_col] = orig_row[self.label_col].values[0]

        orig_features = self.encoder.transform(orig_row)
        aug_features = self.encoder.transform(aug_row)

        if self.label_col in orig_row.columns:
            label = orig_row[self.label_col].values[0]
            label = self.label2idx[label] if self.label2idx is not None else -1
        else:
            label = -1

        orig_features = torch.tensor(orig_features, dtype=torch.float32).squeeze(0)
        aug_features = torch.tensor(aug_features, dtype=torch.float32).squeeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        return orig_features, aug_features, label