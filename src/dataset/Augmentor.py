import numpy as np

cat_cols = ["Soil Type", "Crop Type", "Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
num_cols = []

class DataAugmentor:
    """
    SCARF-style data augmentation following the original paper
    """
    def __init__(self, full_df, corruption_rate=0.6, seed=None):
        self.full_df = full_df.reset_index(drop=True)
        self.corruption_rate = corruption_rate
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, full_df, row_idx):
        """
        按照 SCARF 論文的方法進行增強
        """
        all_cols = cat_cols + num_cols
        target_row = full_df.iloc[[row_idx]].copy()
        drop_cols = [col for col in ['id', 'Fertilizer Name'] if col in target_row.columns]
        augmented_row = target_row.drop(columns=drop_cols).copy()
        
        # q = ⌊c × M⌋ - 論文中要保持不變的特徵數量
        q = max(1, int(len(all_cols) * (1 - self.corruption_rate)))  # 注意這裡是 1 - corruption_rate
        
        # 隨機選擇 q 個特徵索引 Ii（這些特徵將保持不變）
        cols_to_keep = np.random.choice(all_cols, size=q, replace=False)
        
        # 其他特徵進行 corruption（從經驗分佈中採樣）
        for col in all_cols:
            if col not in cols_to_keep and col in augmented_row.columns:
                # 從該特徵的經驗分佈 X_j 中採樣
                replacement_value = np.random.choice(full_df[col].values)
                augmented_row.loc[augmented_row.index[0], col] = replacement_value
        
        # 把 label 加回去
        if 'Fertilizer Name' in target_row.columns:
            augmented_row['Fertilizer Name'] = target_row['Fertilizer Name'].values[0]
            
        return augmented_row