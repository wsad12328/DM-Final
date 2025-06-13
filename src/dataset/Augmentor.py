import numpy as np

cat_cols = ["Soil Type", "Crop Type", "Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
num_cols = []  # 若有數值型欄位可補上

class DataAugmentor:
    """
    SCARF-style data augmentation: randomly select a subset of features and replace their values
    with samples from the empirical distribution of that feature.
    """
    def __init__(self, full_df,corruption_rate=0.6, seed=None):
        self.full_df = full_df.reset_index(drop=True)
        self.corruption_rate = corruption_rate

    def __call__(self, df):
        all_cols = cat_cols + num_cols
        augmented_df = df.copy()
        n_cols_to_corrupt = max(1, int(len(all_cols) * self.corruption_rate))
        cols_to_corrupt = np.random.choice(all_cols, size=n_cols_to_corrupt, replace=False)

        for col in cols_to_corrupt:
            replacement_values = np.random.choice(self.full_df[col].values, size=1, replace=True)
            augmented_df[col] = replacement_values
            
        return augmented_df