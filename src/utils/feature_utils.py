import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

num_cols = ['Temperature_Humidity', 'Temperature_Moisture', 'Humidity_to_Moisture',
        'NPK_Total', 'N_to_Moisture', 'P_K_Interaction', 'NP_ratio', 'Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

def get_cat_cols(df):
    exclude = set(num_cols + ['Fertilizer Name', 'id'])
    return [col for col in df.columns if col not in exclude]

def get_cat_cardinalities(cat_x):
    return [int(np.max(cat_x[:, i]) + 1) for i in range(cat_x.shape[1])]


def onehot_encode(df, cat_cols, num_cols):
    """
    將 cat_cols 做 one-hot encoding，num_cols 保持原樣，合併後回傳 numpy array。
    """
    cat_df = pd.get_dummies(df[cat_cols], columns=cat_cols)
    if num_cols:
        num_df = df[num_cols].reset_index(drop=True)
        out = pd.concat([num_df, cat_df], axis=1)
    else:
        out = cat_df
    return out.values

class OneHotEncoderWrapper:
    def __init__(self, cat_cols, num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, df):
        self.encoder.fit(df[self.cat_cols])

    def transform(self, df):
        cat_encoded = self.encoder.transform(df[self.cat_cols])
        if self.num_cols:
            num_data = df[self.num_cols].to_numpy()
            return np.hstack([num_data, cat_encoded])
        else:
            return cat_encoded

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
