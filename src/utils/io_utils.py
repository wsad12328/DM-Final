import os
import pandas as pd
import joblib
import pickle
import torch

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_preprocessed_data(data_type='train', encoding_method='label'):
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, '../../data', f'preprocessed_{data_type}_{encoding_method}.csv')
    return pd.read_csv(file_path)

def load_model(model_name='xgboost', encoding_method='label'):
    base_dir = get_base_dir()
    if model_name == 'mlp':
        model_path = os.path.join(base_dir, '../../models', f'{model_name}_{encoding_method}_fold3.pth')
        return torch.load(model_path, map_location='cpu', weights_only=True)
    elif 'boost' in model_name or 'forest' in model_name:
        model_path = os.path.join(base_dir, '../../models', f'{model_name}_{encoding_method}.pkl')
        return joblib.load(model_path)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def load_label_encoders():
    base_dir = get_base_dir()
    encoders_path = os.path.join(base_dir, '../../data', 'label_encoders.pkl')
    with open(encoders_path, 'rb') as f:
        return pickle.load(f)