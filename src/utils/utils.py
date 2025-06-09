import os
import pandas as pd
import joblib
import pickle

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_preprocessed_data(data_type='train', encoding_method='label'):
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, '../../data', f'preprocessed_{data_type}_{encoding_method}.csv')
    return pd.read_csv(file_path)

def load_model(model_name='xgboost', encoding_method='label'):
    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, '../../models', f'{model_name}_{encoding_method}.pkl')
    return joblib.load(model_path)

def load_label_encoders():
    base_dir = get_base_dir()
    encoders_path = os.path.join(base_dir, '../../data', 'label_encoders.pkl')
    with open(encoders_path, 'rb') as f:
        return pickle.load(f)