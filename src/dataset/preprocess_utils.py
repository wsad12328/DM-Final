# src/preprocessing_utils.py
import os
import pickle
import joblib

def load_label_encoders(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_label_encoders(encoders, path):
    with open(path, 'wb') as f:
        pickle.dump(encoders, f)

def load_scaler(path):
    return joblib.load(path)

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def get_data_paths(base_dir):
    """
    Returns full paths to all common files.
    """
    return {
        "train_csv": os.path.join(base_dir, 'train.csv'),
        "test_csv": os.path.join(base_dir, 'test.csv'),
        "preprocessed_train": os.path.join(base_dir, 'preprocessed_train.csv'),
        "label_encoders": os.path.join(base_dir, 'label_encoders.pkl'),
        "scaler": os.path.join(base_dir, 'scaler.save')
    }
