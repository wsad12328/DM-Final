import pandas as pd
import os
import argparse
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

TYPE = None
ENCODE = None
FEATURE_ENG = None

def create_feature_engineering(df, feature_type='none'):
    """
    Create different feature engineering combinations for ablation study
    
    Args:
        df: DataFrame
        feature_type: str, options:
            - 'none': no feature engineering
            - 'npk_features': N/P, N/K, P/K ratios + N+P+K sum
            - 'all': comprehensive agricultural feature engineering
    """
    # Create a copy to avoid warnings
    df = df.copy()
    
    if feature_type == 'none':
        print("No feature engineering applied")
        return df
    
    # Check if required columns exist
    required_cols = ['Nitrogen', 'Potassium', 'Phosphorous', 'Temperature', 'Humidity', 'Moisture']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, skipping some feature engineering")
        return df
    
    if feature_type in ['npk_features', 'all']:
        # Basic NPK features
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-8)
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-8)
        df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-8)
        df['NPK_sum'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
        
        print("Added NPK features: N/P, N/K, P/K ratios + N+P+K sum")
    
    if feature_type == 'all':
        # 1. NPK Percentages
        df['N_percentage'] = df['Nitrogen'] / (df['NPK_sum'] + 1e-8)
        df['P_percentage'] = df['Phosphorous'] / (df['NPK_sum'] + 1e-8)
        df['K_percentage'] = df['Potassium'] / (df['NPK_sum'] + 1e-8)
        
        # 2. Environmental Stress Indicators
        # Heat index
        df['Heat_index'] = df['Temperature'] * df['Humidity'] / 100
        
        # Vapor Pressure Deficit (VPD) - important for plant stress
        df['VPD'] = 0.611 * np.exp(17.27 * df['Temperature'] / (df['Temperature'] + 237.3)) * (1 - df['Humidity']/100)
        
        # Temperature-Moisture interaction
        df['Temp_Moisture_interaction'] = df['Temperature'] * df['Moisture']
        
        # Water stress indicators
        df['Moisture_Humidity_diff'] = df['Moisture'] - df['Humidity']
        df['Water_stress_index'] = df['Moisture'] / (df['Humidity'] + 1e-8)
        
        # 3. Soil Fertility Score
        df['Soil_fertility_score'] = (
            df['Nitrogen'] * 0.33 + 
            df['Phosphorous'] * 0.33 + 
            df['Potassium'] * 0.33
        ) / 100
    
        # 5. Nutrient Balance Index
        nutrient_std = df[['N_percentage', 'P_percentage', 'K_percentage']].std(axis=1)
        df['Nutrient_balance'] = 1 - nutrient_std
        
        # 6. Environmental Extremes
        df['Extreme_temp'] = ((df['Temperature'] < 15) | (df['Temperature'] > 35)).astype(int)
        df['Extreme_humidity'] = ((df['Humidity'] < 55) | (df['Humidity'] > 68)).astype(int)
        df['Extreme_conditions'] = df['Extreme_temp'] + df['Extreme_humidity']
        
        # 7. Polynomial features for key interactions
        df['Temp_squared'] = df['Temperature'] ** 2
        df['Humidity_squared'] = df['Humidity'] ** 2
        df['NPK_product'] = df['Nitrogen'] * df['Phosphorous'] * df['Potassium']
        
        # 8. Binned features for temperature and humidity
        df['Temp_range'] = pd.cut(df['Temperature'], 
                                  bins=[0, 15, 25, 35, 50], 
                                  labels=['cold', 'optimal', 'warm', 'hot'])
        df['Humidity_level'] = pd.cut(df['Humidity'], 
                                      bins=[0, 30, 60, 80, 100], 
                                      labels=['dry', 'moderate', 'humid', 'very_humid'])
        
        # 9. Log transformations for nutrients (handle zeros)
        df['log_Nitrogen'] = np.log1p(df['Nitrogen'])
        df['log_Phosphorous'] = np.log1p(df['Phosphorous'])
        df['log_Potassium'] = np.log1p(df['Potassium'])
        
        # 10. Create crop-soil interaction (if both columns exist)
        if 'Crop Type' in df.columns and 'Soil Type' in df.columns:
            df['Crop_Soil_combo'] = df['Crop Type'].astype(str) + '_' + df['Soil Type'].astype(str)
        
        print("Added comprehensive agricultural features: percentages, stress indicators, suitability indices, polynomials, and interactions")
    
    return df

def preprocess_data(df, encoding_method='label', is_train=True, feature_engineering='none'):    
    # Apply feature engineering first
    df = create_feature_engineering(df, feature_engineering)
    
    # Get numeric columns (excluding id)
    numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != 'id']

    # Create binned features for original numeric columns only
    original_numeric_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    for col in original_numeric_cols:
        if col in df.columns:
            df[f'{col}_Binned'] = df[col].astype(str).astype('category')

    # Get categorical columns (including engineered categorical features)
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != "Fertilizer Name"]
    label_encoders = {}

    # 類別欄位編碼
    if encoding_method == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    elif encoding_method == 'onehot':
        original_cat_cols = cat_cols.copy()
        df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)
    else:
        raise ValueError("encoding_method must be 'label' or 'onehot'")

    # 目標欄位編碼
    if is_train and 'Fertilizer Name' in df.columns:
        target_le = LabelEncoder()
        df['Fertilizer Name'] = target_le.fit_transform(df['Fertilizer Name'])
        label_encoders['Fertilizer Name'] = target_le

    # get new categorical columns if one-hot encoding is used
    if encoding_method == 'onehot':
        cat_cols = [col for col in df.columns if any(col.startswith(orig_col + '_') for orig_col in original_cat_cols)]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"Categorical columns after preprocessing: {cat_cols}")
    print(f"Numerical columns: {numeric_cols}")
    print(f"Feature engineering applied: {feature_engineering}")
    print(f"Total features: {len(cat_cols) + len(numeric_cols)}")

    # save the categorical and numerical columns
    cols_info = {
        'cat_cols': cat_cols,
        'num_cols': numeric_cols,
        'feature_engineering': feature_engineering
    }
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cols_info_path = os.path.join(base_dir, f'../../data/cols_info_{TYPE}_{ENCODE}_{FEATURE_ENG}.pkl')
    with open(cols_info_path, 'wb') as f:
        pickle.dump(cols_info, f)
    print(f"Column information saved to: {cols_info_path}")

    return df, label_encoders

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['train', 'test'], required=True, help="Specify train or test file")
    parser.add_argument('--encoding', choices=['label', 'onehot'], default='label', help="Encoding method")
    parser.add_argument('--feature_eng', choices=['none', 'npk_features', 'all'], default='none', 
                       help="Feature engineering type: none, npk_features (NPK ratios+sum), all (comprehensive agricultural features)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'train.csv' if args.type == 'train' else 'test.csv'
    file_path = os.path.join(base_dir, '../../data', file_name)

    TYPE = args.type
    ENCODE = args.encoding
    FEATURE_ENG = args.feature_eng

    df = pd.read_csv(file_path)
    print("Before preprocessing:")
    print(df.head())
    print(f"Original shape: {df.shape}")

    is_train = args.type == 'train'
    df_processed, label_encoders = preprocess_data(df, encoding_method=args.encoding, 
                                                  is_train=is_train, feature_engineering=args.feature_eng)

    print("\nAfter preprocessing:")
    print(df_processed.head())
    print(f"Final shape: {df_processed.shape}")

    filename = f'preprocessed_{args.type}_{args.encoding}_{args.feature_eng}.csv'
    save_path = os.path.join(base_dir, '../../data', filename)
    df_processed.to_csv(save_path, index=False)
    print(f"\nPreprocessed data saved to: {save_path}")

    # 儲存 label encoder
    if is_train:
        encoders_path = os.path.join(base_dir, '../../data', f'label_encoders_{args.feature_eng}.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved to: {encoders_path}")