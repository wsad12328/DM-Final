import numpy as np

num_cols = ['Temperature_Humidity', 'Temperature_Moisture', 'Humidity_to_Moisture',
        'NPK_Total', 'N_to_Moisture', 'P_K_Interaction', 'NP_ratio', 'Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

def get_cat_cols(df):
    exclude = set(num_cols + ['Fertilizer Name', 'id'])
    return [col for col in df.columns if col not in exclude]

def get_cat_cardinalities(cat_x):
    return [int(np.max(cat_x[:, i]) + 1) for i in range(cat_x.shape[1])]
