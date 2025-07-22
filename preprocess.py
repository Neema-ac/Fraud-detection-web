
import pandas as pd

def preprocess_uploaded_file(uploaded_df, model_features):
    df = uploaded_df.copy()
    numeric_df = df.select_dtypes(include=['number']).fillna(0)
    
    for col in model_features:
        if col not in numeric_df.columns:
            numeric_df[col] = 0
    return numeric_df[model_features]
