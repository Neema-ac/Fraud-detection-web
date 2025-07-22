import os
import pandas as pd
import gdown

MODEL_FILE = 'fraud_model.pkl'

def download_if_missing(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists. Skipping download.")

def load_data():
    transaction_url = 'https://drive.google.com/uc?id=1tpJXw8dznuAXZpCCs8GpTne22CvYmt4y'
    identity_url = 'https://drive.google.com/uc?id=19bbfXkc4-59SoXYbTGZUeb_ZVwl92T1_'

    download_if_missing('train_transaction.csv', transaction_url)
    download_if_missing('train_identity.csv', identity_url)

    transaction = pd.read_csv('train_transaction.csv')
    identity = pd.read_csv('train_identity.csv')
    return transaction, identity


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    transaction, identity = load_data()
    data = transaction.merge(identity, how='left', on='TransactionID')
    
    y = data['isFraud']
    X = data.drop(['isFraud', 'TransactionID'], axis=1).select_dtypes(include=['number']).fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, MODEL_FILE)
    return model, X.columns.tolist()

def load_model():
    if not os.path.exists(MODEL_FILE):
        model, features = train_model()
        return model, features
    model = joblib.load(MODEL_FILE)
    features = getattr(model, 'feature_names_in_', None)
    return model, features
