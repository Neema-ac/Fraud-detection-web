
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

MODEL_FILE = 'fraud_model.pkl'

def train_model():
    transaction = pd.read_csv(r"C:\Users\USER\Desktop\FDD 2\train_transaction.csv")
    identity = pd.read_csv(r"C:\Users\USER\Desktop\FDD 2\train_identity.csv")
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
