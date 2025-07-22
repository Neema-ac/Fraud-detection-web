import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_FILE = "fraud_model.pkl"

def load_data():
    transaction = pd.read_csv(r"C:\Users\USER\Desktop\FDD 2\train_transaction.csv")
    identity = pd.read_csv(r"C:\Users\USER\Desktop\FDD 2\train_identity.csv")
    return transaction, identity


def train_model():
    transaction, identity = load_data()
    data = transaction.merge(identity, how='left', on='TransactionID')
    data = data.select_dtypes(include=['number']).fillna(0)

    X = data.drop('isFraud', axis=1)
    y = data['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    return model, list(X.columns)

def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_model()
    model = joblib.load(MODEL_FILE)
    features = model.feature_names_in_
    return model, list(features)
