
import streamlit as st
import pandas as pd
from fraud_model import load_model
from preprocess import preprocess_uploaded_file
from auth import signup, login
import os

AUDIT_LOG_FILE = 'audit_log.csv'

def log_audit(action, username):
    log_entry = pd.DataFrame([[username, action]], columns=['user', 'action'])
    if os.path.exists(AUDIT_LOG_FILE):
        log_entry.to_csv(AUDIT_LOG_FILE, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(AUDIT_LOG_FILE, index=False)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

def main():
    st.title("Fraud Detection Dashboard (FDD)")

    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Sign Up":
        st.subheader("Create New Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        role = st.selectbox("Select Role", ['user', 'auditor', 'admin'])
        if st.button("Sign Up"):
            if signup(username, password, role):
                st.success("Account created. Please Login.")
            else:
                st.error("User already exists.")
    else:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            role = login(username, password)
            if role:
                st.success(f"Welcome {username} ({role})")
                log_audit("Logged in", username)
                run_dashboard(username, role)
            else:
                st.error("Invalid credentials")

def run_dashboard(username, role):
    model, model_features = load_model()

    st.sidebar.header("Upload CSV for Fraud Detection")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Sample", user_data.head())

        processed_data = preprocess_uploaded_file(user_data, model_features)
        predictions = model.predict(processed_data)
        user_data['Fraud_Prediction'] = predictions

        st.write("### Prediction Summary")
        st.dataframe(user_data['Fraud_Prediction'].value_counts().rename_axis('Fraud').reset_index(name='Count'))

        st.write("### Prediction Details")
        st.dataframe(user_data)

        csv = user_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv, file_name="fraud_results.csv", mime='text/csv')

        if role == 'auditor':
            if st.button("Confirm Review of Predictions"):
                log_audit("Auditor reviewed predictions", username)
                st.success("Review logged.")

        if role == 'admin':
            st.subheader("Admin Panel")
            if st.button("View Audit Logs"):
                if os.path.exists(AUDIT_LOG_FILE):
                    audit_log = pd.read_csv(AUDIT_LOG_FILE)
                    st.dataframe(audit_log)
                else:
                    st.info("No audit logs yet.")

            if st.button("Retrain Model (Admin Only)"):
                from fraud_model import train_model
                model, features = train_model()
                st.success("Model retrained successfully.")

if __name__ == '__main__':
    main()
