import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble ML model** to predict loan approval "
    "by combining multiple base models for better decision making."
)

st.divider()

# ------------------ DATA UPLOAD ------------------
st.sidebar.header("📥 Upload Your Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ------------------ PREPROCESS ------------------
    if all(col in df.columns for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                         'Loan_Amount_Term', 'Credit_History', 'Self_Employed',
                                         'Property_Area', 'Loan_Status']):
        df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})
        df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})
        df['Property_Area'] = df['Property_Area'].map({'Urban':2, 'Semiurban':1, 'Rural':0})

        df = df.dropna()

        X = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                'Loan_Amount_Term', 'Credit_History', 'Self_Employed', 'Property_Area']]
        y = df['Loan_Status']

        # ------------------ TRAIN TEST SPLIT ------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ------------------ SCALING ------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ------------------ TRAIN BASE MODELS ------------------
        lr = LogisticRegression(max_iter=2000)
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        lr.fit(X_train_scaled, y_train)
        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        # ------------------ STACKING ------------------
        stack_input = np.column_stack((
            lr.predict(X_train_scaled),
            dt.predict(X_train),
            rf.predict(X_train)
        ))

        meta_model = LogisticRegression(max_iter=2000)
        meta_model.fit(stack_input, y_train)

        # ------------------ SIDEBAR INPUTS ------------------
        st.sidebar.header("📥 Enter Applicant Details")

        income = st.sidebar.number_input("Applicant Income", min_value=0)
        co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
        loan_amt = st.sidebar.number_input("Loan Amount", min_value=0)
        loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)
        credit = st.sidebar.radio("Credit History", ["Yes", "No"])
        employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
        property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

        # Encode input
        credit_val = 1 if credit=="Yes" else 0
        employment_val = 1 if employment=="Self-Employed" else 0
        prop_val = {"Urban":2, "Semi-Urban":1, "Rural":0}[property_area]

        user_data = np.array([[income, co_income, loan_amt, loan_term, credit_val, employment_val, prop_val]])
        user_scaled = scaler.transform(user_data)

        # ------------------ MODEL ARCHITECTURE ------------------
        st.subheader("🧩 Stacking Model Architecture")
        st.markdown("""
        **Base Models Used:**  
        - Logistic Regression  
        - Decision Tree  
        - Random Forest  

        **Meta Model Used:**  
        - Logistic Regression  

        Predictions from base models are combined and passed to the meta-model for final decision.
        """)

        # ------------------ PREDICTION ------------------
        if st.button("🔘 Check Loan Eligibility (Stacking Model)"):
            pred_lr = lr.predict(user_scaled)[0]
            pred_dt = dt.predict(user_data)[0]
            pred_rf = rf.predict(user_data)[0]

            stack_input_user = np.array([[pred_lr, pred_dt, pred_rf]])
            final_pred = meta_model.predict(stack_input_user)[0]
            confidence = meta_model.predict_proba(stack_input_user)[0][final_pred] * 100

            # ------------------ OUTPUT ------------------
            st.subheader("📊 Base Model Predictions")
            st.write(f"• Logistic Regression → {'Approved' if pred_lr==1 else 'Rejected'}")
            st.write(f"• Decision Tree → {'Approved' if pred_dt==1 else 'Rejected'}")
            st.write(f"• Random Forest → {'Approved' if pred_rf==1 else 'Rejected'}")

            st.subheader("🧠 Final Stacking Decision")
            if final_pred==1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

            st.write(f"📈 Confidence Score: {confidence:.2f}%")

            st.subheader("💼 Business Explanation")
            if final_pred==1:
                st.info(
                    "Based on applicant income, credit history, and combined predictions from multiple models, "
                    "the applicant is **likely to repay the loan**. Therefore, the stacking model predicts **loan approval**."
                )
            else:
                st.info(
                    "Based on applicant income, credit history, and combined predictions from multiple models, "
                    "the applicant is **unlikely to repay the loan**. Therefore, the stacking model predicts **loan rejection**."
                )

    else:
        st.warning("The uploaded dataset is missing required columns!")
else:
    st.info("Please upload a CSV file to proceed.")
