import streamlit as st
import pandas as pd
import pickle

# Load the model pipeline
with open(r"C:\Users\Admin\Downloads\TTNT\xgboost_pipeline.pkl", 'rb') as file:
    loaded_pipeline = pickle.load(file)

st.title("Loan Approval Prediction")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
education_level = st.selectbox("Education Level", ["Bachelor", "Master", "Associate", "Doctorate", "High School"])
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
loan_duration = st.number_input("Loan Duration (months)", min_value=1, value=12)
experience = st.number_input("Experience (years)", min_value=0, max_value=62, value=10)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=0)
home_ownership_status = st.selectbox("Home Ownership Status", ["Rent", "Own", "Mortgage", "Other"])
monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
monthly_debt_payments = st.number_input("Monthly Debt Payments", min_value=50, value=250)
total_liabilities = st.number_input("Total Liabilities", min_value=0, value=5000)
total_assets = st.number_input("Total Assets", min_value=0, value=80000)
job_tenure = st.number_input("Job Tenure (years)", min_value=0, value=3)
number_of_open_credit_lines = st.number_input("Number of Open Credit Lines", min_value=0, value=1)
number_of_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=0)
length_of_credit_history = st.number_input("Length of Credit History (years)", min_value=0, value=10)
savings_account_balance = st.number_input("Savings Account Balance", min_value=0, value=5000)
checking_account_balance = st.number_input("Checking Account Balance", min_value=0, value=2000)
loan_purpose = st.selectbox("Loan Purpose", ["Home", "Debt Consolidation", "Education", "Auto", "Other"])

# Fixed: Percentage-based sliders (0-100)
credit_card_utilization_rate = st.slider("Credit Card Utilization Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
debt_to_income_ratio = st.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=20.0)
total_debt_to_income_ratio = st.slider("Total Debt-to-Income Ratio (%)", min_value=0.0, max_value=300.0, value=10.0)

# Fixed: Interest rates (0-100)
base_interest_rate = st.slider("Base Interest Rate (%)", min_value=0.0, max_value=50.0, value=2.2)
interest_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=50.0, value=3.5)

payment_history = st.number_input("Payment History", min_value=10, value=20)
monthly_loan_payment = st.number_input("Monthly Loan Payment", min_value=0, value=500)
utility_bills_payment_history = st.slider("Utility Bills Payment History", min_value=0.0, max_value=1.0, value=0.9)

# Binary inputs (0 = No, 1 = Yes)
previous_loan_defaults = st.radio("Previous Loan Defaults", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
bankruptcy_history = st.radio("Bankruptcy History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Compute net worth dynamically
net_worth = total_assets - total_liabilities

# Prediction button
if st.button("Predict"):
    # Create DataFrame from user inputs
    new_data = pd.DataFrame({
        'Age': [age],
        'AnnualIncome': [annual_income],
        'CreditScore': [credit_score],
        'EmploymentStatus': [employment_status],
        'EducationLevel': [education_level],
        'LoanAmount': [loan_amount],
        'LoanDuration': [loan_duration],
        'Experience': [experience],
        'MaritalStatus': [marital_status],
        'NumberOfDependents': [number_of_dependents],
        'HomeOwnershipStatus': [home_ownership_status],
        'MonthlyIncome': [monthly_income],
        'MonthlyDebtPayments': [monthly_debt_payments],
        'TotalLiabilities': [total_liabilities],
        'TotalAssets': [total_assets],
        'NetWorth': [net_worth],
        'JobTenure': [job_tenure],
        'NumberOfOpenCreditLines': [number_of_open_credit_lines],
        'NumberOfCreditInquiries': [number_of_credit_inquiries],
        'LengthOfCreditHistory': [length_of_credit_history],
        'SavingsAccountBalance': [savings_account_balance],
        'CheckingAccountBalance': [checking_account_balance],
        'LoanPurpose': [loan_purpose],
        'CreditCardUtilizationRate': [credit_card_utilization_rate],
        'PreviousLoanDefaults': [previous_loan_defaults],
        'PaymentHistory': [payment_history],
        'DebtToIncomeRatio': [debt_to_income_ratio],
        'BankruptcyHistory': [bankruptcy_history],
        'BaseInterestRate': [base_interest_rate],
        'InterestRate': [interest_rate],
        'MonthlyLoanPayment': [monthly_loan_payment],
        'UtilityBillsPaymentHistory': [utility_bills_payment_history],
        'TotalDebtToIncomeRatio': [total_debt_to_income_ratio]
    })

    # Make prediction using the loaded pipeline
    try:
        prediction = loaded_pipeline.predict(new_data)
        st.success("Loan Approved!" if prediction[0] == 1 else "Loan Rejected.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
