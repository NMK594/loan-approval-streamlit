from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# List of numerical and categorical features
numerical_features = ["Age", "AnnualIncome", "CreditScore", "LoanAmount", "LoanDuration",
                      "Experience", "MonthlyIncome", "TotalLiabilities", "TotalAssets",
                      "NetWorth", "JobTenure", "NumberOfOpenCreditLines", "NumberOfCreditInquiries",
                      "LengthOfCreditHistory", "SavingsAccountBalance", "CheckingAccountBalance",
                      "MonthlyLoanPayment", "BaseInterestRate", "InterestRate"]

categorical_features = ["EmploymentStatus", "EducationLevel", "MaritalStatus", "HomeOwnershipStatus",
                        "LoanPurpose", "PreviousLoanDefaults", "PaymentHistory", "BankruptcyHistory",
                        "UtilityBillsPaymentHistory"]

# Transformers for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
    ("scaler", StandardScaler())  # Normalize numerical data
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with most frequent
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # Convert categorical to one-hot encoding
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Check if your pipeline is working
import pandas as pd

# Dummy DataFrame (Replace with actual data)
df_test = pd.DataFrame({
    "Age": [30, None, 25],  # Includes missing value for testing
    "AnnualIncome": [50000, 60000, None],  # Includes missing value
    "CreditScore": [650, None, 700],
    "EmploymentStatus": ["Employed", None, "Self-Employed"],  # Missing categorical
    "EducationLevel": ["Bachelor", "Master", None],
    "LoanAmount": [10000, 15000, 20000],
    "LoanDuration": [12, None, 24],
    "Experience": [5, 10, None],
    "MaritalStatus": ["Single", "Married", None]
})

# Transform the test data
df_transformed = preprocessor.fit_transform(df_test)
print(df_transformed)
