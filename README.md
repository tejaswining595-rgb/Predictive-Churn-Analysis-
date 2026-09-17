# Predictive-Churn-Analysis-
# ============================================================
# PREDICTIVE CHURN ANALYSIS
# Customer Churn Prediction using Logistic Regression
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

# Change this filename if your dataset has a different name
file_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(file_name)
    print("Dataset loaded successfully!")

except FileNotFoundError:
    print("Dataset file not found.")
    print("Please upload your Telco Customer Churn CSV file.")

    # For Google Colab
    from google.colab import files

    uploaded = files.upload()

    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(file_name)

    print("Dataset uploaded and loaded successfully!")


# ------------------------------------------------------------
# 3. DISPLAY BASIC INFORMATION
# ------------------------------------------------------------

print("\n================ DATASET INFORMATION ================")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 4. DATA CLEANING
# ------------------------------------------------------------

# Remove unnecessary customer ID
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)


# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Replace missing values with median
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )


# Remove any remaining missing rows
df = df.dropna()


print("\nDataset after cleaning:")
print(df.shape)


# ------------------------------------------------------------
# 5. CONVERT TARGET VARIABLE
# ------------------------------------------------------------

# Target variable = Churn
# Yes = 1
# No = 0

if "Churn" not in df.columns:
    raise ValueError(
        "The dataset must contain a column named 'Churn'."
    )

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ------------------------------------------------------------
# 6. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]


print("\n================ TARGET INFORMATION ================")

print("\nChurn distribution:")
print(y.value_counts())

print("\nChurn percentage:")
print(y.value_counts(normalize=True) * 100)


# ------------------------------------------------------------
# 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ------------------------------------------------------------
# 8. ONE-HOT ENCODING + MINMAX SCALING
# ------------------------------------------------------------

# Numerical columns:
# Apply MinMaxScaler

numeric_transformer = Pipeline(
    steps=[
        ("scaler", MinMaxScaler())
    ]
)


# Categorical columns:
# Apply One-Hot Encoding

categorical_transformer = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Combine both transformations

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 9. SPLIT DATA INTO TRAIN AND TEST
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n================ TRAIN TEST SPLIT ================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 10. CREATE LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# 11. TRAIN MODEL
# ------------------------------------------------------------

print("\n================ MODEL TRAINING ================")

model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")


# ------------------------------------------------------------
# 12. MAKE PREDICTIONS
# ------------------------------------------------------------

# Predict class
y_pred = model.predict(X_test)

# Predict probability of churn
y_probability = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 13. MODEL EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n================ MODEL EVALUATION ================")

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1-Score  :", round(f1, 4))
print("ROC-AUC   :", round(roc_auc, 4))


# ------------------------------------------------------------
# 14. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n================ CLASSIFICATION REPORT ================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ]
    )
)


# ------------------------------------------------------------
# 15. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n================ CONFUSION MATRIX ================")

print(cm)


# Display confusion matrix

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["No Churn", "Churn"]
)

plt.yticks(
    [0, 1],
    ["No Churn", "Churn"]
)

plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 16. ROC CURVE
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 17. ROC-AUC SCORE
# ------------------------------------------------------------

print("\n================ ROC-AUC RESULT ================")

print(
    "ROC-AUC Score:",
    round(roc_auc, 4)
)


# ------------------------------------------------------------
# 18. CREATE CUSTOMER CHURN RISK SCORE
# ------------------------------------------------------------

# Create a copy of test data

results = X_test.copy()

# Add actual churn value
results["Actual_Churn"] = y_test.values

# Add predicted churn
results["Predicted_Churn"] = y_pred

# Add churn probability
results["Churn_Probability"] = y_probability


# Convert probability into percentage

results["Churn_Risk_Score"] = (
    results["Churn_Probability"] * 100
).round(2)


# ------------------------------------------------------------
# 19. CREATE RISK CATEGORY
# ------------------------------------------------------------

def risk_category(probability):

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


results["Risk_Category"] = results[
    "Churn_Probability"
].apply(risk_category)


# ------------------------------------------------------------
# 20. DISPLAY CUSTOMER RISK PREDICTIONS
# ------------------------------------------------------------

print("\n================ CUSTOMER CHURN RISK ================")

print(
    results[
        [
            "Actual_Churn",
            "Predicted_Churn",
            "Churn_Risk_Score",
            "Risk_Category"
        ]
    ].head(20)
)


# ------------------------------------------------------------
# 21. COUNT RISK CATEGORIES
# ------------------------------------------------------------

print("\n================ RISK CATEGORY SUMMARY ================")

print(
    results["Risk_Category"].value_counts()
)


# ------------------------------------------------------------
# 22. EXPORT RESULTS TO CSV
# ------------------------------------------------------------

output_file = "customer_churn_risk_predictions.csv"

results.to_csv(
    output_file,
    index=True
)

print("\n================ EXPORT ================")

print(
    "Customer churn risk predictions saved as:",
    output_file
)


# ------------------------------------------------------------
# 23. DISPLAY HIGH-RISK CUSTOMERS
# ------------------------------------------------------------

high_risk_customers = results[
    results["Risk_Category"] == "High Risk"
].sort_values(
    by="Churn_Probability",
    ascending=False
)


print("\n================ HIGH-RISK CUSTOMERS ================")

print(
    high_risk_customers[
        [
            "Churn_Probability",
            "Churn_Risk_Score",
            "Risk_Category"
        ]
    ].head(20)
)


# ------------------------------------------------------------
# 24. FINAL SUMMARY
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("             FINAL CHURN ANALYSIS SUMMARY")
print("=" * 60)

print(
    "Total Customers Analysed:",
    len(df)
)

print(
    "Training Customers:",
    len(X_train)
)

print(
    "Testing Customers:",
    len(X_test)
)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1-Score:",
    round(f1 * 100, 2),
    "%"
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)

print(
    "High-Risk Customers:",
    len(high_risk_customers)
)

print(
    "\nOutput file:",
    output_file
)

print("=" * 60)
print("             ANALYSIS COMPLETED")
print("=" * 60)