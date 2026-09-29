import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("ai4i2020.csv")

# Same features used during training
X = df[
    [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
]

y = df["Machine failure"]

# ============================================================
# 2. ENCODE PRODUCT TYPE
# ============================================================

X_encoded = pd.get_dummies(
    X,
    columns=["Type"],
    dtype=int
)

# ============================================================
# 3. RECREATE THE SAME 80/20 TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. LOAD SAVED MODEL
# ============================================================

model = joblib.load("predictive_maintenance_model.pkl")

# ============================================================
# 5. PREDICT ON UNSEEN TEST DATA
# ============================================================

failure_probability = model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.30

predicted_failure = (
    failure_probability >= THRESHOLD
).astype(int)

# ============================================================
# 6. EVALUATE MODEL
# ============================================================

accuracy = accuracy_score(y_test, predicted_failure)

precision = precision_score(
    y_test,
    predicted_failure,
    zero_division=0
)

recall = recall_score(
    y_test,
    predicted_failure,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predicted_failure,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predicted_failure
)

# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("=" * 65)
print("       PREDICTIVE MAINTENANCE MODEL VALIDATION")
print("=" * 65)

print(f"Total dataset records : {len(df)}")
print(f"Training records      : {len(X_train)}")
print(f"Unseen test records   : {len(X_test)}")

print()
print(f"Actual failures in test set : {(y_test == 1).sum()}")
print(f"Actual normal in test set   : {(y_test == 0).sum()}")

print()
print(f"Decision threshold    : {THRESHOLD}")
print(f"Accuracy              : {accuracy * 100:.2f}%")
print(f"Precision             : {precision * 100:.2f}%")
print(f"Recall                : {recall * 100:.2f}%")
print(f"F1 Score              : {f1 * 100:.2f}%")

print()
print("Confusion Matrix:")
print(matrix)

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        predicted_failure,
        zero_division=0
    )
)

# ============================================================
# 8. SHOW REAL TEST-SET EXAMPLES
# ============================================================

test_results = df.loc[
    X_test.index,
    [
        "UDI",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]
].copy()

test_results["Failure Probability (%)"] = (
    failure_probability * 100
)

test_results["Predicted Failure"] = predicted_failure

test_results["Correct"] = (
    test_results["Machine failure"]
    == test_results["Predicted Failure"]
)

print()
print("Sample Unseen Test-Set Results:")
print(
    test_results.head(20).to_string(index=False)
)

# ============================================================
# 9. SHOW SOME ACTUAL FAILURE CASES
# ============================================================

print()
print("Sample Test Records with Actual Failure = 1:")

failure_cases = test_results[
    test_results["Machine failure"] == 1
].head(10)

print(
    failure_cases.to_string(index=False)
)

print()
print("=" * 65)
print("Validation completed.")
print("=" * 65)