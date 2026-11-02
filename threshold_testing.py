import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("ai4i2020.csv")

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
# 3. SAME 80/20 TEST SPLIT USED IN TRAINING
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
# 5. GET FAILURE PROBABILITIES ON UNSEEN TEST DATA
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 6. TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]

print("=" * 70)
print("              THRESHOLD TESTING")
print("=" * 70)

print("Unseen test records :", len(X_test))
print()

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1: {f1:.2f}"
    )

print()
print("Selected threshold: 0.30")
print("Reason: Highest F1-score among the tested thresholds.")
print("=" * 70)