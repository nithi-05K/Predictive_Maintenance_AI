import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("ai4i2020.csv")

# Select features and target
selected_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

clean_df = df[selected_columns]

print("Original dataset shape:", df.shape)
print("Clean dataset shape:", clean_df.shape)

print("\nSelected columns:")
print(clean_df.columns.tolist())

print("\nFirst 5 rows:")
print(clean_df.head())


# ============================================================
# 2. STATISTICAL SUMMARY
# ============================================================

print("\n--- Statistical Summary ---")
print(clean_df.describe())


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

for feature in numerical_features:
    plt.figure(figsize=(7, 4))
    plt.hist(clean_df[feature], bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Records")
    plt.show()


# ============================================================
# 4. HEALTHY VS FAILED MACHINES
# ============================================================

print("\n--- Healthy vs Failed Machines ---")

print("\nAverage values by Machine Failure:")

print(
    clean_df.groupby("Machine failure")[
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    ].mean()
)


# ============================================================
# 5. FAILURE RATE BY MACHINE TYPE
# ============================================================

print("\n--- Failure Rate by Machine Type ---")

failure_by_type = clean_df.groupby("Type")["Machine failure"].agg(
    ["count", "sum", "mean"]
)

failure_by_type["failure_percentage"] = failure_by_type["mean"] * 100

print(failure_by_type)


# ============================================================
# 6. PREPARE X AND y
# ============================================================

X = clean_df[
    [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
]

y = clean_df["Machine failure"]

print("\n--- ML Data Preparation ---")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nX columns:")
print(X.columns.tolist())

print("\ny values:")
print(y.value_counts())


# ============================================================
# 7. ENCODE MACHINE TYPE
# ============================================================

X_encoded = pd.get_dummies(
    X,
    columns=["Type"],
    dtype=int
)

print("\n--- After Encoding Type ---")

print("X_encoded shape:", X_encoded.shape)

print("\nEncoded columns:")
print(X_encoded.columns.tolist())

print("\nFirst 5 encoded rows:")
print(X_encoded.head())


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- Train/Test Split ---")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 9. TRAIN RANDOM FOREST MODEL
# ============================================================

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\n--- Model Training ---")
print("Random Forest model trained successfully!")


# ============================================================
# 10. PREDICT FAILURE PROBABILITY
# ============================================================

y_probability = model.predict_proba(X_test)[:, 1]

# Decision threshold
threshold = 0.30

# Convert probabilities to predictions
y_pred = (y_probability >= threshold).astype(int)

print("\n--- Predictions ---")

print("First 20 failure probabilities:")
print(y_probability[:20])

print("\nFirst 20 predicted values:")
print(y_pred[:20])


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Evaluation ---")

print("Threshold :", threshold)
print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ============================================================
# 12. THRESHOLD ANALYSIS
# ============================================================

print("\n--- Threshold Analysis ---")

thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]

for threshold_value in thresholds:

    threshold_pred = (
        y_probability >= threshold_value
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    print(
        f"Threshold: {threshold_value:.2f} | "
        f"Precision: {threshold_precision:.2f} | "
        f"Recall: {threshold_recall:.2f} | "
        f"F1: {threshold_f1:.2f}"
    )


    # ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

print("\n--- Feature Importance ---")

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)


# ============================================================
# 14. SAVE TRAINED MODEL
# ============================================================

import joblib

joblib.dump(model, "predictive_maintenance_model.pkl")

print("\n--- Model Saving ---")
print("Model saved successfully!")
print("File: predictive_maintenance_model.pkl")
