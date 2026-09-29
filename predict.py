import joblib
import pandas as pd

# ============================================================
# PREDICTIVE MAINTENANCE SYSTEM
# ============================================================

# Load trained model
model = joblib.load("predictive_maintenance_model.pkl")

# Decision threshold selected during model evaluation
THRESHOLD = 0.30

print("==========================================")
print("      PREDICTIVE MAINTENANCE SYSTEM")
print("==========================================")

# ============================================================
# 1. MACHINE TYPE
# ============================================================

while True:
    machine_type = input("Enter machine type (L/M/H): ").upper()

    if machine_type in ["L", "M", "H"]:
        break

    print("Invalid machine type. Please enter L, M, or H.")


# ============================================================
# 2. INPUT VALIDATION FUNCTION
# ============================================================

def get_value(message, minimum, maximum):
    while True:
        try:
            value = float(input(message))

            if minimum <= value <= maximum:
                return value

            print(
                f"Invalid value. Please enter a value "
                f"between {minimum} and {maximum}."
            )

        except ValueError:
            print("Please enter a valid number.")


# ============================================================
# 3. GET MACHINE PARAMETERS
# ============================================================

air_temperature = get_value(
    "Enter air temperature [K] (295.3 - 304.5): ",
    295.3,
    304.5
)

process_temperature = get_value(
    "Enter process temperature [K] (305.7 - 313.8): ",
    305.7,
    313.8
)

rotational_speed = get_value(
    "Enter rotational speed [rpm] (1168 - 2886): ",
    1168,
    2886
)

torque = get_value(
    "Enter torque [Nm] (3.8 - 76.6): ",
    3.8,
    76.6
)

tool_wear = get_value(
    "Enter tool wear [min] (0 - 253): ",
    0,
    253
)


# ============================================================
# 4. ENCODE MACHINE TYPE
# ============================================================

input_data = pd.DataFrame({
    "Air temperature [K]": [air_temperature],
    "Process temperature [K]": [process_temperature],
    "Rotational speed [rpm]": [rotational_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear],
    "Type_H": [1 if machine_type == "H" else 0],
    "Type_L": [1 if machine_type == "L" else 0],
    "Type_M": [1 if machine_type == "M" else 0]
})


# ============================================================
# 5. PREDICT FAILURE PROBABILITY
# ============================================================

failure_probability = model.predict_proba(input_data)[0][1]

failure_percentage = failure_probability * 100


# ============================================================
# 6. DETERMINE MACHINE STATUS
# ============================================================

if failure_probability >= THRESHOLD:
    machine_status = "FAILURE RISK"
else:
    machine_status = "NORMAL"


# ============================================================
# 7. DETERMINE RISK LEVEL
# ============================================================

if failure_probability >= 0.50:
    risk_level = "HIGH"
elif failure_probability >= 0.30:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"


# ============================================================
# 8. MAINTENANCE RECOMMENDATION
# ============================================================

if machine_status == "FAILURE RISK":
    recommendation = (
        "Schedule maintenance inspection and check the machine "
        "parameters before continued operation."
    )
else:
    recommendation = (
        "Machine appears to be operating normally. "
        "Continue regular monitoring and preventive maintenance."
    )


# ============================================================
# 9. DISPLAY RESULT
# ============================================================

print("\n==========================================")
print("              PREDICTION")
print("==========================================")

print(f"Machine type        : {machine_type}")
print(f"Air temperature     : {air_temperature:.1f} K")
print(f"Process temperature  : {process_temperature:.1f} K")
print(f"Rotational speed    : {rotational_speed:.0f} rpm")
print(f"Torque              : {torque:.1f} Nm")
print(f"Tool wear           : {tool_wear:.0f} min")

print("------------------------------------------")

print(f"Failure probability : {failure_percentage:.3f}%")
print(f"Risk level          : {risk_level}")
print(f"Decision threshold  : {THRESHOLD * 100:.0f}%")
print(f"Machine status      : {machine_status}")

print("------------------------------------------")

print("Recommendation:")
print(recommendation)

print("==========================================")