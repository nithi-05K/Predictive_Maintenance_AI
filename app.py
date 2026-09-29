from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("predictive_maintenance_model.pkl")

# Decision threshold selected during evaluation
THRESHOLD = 0.30


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        # Get values from HTML form
        machine_type = request.form.get("machine_type", "L")
        air_temperature = float(request.form.get("air_temperature", 300.0))
        process_temperature = float(request.form.get("process_temperature", 310.0))
        rotational_speed = float(request.form.get("rotational_speed", 1500))
        torque = float(request.form.get("torque", 40.0))
        tool_wear = float(request.form.get("tool_wear", 0))

        # Create input dataframe matching model's expected feature names
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

        # Calculate failure probability
        probability = float(model.predict_proba(input_data)[0][1])
        prob_percent = probability * 100

        # Sensor diagnostics & insights
        anomalies = []
        temp_diff = round(process_temperature - air_temperature, 2)
        if temp_diff < 8.6:
            anomalies.append(f"Low Temperature Dissipation (ΔT: {temp_diff} K) - possible heat dissipation failure risk")
        if tool_wear >= 200:
            anomalies.append(f"Severe Tool Wear ({tool_wear:.0f} min) - tool approaching end-of-life limit (250 min)")
        elif tool_wear >= 150:
            anomalies.append(f"Moderate Tool Wear ({tool_wear:.0f} min) - monitor closely during cycle")
        if torque >= 60:
            anomalies.append(f"High Torque Load ({torque:.1f} Nm) - potential overload strain")
        if rotational_speed <= 1300 and torque >= 55:
            anomalies.append(f"Power Strain Anomaly - low RPM ({rotational_speed:.0f}) with elevated Torque ({torque:.1f} Nm)")

        # 3-Tier Risk & Status Determination
        if probability >= 0.50:
            status = "CRITICAL FAILURE RISK"
            risk = "HIGH"
            status_class = "danger"
            recommendation = (
                "Immediate intervention required: Halt continuous high-load operations, inspect tool assembly "
                "and rotational bearings, and replace heavily worn components before resuming production."
            )
        elif probability >= THRESHOLD:
            status = "ELEVATED RISK"
            risk = "MEDIUM"
            status_class = "warning"
            recommendation = (
                "Preventive maintenance advised: Machine operating parameters are showing elevated strain patterns. "
                "Schedule an inspection at next maintenance window and monitor thermal dissipation."
            )
        else:
            status = "NORMAL OPERATION"
            risk = "LOW"
            status_class = "success"
            recommendation = (
                "Machine operating smoothly within optimal parameters. Continue standard routine maintenance "
                "and telemetry logging."
            )

        result = {
            "machine_type": machine_type,
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "temp_diff": temp_diff,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
            "probability": prob_percent,
            "threshold": THRESHOLD * 100,
            "risk": risk,
            "status": status,
            "status_class": status_class,
            "recommendation": recommendation,
            "anomalies": anomalies
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)