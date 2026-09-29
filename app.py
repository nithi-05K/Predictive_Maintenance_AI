import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request
from google import genai
import joblib
import pandas as pd

# Load environment variables securely from .env file
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))

app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("predictive_maintenance_model.pkl")

# Decision threshold selected during evaluation
THRESHOLD = 0.30

# Gemini model identifier
GEMINI_MODEL = "gemini-3.5-flash-lite"


def get_gemini_client():
    """Securely initialize and return the Gemini API client if API key is present."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def parse_gemini_analysis(text):
    """Parse Gemini's structured response into discrete analysis sections."""
    data = {
        "ai_explanation": "",
        "ai_key_parameters": "",
        "ai_maintenance_guidance": "",
        "ai_advisory": ""
    }

    patterns = [
        ("ai_explanation", r"(?:\*{1,2}|#{1,4}\s*)?AI Explanation(?:\*{1,2})?\s*:\s*(.*?)(?=(?:\*{1,2}|#{1,4}\s*)?(?:Key Parameters|Maintenance Guidance|Advisory)(?:\*{1,2})?\s*:|$)", re.DOTALL | re.IGNORECASE),
        ("ai_key_parameters", r"(?:\*{1,2}|#{1,4}\s*)?Key Parameters(?:\*{1,2})?\s*:\s*(.*?)(?=(?:\*{1,2}|#{1,4}\s*)?(?:Maintenance Guidance|Advisory)(?:\*{1,2})?\s*:|$)", re.DOTALL | re.IGNORECASE),
        ("ai_maintenance_guidance", r"(?:\*{1,2}|#{1,4}\s*)?Maintenance Guidance(?:\*{1,2})?\s*:\s*(.*?)(?=(?:\*{1,2}|#{1,4}\s*)?(?:Advisory)(?:\*{1,2})?\s*:|$)", re.DOTALL | re.IGNORECASE),
        ("ai_advisory", r"(?:\*{1,2}|#{1,4}\s*)?Advisory(?:\*{1,2})?\s*:\s*(.*?)$", re.DOTALL | re.IGNORECASE),
    ]

    for key, pattern, flags in patterns:
        match = re.search(pattern, text, flags)
        if match:
            cleaned = match.group(1).strip()
            cleaned = re.sub(r'^\*+|\*+$', '', cleaned).strip()
            data[key] = cleaned

    # Fallback if headings were absent or not recognized
    if not data["ai_explanation"] and not data["ai_key_parameters"]:
        data["ai_explanation"] = text.strip()
        data["ai_advisory"] = "This Gemini explanation is advisory. The Random Forest model is responsible for the actual failure prediction."

    return data


def analyze_with_gemini(machine_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear, prob_percent, risk, status):
    """
    Generate advisory explanation and maintenance guidance using Gemini.
    Gemini runs strictly AFTER the Random Forest model and does NOT decide failure.
    Catches all exceptions gracefully so ML predictions never fail.
    """
    fallback_result = {
        "ai_available": False,
        "ai_error": "AI explanation unavailable. ML prediction is still available.",
        "ai_explanation": "AI explanation unavailable. ML prediction is still available.",
        "ai_key_parameters": "AI analysis unavailable.",
        "ai_maintenance_guidance": "Please follow the standard machine learning recommendation above.",
        "ai_advisory": "This Gemini explanation is advisory. The Random Forest model is responsible for the actual failure prediction."
    }

    client = get_gemini_client()
    if not client:
        return fallback_result

    prompt = f"""You are an AI Industrial Diagnostics Assistant for a Predictive Maintenance System.
A machine learning Random Forest model has already evaluated the sensor telemetry and predicted the failure probability and risk level.

Supplied Telemetry and Model Output:
- Product Type: {machine_type}
- Air Temperature: {air_temperature:.1f} K
- Process Temperature: {process_temperature:.1f} K
- Rotational Speed: {rotational_speed:.0f} rpm
- Torque: {torque:.1f} Nm
- Tool Wear: {tool_wear:.0f} min
- Model Failure Probability: {prob_percent:.2f}%
- Model Risk Level: {risk}
- Model Machine Status: {status}

STRICT WORDING RULES:
1. Describe the Random Forest model's result objectively. Do not claim that the prediction has been proven correct, that a failure is confirmed, or that the machine will definitely fail.
2. Prohibited wording:
   - Never say "the machine will fail", "the machine is unstable", "this parameter caused the failure", "the prediction is correct", "safe", or "unsafe".
   - Do NOT use adjectives like "severe", "dangerous", "abnormal", "extreme", "high", "low", or "advanced" to describe numeric values unless quoting the model risk tier itself.
3. Do not infer causation. Use phrasing such as:
   - "the model indicates"
   - "the model predicts"
   - "may contribute"
   - "is relevant to the model prediction"
   - "is associated with the predicted risk"
   - "should be monitored"
4. Do not invent machine conditions, component damage, bearing problems, tool failure, overheating, or mechanical faults that were not supplied.
5. Do not recommend emergency shutdown.
6. When discussing tool wear, describe the actual value only (e.g., "Tool wear is {tool_wear:.0f} minutes and should continue to be monitored as it increases.").
7. When discussing rotational speed, use the actual supplied value and avoid unsupported claims (e.g., "Rotational speed is {rotational_speed:.0f} rpm and is one of the telemetry parameters used by the Random Forest model.").
8. When discussing temperature, describe the actual supplied values only without claiming overheating or thermal failure.
9. Maintenance guidance must be practical and general:
   - inspect relevant machine components
   - monitor operating parameters
   - schedule preventive maintenance
   - continue telemetry monitoring
   - review the machine if the risk remains elevated
10. The Advisory section must be exactly:
    "This Gemini explanation is advisory. The Random Forest model is responsible for the actual failure prediction."

Format your output with exactly these 4 section headings:

AI Explanation:
(1-2 sentences explaining what the Random Forest prediction means for the supplied telemetry)

Key Parameters:
(1-2 sentences mentioning the supplied parameters that are relevant to the model prediction)

Maintenance Guidance:
(1-2 sentences providing practical and general preventive-maintenance guidance)

Advisory:
This Gemini explanation is advisory. The Random Forest model is responsible for the actual failure prediction.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if response and response.text:
            parsed = parse_gemini_analysis(response.text)
            parsed["ai_available"] = True
            parsed["ai_error"] = ""
            return parsed
        else:
            return fallback_result

    except Exception:
        # Never expose raw exceptions, stack traces, or API keys to the user
        return fallback_result


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

        # Calculate failure probability using validated Random Forest model
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
                "Preventive maintenance is advised. Review the current operating conditions and inspect "
                "relevant machine components before continued production. Continue monitoring the telemetry."
            )
        elif probability >= THRESHOLD:
            status = "ELEVATED RISK"
            risk = "MEDIUM"
            status_class = "warning"
            recommendation = (
                "Preventive maintenance is advised. Operating parameters are showing elevated strain patterns "
                "according to the model. Schedule an inspection during the next maintenance window and continue monitoring telemetry."
            )
        else:
            status = "NORMAL OPERATION"
            risk = "LOW"
            status_class = "success"
            recommendation = (
                "Machine is operating within normal parameters according to the model. Continue standard routine "
                "maintenance and telemetry monitoring."
            )

        # Call Gemini Generative AI for advisory explanation & maintenance guidance
        gemini_data = analyze_with_gemini(
            machine_type=machine_type,
            air_temperature=air_temperature,
            process_temperature=process_temperature,
            rotational_speed=rotational_speed,
            torque=torque,
            tool_wear=tool_wear,
            prob_percent=prob_percent,
            risk=risk,
            status=status
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
            "anomalies": anomalies,
            "ai_available": gemini_data.get("ai_available", False),
            "ai_error": gemini_data.get("ai_error", ""),
            "ai_explanation": gemini_data.get("ai_explanation", ""),
            "ai_key_parameters": gemini_data.get("ai_key_parameters", ""),
            "ai_maintenance_guidance": gemini_data.get("ai_maintenance_guidance", ""),
            "ai_advisory": gemini_data.get("ai_advisory", "")
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)