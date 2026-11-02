import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
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

# Fixed maintenance recommendations based strictly on Random Forest risk level
FIXED_RECOMMENDATIONS = {
    "HIGH": (
        "Because the predicted failure risk is high, prioritize a machine inspection "
        "before continued production. Check the current operating conditions and follow "
        "the machine's maintenance procedure if an issue is identified. Continue monitoring "
        "the machine parameters."
    ),
    "MEDIUM": (
        "Because the predicted failure risk is medium, schedule a preventive inspection "
        "during the next maintenance window. Review the current operating conditions and "
        "continue monitoring the machine parameters for changes in risk."
    ),
    "LOW": (
        "Because the predicted failure risk is low, continue routine maintenance and "
        "regular monitoring of the machine parameters according to the machine's normal "
        "operating procedure."
    )
}


def get_gemini_client():
    """Securely initialize and return the Gemini API client if API key is present."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def compute_prediction(machine_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear):
    """
    Execute Random Forest model prediction on the six machine telemetry features.
    Applies the validated 0.30 decision threshold to determine risk level and machine status.
    """
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

    # 3-Tier Risk & Status Determination
    if probability >= 0.50:
        status = "CRITICAL FAILURE RISK"
        risk = "HIGH"
        status_class = "danger"
    elif probability >= THRESHOLD:
        status = "ELEVATED RISK"
        risk = "MEDIUM"
        status_class = "warning"
    else:
        status = "NORMAL OPERATION"
        risk = "LOW"
        status_class = "success"

    fixed_recommendation = FIXED_RECOMMENDATIONS[risk]

    return {
        "machine_type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "probability": prob_percent,
        "threshold": THRESHOLD * 100,
        "risk": risk,
        "status": status,
        "status_class": status_class,
        "recommendation": fixed_recommendation
    }


def parse_gemini_response(text, air_temperature, process_temperature, rotational_speed, torque, tool_wear, prob_percent, risk, fixed_recommendation):
    """Parse Gemini's structured response into the three display sections."""
    data = {
        "ai_explanation": "",
        "what_to_check": "",
        "maintenance_recommendation": fixed_recommendation
    }

    # Extract AI Explanation section
    expl_match = re.search(
        r"AI Explanation\s*:?\s*(.*?)(?=What to Check|Maintenance Recommendation|$)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if expl_match:
        cleaned = re.sub(r'[*_#`]', '', expl_match.group(1)).strip()
        # Remove any leading numbering like "1." or "A."
        cleaned = re.sub(r'^[A-Z0-9]\.\s*', '', cleaned).strip()
        if cleaned:
            data["ai_explanation"] = cleaned

    # Extract What to Check section
    check_match = re.search(
        r"What to Check\s*:?\s*(.*?)(?=Maintenance Recommendation|$)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if check_match:
        cleaned = re.sub(r'[*_#`]', '', check_match.group(1)).strip()
        cleaned = re.sub(r'^[A-Z0-9]\.\s*', '', cleaned).strip()
        if cleaned:
            data["what_to_check"] = cleaned

    # Fallback: generate explanation if Gemini did not return a valid one
    if not data["ai_explanation"]:
        data["ai_explanation"] = (
            f"The Random Forest model predicts a {prob_percent:.2f}% probability of machine failure "
            f"based on the current operating parameters. This places the machine in the {risk}-risk category. "
            f"The prediction is based on patterns learned from historical machine data."
        )

    # Fallback: generate what-to-check if Gemini did not return a valid one
    if not data["what_to_check"]:
        data["what_to_check"] = (
            f"Review the current machine readings:\n\n"
            f"Air Temperature: {air_temperature:.1f} K\n"
            f"Process Temperature: {process_temperature:.1f} K\n"
            f"Rotational Speed: {rotational_speed:.0f} rpm\n"
            f"Torque: {torque:.1f} Nm\n"
            f"Tool Wear: {tool_wear:.0f} min\n\n"
            f"Compare these values with the machine's normal operating limits and maintenance guidelines."
        )

    # Always enforce fixed recommendation — Gemini must never override this
    data["maintenance_recommendation"] = fixed_recommendation

    return data


def analyze_with_gemini(machine_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear, prob_percent, risk, status):
    """
    Generate natural-language explanation using Gemini.
    Random Forest = actual failure-risk prediction.
    Application rules = maintenance recommendation.
    Gemini = natural-language explanation.
    Never lets Gemini break prediction or invent maintenance actions.
    """
    fixed_recommendation = FIXED_RECOMMENDATIONS[risk]

    fallback_result = {
        "ai_available": False,
        "ai_error": "AI explanation unavailable. ML prediction is still available.",
        "ai_explanation": "AI explanation unavailable. ML prediction is still available.",
        "what_to_check": (
            f"Review the current machine readings:\n\n"
            f"Air Temperature: {air_temperature:.1f} K\n"
            f"Process Temperature: {process_temperature:.1f} K\n"
            f"Rotational Speed: {rotational_speed:.0f} rpm\n"
            f"Torque: {torque:.1f} Nm\n"
            f"Tool Wear: {tool_wear:.0f} min\n\n"
            f"Compare these values with the machine's normal operating limits and maintenance guidelines."
        ),
        "maintenance_recommendation": fixed_recommendation
    }

    client = get_gemini_client()
    if not client:
        return fallback_result

    prompt = f"""You are an AI Industrial Diagnostics Assistant for a Predictive Maintenance System.
A Random Forest machine learning model has already evaluated the operating telemetry and predicted the failure probability and risk level.

Supplied Telemetry and Model Prediction:
- Product Type: {machine_type}
- Air Temperature: {air_temperature:.1f} K
- Process Temperature: {process_temperature:.1f} K
- Rotational Speed: {rotational_speed:.0f} rpm
- Torque: {torque:.1f} Nm
- Tool Wear: {tool_wear:.0f} min
- Model Failure Probability: {prob_percent:.2f}%
- Model Risk Level: {risk}
- Model Machine Status: {status}

ROLE & PROHIBITIONS:
1. The Random Forest model is solely responsible for the failure prediction. You must NOT make the prediction or alter the risk level.
2. The application rules define the maintenance action. You must NOT invent a different maintenance action.
3. You must NOT diagnose a specific component (do NOT claim a bearing is damaged, do NOT claim a motor is failing, do NOT claim an electrical fault, do NOT claim overheating, do NOT claim tool breakage).
4. You must NOT claim a specific parameter definitely caused the failure or that a failure has already occurred.
5. You must NOT recommend replacing a specific component without evidence.
6. Do NOT say the machine "will fail" or "is broken". Use objective phrasing like "the model predicts", "based on patterns learned from historical data".
7. Keep language practical and easy for a mechanic to understand.

REQUIRED OUTPUT — provide exactly these 3 sections:

AI Explanation:
The Random Forest model predicts a {prob_percent:.2f}% probability of machine failure based on the current operating parameters. This places the machine in the {risk}-risk category. The prediction is based on patterns learned from historical machine data.

What to Check:
Review the current machine readings:

Air Temperature: {air_temperature:.1f} K
Process Temperature: {process_temperature:.1f} K
Rotational Speed: {rotational_speed:.0f} rpm
Torque: {torque:.1f} Nm
Tool Wear: {tool_wear:.0f} min

Compare these values with the machine's normal operating limits and maintenance guidelines.

Maintenance Recommendation:
{fixed_recommendation}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if response and response.text:
            parsed = parse_gemini_response(
                text=response.text,
                air_temperature=air_temperature,
                process_temperature=process_temperature,
                rotational_speed=rotational_speed,
                torque=torque,
                tool_wear=tool_wear,
                prob_percent=prob_percent,
                risk=risk,
                fixed_recommendation=fixed_recommendation
            )
            parsed["ai_available"] = True
            parsed["ai_error"] = ""
            return parsed
        else:
            return fallback_result

    except Exception:
        # Never expose raw exceptions, stack traces, or API keys to the user
        return fallback_result


# ==============================================================================
# PAGE 1 — Machine Inputs + ML Prediction
# ==============================================================================

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        machine_type = str(request.form.get("machine_type", "L")).strip().upper()
        if machine_type not in ["L", "M", "H"]:
            machine_type = "L"
        air_temperature = float(request.form.get("air_temperature", 300.0))
        process_temperature = float(request.form.get("process_temperature", 310.0))
        rotational_speed = float(request.form.get("rotational_speed", 1500))
        torque = float(request.form.get("torque", 40.0))
        tool_wear = float(request.form.get("tool_wear", 0))

        result = compute_prediction(machine_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear)

    return render_template("index.html", result=result)


# ==============================================================================
# AJAX Prediction Endpoint (for client-side fetch without full page reload)
# ==============================================================================

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """AJAX endpoint for Step 1 Failure Prediction."""
    try:
        machine_type = str(request.form.get("machine_type", "L")).strip().upper()
        if machine_type not in ["L", "M", "H"]:
            machine_type = "L"
        air_temperature = float(request.form.get("air_temperature", 300.0))
        process_temperature = float(request.form.get("process_temperature", 310.0))
        rotational_speed = float(request.form.get("rotational_speed", 1500))
        torque = float(request.form.get("torque", 40.0))
        tool_wear = float(request.form.get("tool_wear", 0))

        result = compute_prediction(machine_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ==============================================================================
# PAGE 2 — AI Maintenance Recommendation (separate page)
# ==============================================================================

@app.route("/recommendation", methods=["POST"])
def recommendation_page():
    """
    Page 2: AI Maintenance Recommendation.
    Receives the six machine parameters via hidden POST form from Page 1.
    Re-runs the same prediction (identical result) and calls Gemini for explanation.
    The user does NOT re-enter any values.
    """
    machine_type = str(request.form.get("machine_type", "L")).strip().upper()
    if machine_type not in ["L", "M", "H"]:
        machine_type = "L"
    air_temperature = float(request.form.get("air_temperature", 300.0))
    process_temperature = float(request.form.get("process_temperature", 310.0))
    rotational_speed = float(request.form.get("rotational_speed", 1500))
    torque = float(request.form.get("torque", 40.0))
    tool_wear = float(request.form.get("tool_wear", 0))

    # Re-compute identical prediction (same model, same threshold)
    prediction = compute_prediction(machine_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear)

    # Get Gemini explanation (with safe fallback)
    ai_data = analyze_with_gemini(
        machine_type=machine_type,
        air_temperature=air_temperature,
        process_temperature=process_temperature,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
        prob_percent=prediction["probability"],
        risk=prediction["risk"],
        status=prediction["status"]
    )

    return render_template(
        "recommendation.html",
        prediction=prediction,
        ai=ai_data
    )


if __name__ == "__main__":
    app.run(debug=True)