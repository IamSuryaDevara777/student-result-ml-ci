from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "student_result_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Student Result Prediction API"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model_not_found"}), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "invalid_json"}), 400

    required_fields = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "missing_fields",
            "missing_fields": missing_fields
        }), 400

    try:
        features = [[
            float(data["attendance"]),
            float(data["internal_marks"]),
            float(data["assignment_marks"]),
            float(data["previous_score"])
        ]]

        prediction_value = int(model.predict(features)[0])
        prediction = "PASS" if prediction_value == 1 else "FAIL"

        return jsonify({
            "prediction": prediction,
            "prediction_value": prediction_value,
            "result": prediction
        }), 200

    except (ValueError, TypeError):
        return jsonify({
            "error": "invalid_input",
            "message": "Input values must be numeric"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
