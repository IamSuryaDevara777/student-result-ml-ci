
from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "student_result_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not found"}), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON input"}), 400

    required_fields = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    if not all(key in data for key in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        features = [[
            float(data["attendance"]),
            float(data["internal_marks"]),
            float(data["assignment_marks"]),
            float(data["previous_score"])
        ]]

        prediction = int(model.predict(features)[0])

        result = "PASS" if prediction == 1 else "FAIL"

        return jsonify({
            "prediction": prediction,
            "result": result
        }), 200

    except (ValueError, TypeError):
        return jsonify({"error": "Input values must be numeric"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
