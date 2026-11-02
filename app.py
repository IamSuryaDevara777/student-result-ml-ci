
from flask import Flask, request, jsonify, g, Response
import joblib
import os
import pandas as pd
import logging
import sys
import time
import uuid

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

app = Flask(__name__)

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

# Prometheus metrics
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the ML model is loaded"
)

MODEL_PATH = "student_result_model.pkl"
model = None

# Load machine learning model
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH
        )
    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("event=model_loading_failed")
else:
    MODEL_LOADED.set(0)

    logger.error(
        "event=model_file_missing model_file=%s",
        MODEL_PATH
    )


# Log incoming requests
@app.before_request
def log_request():

    g.request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )

    g.start_time = time.perf_counter()

    logger.info(
        "event=request_received request_id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path
    )


# Log outgoing responses and record Prometheus metrics
@app.after_request
def log_response(response):

    duration_seconds = (
        time.perf_counter() - g.start_time
    )

    duration_ms = round(duration_seconds * 1000, 2)

    # Exclude Prometheus scrape requests from HTTP metrics
    if request.path != "/metrics":

        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration_seconds)

    log_function = (
        logger.info
        if response.status_code < 400
        else logger.warning
    )

    log_function(
        "event=response_sent request_id=%s method=%s path=%s status=%s duration_ms=%s",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    response.headers["X-Request-ID"] = g.request_id

    return response


# Home endpoint
@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Student Result Prediction API-Updated Version"
    }), 200


# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


# Prometheus metrics endpoint
@app.route("/metrics", methods=["GET"])
def metrics():

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


# Student result prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:

        logger.error(
            "event=prediction_failed reason=model_not_found request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "model_not_found"
        }), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):

        logger.warning(
            "event=invalid_json request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "invalid_json"
        }), 400

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

        logger.warning(
            "event=missing_fields fields=%s request_id=%s",
            missing_fields,
            g.request_id
        )

        return jsonify({
            "error": "missing_fields",
            "missing_fields": missing_fields
        }), 400

    inference_started = False

    try:

        features = pd.DataFrame([{
            "attendance": float(data["attendance"]),
            "internal_marks": float(data["internal_marks"]),
            "assignment_marks": float(data["assignment_marks"]),
            "previous_score": float(data["previous_score"])
        }])

        logger.info(
            "event=prediction_started feature_names=%s request_id=%s",
            list(features.columns),
            g.request_id
        )

        inference_started = True

        prediction_value = int(
            model.predict(features)[0]
        )

        PREDICTIONS.labels(result="success").inc()

        prediction = (
            "PASS" if prediction_value == 1 else "FAIL"
        )

        logger.info(
            "event=prediction_success prediction=%s request_id=%s",
            prediction,
            g.request_id
        )

        return jsonify({
            "prediction": prediction,
            "prediction_value": prediction_value,
            "result": prediction
        }), 200

    except (ValueError, TypeError):

        if inference_started:
            PREDICTIONS.labels(result="error").inc()

        logger.warning(
            "event=invalid_input request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "invalid_input"
        }), 400

    except Exception:

        if inference_started:
            PREDICTIONS.labels(result="error").inc()

        logger.exception(
            "event=prediction_error request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "internal_server_error"
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
