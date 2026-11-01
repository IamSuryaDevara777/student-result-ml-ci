from flask import Flask, request, jsonify

app = Flask(__name__)

# Add this health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Your model prediction logic goes here
    data = request.get_json()
    # Example dummy prediction response:
    return jsonify({"prediction": 1}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
