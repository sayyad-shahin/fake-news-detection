"""Flask web app + JSON API for fake news detection."""
from flask import Flask, jsonify, render_template, request

from predict import predict

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(predict(data.get("text", "")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
