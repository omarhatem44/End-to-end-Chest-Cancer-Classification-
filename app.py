from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set environment variables
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

app = Flask(__name__)
CORS(app)

# Initialize once (important for performance)
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()  # ✅ moved outside main

# -------------------- ROUTES --------------------

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


# ❌ REMOVED TRAIN ROUTE (bad practice in production)
# Training should be done offline


@app.route("/predict", methods=['POST'])
def predictRoute():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        image = data['image']

        # Decode and save image
        decodeImage(image, clApp.filename)

        # Prediction
        result = clApp.classifier.predict()

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Health check endpoint (important for Kubernetes)
@app.route("/health", methods=['GET'])
def health():
    return jsonify({"status": "ok"})


# -------------------- MAIN --------------------

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)