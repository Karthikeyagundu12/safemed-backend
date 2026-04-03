from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.h5")

@app.route('/')
def home():
    return "SafeMed AI Backend Running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("📩 Request received")

        # Get image file
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"})

        file = request.files['image']

        # Open & preprocess image
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))  # MUST match training size

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        print("🔍 Prediction value:", prediction)

        # Decision logic
        if prediction > 0.7:
            result = "Real"
            confidence = prediction * 100
        else:
            result = "Fake"
            confidence = (1 - prediction) * 100

        return jsonify({
            "prediction": result,
            "confidence": round(float(confidence), 2)
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)})

# Run server
if __name__ == "__main__":
             app.run()