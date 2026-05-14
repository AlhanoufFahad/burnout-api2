import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

# Initialize Flask application
app = Flask(__name__)

# Define the paths to your model and scaler files
# Ensure these files ('burnout_model_fixed.pkl' and 'scaler_fixed.pkl')
# are present in the same directory as your Flask app when deploying to Render.
MODEL_PATH = "burnout_model_fixed.pkl"
SCALER_PATH = "scaler_fixed.pkl"

model = None
scaler = None

try:
    # Load the pre-trained model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"✅ Scaler loaded from {SCALER_PATH}")
except FileNotFoundError:
    print(f"Error: Model or scaler files not found. Expected: {MODEL_PATH}, {SCALER_PATH}")
    print("Please ensure these files are in the deployment directory.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

# Re-define the predict_burnout function (from cell 5M3tG2JTcRQg in your notebook)
def predict_burnout(designation, resource_allocation, mental_fatigue):
    global model, scaler # Access the globally loaded model and scaler

    if model is None or scaler is None:
        return "Model not loaded. Cannot predict."

    # Rule-based cases as defined in your notebook
    if mental_fatigue <= 4:
        return "Low Burnout"
    elif mental_fatigue >= 17:
        return "High Burnout"
    else:
        # Raw input array should match the 3 features ('Designation', 'Resource Allocation', 'Mental Fatigue Score')
        # that your model was trained on.
        input_data = np.array([[designation, resource_allocation, mental_fatigue]], dtype=np.float32)

        # Apply scaling using the loaded scaler
        # Note: The scaler was fitted on the training data's numerical columns.
        scaled_input = scaler.transform(input_data)

        # Make prediction with the loaded model
        prediction = model.predict(scaled_input)[0]

        # Convert numerical prediction to burnout level string
        if prediction == 0:
            return "Low Burnout"
        elif prediction == 1:
            return "Medium Burnout"
        else:
            return "High Burnout"

# Re-define the burnout_recommendation function (from cell SlwjTI1seJk- in your notebook)
def burnout_recommendation(burnout_level_str):
    if burnout_level_str == "Low Burnout":
        return "Your stress level is low. Keep maintaining a healthy work-life balance."
    elif burnout_level_str == "Medium Burnout":
        return "You may be experiencing moderate burnout. Consider taking short breaks and managing workload."
    elif burnout_level_str == "High Burnout":
        return "High burnout detected. We recommend rest, reducing workload, and seeking support if needed."
    else:
        return "No specific recommendation available for this burnout level."

@app.route('/predict', methods=['POST'])
def predict_api():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded. Please check server logs."}), 500

    data = request.get_json(force=True)

    # Extract input features from the request JSON
    try:
        designation = float(data['designation'])
        resource_allocation = float(data['resource_allocation'])
        mental_fatigue = float(data['mental_fatigue'])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input data: {e}. Expected 'designation', 'resource_allocation', and 'mental_fatigue' as numbers."}), 400

    # Get burnout prediction
    burnout_level = predict_burnout(designation, resource_allocation, mental_fatigue)

    # Get recommendation based on the burnout level
    recommendation = burnout_recommendation(burnout_level)

    return jsonify({
        "burnout_level": burnout_level,
        "recommendation": recommendation
    })


# To run this API locally for testing:
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))

# For deployment on platforms like Render, a WSGI server (e.g., Gunicorn)
# will typically be used to run the 'app' object. You usually don't run app.run() directly
# in the deployed environment.
