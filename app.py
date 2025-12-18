from flask import Flask, render_template, request
import joblib
import numpy as np


# Flask application setup
app = Flask(__name__)


# Load trained model and scaler
# These were created during the machine-learning pipeline

model = joblib.load("breast_cancer_unified_model.pkl")
scaler = joblib.load("scaler.pkl")

# Exact feature order used during training
# This ensures the input from the HTML form matches
# The exact order expected by the model and scaler

FEATURE_ORDER = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]

# Only the 3 fields that need conversion
# Home route: displays the input form
SPECIAL_MAPPING = {
    "concave_points_mean": "concave points_mean",
    "concave_points_se": "concave points_se",
    "concave_points_worst": "concave points_worst"
}

@app.route("/")
def home():
    return render_template("index.html")


# Prediction route: receives POST data from the form
# Converts input to numeric values, scales them
# Generates a prediction using the trained model

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Convert only the 3 special fields
        data_fixed = {}
        for k, v in data.items():
            if k in SPECIAL_MAPPING:
                data_fixed[SPECIAL_MAPPING[k]] = v
            else:
                data_fixed[k] = v

        # Build input vector in correct order
        input_values = [float(data_fixed[feature]) for feature in FEATURE_ORDER]

        # Scale
        input_scaled = scaler.transform([input_values])

        # Predict
        prediction = model.predict(input_scaled)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        # Render result page
        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"


# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True, port=5001)












