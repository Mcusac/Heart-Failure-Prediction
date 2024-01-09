import joblib
import pandas as pd
from flask import Flask, render_template, request

# Load the saved model
model_filename = "heart_disease_model.joblib"
loaded_model = joblib.load(model_filename)

# Flask Setup
app = Flask(__name__)

# Function to predict HeartDisease for new input
def predict_heart_disease(input_data):
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_df)

    return predictions[0]

# Example input data (replace with actual input)
@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        'Age': float(request.form["Age"]),
        'Sex': float(request.form["Sex"]),  # Example: 1 for male, 0 for female
        'ChestPainType': float(request.form["ChestPainType"]),
        'RestingBP': float(request.form["RestingBP"]),
        'Cholesterol': float(request.form["Cholesterol"]),
        'FastingBS': float(request.form["FastingBS"]),  # Example: 0 for false, 1 for true
        'RestingECG': float(request.form["RestingECG"]),
        'MaxHR': float(request.form["MaxHR"]),
        'ExerciseAngina': float(request.form["ExerciseAngina"]),  # Example: 0 for false, 1 for true
        'Oldpeak': float(request.form["Oldpeak"]),
        'ST_Slope': float(request.form["ST_Slope"])
    }

    # Make a prediction
    prediction = predict_heart_disease(input_data)

    # Render the result on a template
    return render_template("result.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
