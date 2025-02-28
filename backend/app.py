from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load the trained model
with open("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/models/final_diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

def perform_feature_engineering(data):
    """Performs feature engineering on the input data."""
    df = pd.DataFrame([data])

    # Convert string values to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Interaction Features
    df['BMI_Glucose_Interaction'] = df['BMI'] * df['Glucose']
    df['Age_Pregnancies_Interaction'] = df['Age'] * df['Pregnancies']
    df['BP_SkinThickness_Interaction'] = df['BloodPressure'] * df['SkinThickness']

    # Polynomial Features
    df['Glucose_Squared'] = df['Glucose'] ** 2
    df['BMI_Squared'] = df['BMI'] ** 2

    # Ratio Features
    df['Glucose_Age_Ratio'] = df['Glucose'] / df['Age']

    # BMI Category
    df['BMI_category_Normal'] = (df['BMI'] >= 18.5) & (df['BMI'] < 25)
    df['BMI_category_Overweight'] = (df['BMI'] >= 25) & (df['BMI'] < 30)
    df['BMI_category_Obese'] = df['BMI'] >= 30

    # Age Category
    df['Age_category_Meddle_Aged'] = (df['Age'] >= 35) & (df['Age'] <= 60)
    df['Age_category_Senior'] = df['Age'] > 60

    return df

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts diabetes outcome based on input features."""
    try:
        data = request.get_json()
        print("Received data:", data)  # Add this line
        input_data = perform_feature_engineering(data)

        # Ensure data is numeric and handle potential NaN values
        input_data = input_data.apply(pd.to_numeric, errors='coerce')
        input_data = input_data.fillna(input_data.mean())

        print("Data after engineering and cleaning:", input_data) #add this line.

        prediction = model.predict(input_data)[0]
        print("Prediction:", prediction) #add this line.
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print("Error:", str(e)) #add this line.
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=Flase)