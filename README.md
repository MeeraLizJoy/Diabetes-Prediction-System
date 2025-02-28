# Diabetes Prediction System

This project is a web application that predicts the risk of diabetes based on user-provided health metrics. It includes a React frontend for user interaction and a Flask backend for model predictions.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## About

The Diabetes Prediction System aims to provide users with an easy-to-use tool for assessing their risk of diabetes. The application includes a BMI calculator and a prediction form that utilizes a machine learning model to estimate diabetes risk. The system consists of a React frontend and a Flask backend.

## Features

- **BMI Calculator:** Calculates the Body Mass Index based on user-provided height and weight.
- **Diabetes Risk Prediction:** Predicts the risk of diabetes based on user-entered health metrics.
- **User-Friendly Interface:** Provides a clean and intuitive interface for easy interaction.
- **Responsive Design:** Ensures the application is accessible on various devices.
- **Clear Prediction Messages:** Delivers easy to understand messages to the user based on the prediction.
- **Machine Learning Model:** Utilizes a trained machine learning model for accurate predictions.
- **Explainable AI:** Includes SHAP value analysis to help understand the model's predictions.

## Technologies Used

- **Frontend:**
  - React
  - Axios
  - CSS
  - HTML
  - JavaScript
- **Backend:**
  - Flask
  - Flask-CORS
  - Pandas
  - Scikit-learn
  - LightGBM
  - Shap
  - Optuna
  - Gunicorn

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm (Node Package Manager)
- Python (v3.7 or higher)
- pip (Python Package Installer)
- Git

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd DiabetesPredictionSystem
    ```

2.  **Set up the backend:**

    ```bash
    cd backend
    python3 -m venv dia_venv
    source dia_venv/bin/activate  # On macOS/Linux
    dia_venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    cd ..
    ```

3.  **Set up the frontend:**

    ```bash
    cd frontend
    npm install
    cd ..
    ```

### Running the Application

1.  **Start the Flask backend:**

    ```bash
    cd backend
    source dia_venv/bin/activate  # On macOS/Linux
    dia_venv\Scripts\activate  # On Windows
    python app.py
    ```

2.  **Start the React frontend:**

    ```bash
    cd frontend
    npm start
    ```

3.  **Open the application in your browser:**

    Visit `http://localhost:3000` in your web browser.


## Model Details

The model used for diabetes prediction is an ensemble model trained using LightGBM and Random Forest. It was trained on the Pima Indians Diabetes Database and achieved an F1-score of 83.7% and an ROC-AUC of 83.0%. Feature engineering and hyperparameter tuning were performed to optimize model performance. SHAP value analysis was used to understand feature importance and model behavior.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
