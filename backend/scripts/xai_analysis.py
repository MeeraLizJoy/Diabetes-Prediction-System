import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def analyze_model():
    """Performs XAI analysis using SHAP values."""
    df = pd.read_csv("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/data/balanced_diabetes.csv")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the trained model
    with open("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/models/final_diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Explicitly convert all columns to numeric, handling errors
    for col in X_test.columns:
        try:
            X_test[col] = pd.to_numeric(X_test[col], errors='raise')  # Raise error on non-numeric
        except ValueError as e:
            print(f"Error converting column '{col}': {e}")
            return  # Stop if a column fails conversion

    # Handle non-finite values (NaN, inf, -inf)
    X_test = X_test.replace([np.inf, -np.inf], np.nan) #replace inf values with nan.
    X_test = X_test.fillna(X_test.mean()) #fill nan values with the mean.

    # Access the LightGBM model correctly
    lgbm_model = model.named_estimators_['lgbm'] #access the model by name.
    # Initialize SHAP explainer (using TreeExplainer for tree-based models)
    explainer = shap.TreeExplainer(lgbm_model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/visualizations/shap_summary_plot.png")
    plt.show()

    # Waterfall plot (for a single prediction)
    shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test.iloc[0], feature_names=X_test.columns))
    plt.savefig("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/visualizations/shap_waterfall_plot.png")

if __name__ == "__main__":
    analyze_model()