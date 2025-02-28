import pandas as pd
import numpy as np

def impute_zero_values(df):
    """Replaces zero values with the median for specified columns."""
    columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column  in columns_to_impute:
        df[column] = df[column].replace(0, df[column].median())
    return df

def create_features(df):
    """Creates new features based on existing ones."""
    # BMI Category
    df['BMI_category'] = pd.cut(df['BMI'], bins = [0, 18.5, 25, 30, 100], labels = ['Underweight', 'Normal', 'Overweight', 'Obese'])

    # Age Category
    df['Age_category'] = pd.cut(df['Age'], bins = [20, 35, 50, 81], labels = ['Young', 'Meddle_Aged', 'Senior'])

    # Interaction Features
    df['BMI_Glucose_Interaction'] = df['BMI'] * df['Glucose']
    df['Age_Pregnancies_Interaction'] = df['Age'] * df['Pregnancies']
    df['BP_SkinThickness_Interaction'] = df['BloodPressure'] * df['SkinThickness']

    # Polynomial Features (Glucose and BMI)
    df['Glucose_Squared'] = df['Glucose']**2
    df['BMI_Squared'] = df['BMI']**2

    # Glucose/Age ratio
    df['Glucose_Age_Ratio'] = df['Glucose'] / (df['Age'] + 1)

    return df

# Scaling Features
from sklearn.preprocessing import StandardScaler

def scale_features(df):
    """Scales numerical features using StandardScaler"""
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Glucose_Interaction', 'Age_Pregnancies_Interaction']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

# One Hot Encoding
def one_hot_encode(df):
    """One Hot Encodes categorical variables"""
    df = pd.get_dummies(df, columns=['BMI_category', 'Age_category'], drop_first=True)
    return df

def remove_outliers_iqr(df, columns):
    """Removes outliers using the IQR method for specified columns."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 = Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def preprocess_data(df, remove_outliers = False):
    """Combines all preprocessing steps into one function."""
    df = impute_zero_values(df)
    if remove_outliers:
        outlier_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        df = remove_outliers_iqr(df, outlier_columns)
    df = create_features(df)
    df = scale_features(df)
    df = one_hot_encode(df)
    return df

def save_preprocessed_data(df, filename="/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/data/preprocessed_diabetes.csv"):
    """Saves the preprocessed DataFrame to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Preprocessed data saved to {filename}")

if __name__ == "__main__":
    df = pd.read_csv("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/data/diabetes.csv")
    df = preprocess_data(df)
    save_preprocessed_data(df)