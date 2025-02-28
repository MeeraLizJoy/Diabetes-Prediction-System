import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def balance_data(df):
    """Balances the 'Outcome' feature using SMOTE."""

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Create a new balanced DataFrame
    balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Outcome')], axis=1)

    return balanced_df

def main():
    """Loads preprocessed data, balances it, and saves the balanced data."""

    try:
        df = pd.read_csv("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/data/preprocessed_diabetes.csv")
    except FileNotFoundError:
        print("Error: preprocessed_diabetes.csv not found. Please run data_preprocessing.py first.")
        return

    balanced_df = balance_data(df)

    # Save the balanced data to a new CSV file
    balanced_df.to_csv("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/data/balanced_diabetes.csv", index=False)
    print("Balanced data saved to backend/data/balanced_diabetes.csv")

if __name__ == "__main__":
    main()