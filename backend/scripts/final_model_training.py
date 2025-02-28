import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

def train_final_voting_ensemble():
    """Trains the final voting ensemble with Optuna's best hyperparameters."""
    df = pd.read_csv("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/data/balanced_diabetes.csv")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Best hyperparameters from Optuna
    rf_params = {
        'n_estimators': 50,
        'max_depth': 21,
        'min_samples_split': 10,
        'min_samples_leaf': 1,
        'bootstrap': False,
        'random_state': 42
    }

    lgbm_params = {
        'num_leaves': 33,
        'max_depth': 6,
        'learning_rate': 0.030590506068458655,
        'min_child_samples': 78,
        'subsample': 0.9634328027871369,
        'colsample_bytree': 0.951133577109212,
        'reg_alpha': 0.9078880441603634,
        'reg_lambda': 0.789268400354689,
        'random_state': 42
    }

    # Initialize models with best hyperparameters
    rf_model = RandomForestClassifier(**rf_params)
    lgbm_model = lgb.LGBMClassifier(**lgbm_params)

    # Create voting ensemble
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_model), ('lgbm', lgbm_model)],
        voting='hard'
    )

    # Train the ensemble
    voting_clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print("Final Voting Ensemble Results (Optuna Tuned):")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC-AUC:", roc_auc)

    # Save the model using pickle
    with open("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/models/final_diabetes_model.pkl", "wb") as f: # or final_diabetes_model.joblib
        pickle.dump(voting_clf, f) # or joblib.dump(voting_clf, f)

    print("Final model saved as final_diabetes_model.pkl")

if __name__ == "__main__":
    final_model = train_final_voting_ensemble()
    