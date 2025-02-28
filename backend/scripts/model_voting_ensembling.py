import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def create_voting_ensemble():
    """Creates a voting ensemble using Random Forest and LightGBM."""
    df = pd.read_csv("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/data/balanced_diabetes.csv")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models with best parameters found during hyperparameter tuning
    rf_model = RandomForestClassifier(n_estimators=200, min_samples_split=10, min_samples_leaf=1, max_depth=20, bootstrap=False, random_state=42)
    lgbm_model = lgb.LGBMClassifier(subsample=0.8, reg_lambda=0.5, reg_alpha=0, num_leaves=20, min_child_samples=40, max_depth=-1, learning_rate=0.05, colsample_bytree=1.0, random_state=42)

    # Create voting ensemble
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_model), ('lgbm', lgbm_model)],
        voting='hard'  # 'hard' for majority voting, 'soft' for probability averaging
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

    print("Voting Ensemble Results:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC-AUC:", roc_auc)

if __name__ == "__main__":
    create_voting_ensemble()