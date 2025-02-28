import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def create_stacking_ensemble():
    """Creates a stacking ensemble using Random Forest and LightGBM."""
    df = pd.read_csv("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/data/balanced_diabetes.csv")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize base models with best parameters found during hyperparameter tuning
    rf_model = RandomForestClassifier(n_estimators=200, min_samples_split=10, min_samples_leaf=1, max_depth=20, bootstrap=False, random_state=42)
    lgbm_model = lgb.LGBMClassifier(subsample=0.8, reg_lambda=0.5, reg_alpha=0, num_leaves=20, min_child_samples=40, max_depth=-1, learning_rate=0.05, colsample_bytree=1.0, random_state=42)

    # Initialize meta-learner (Logistic Regression)
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)

    # Create stacking ensemble
    stacking_clf = StackingClassifier(
        estimators=[('rf', rf_model), ('lgbm', lgbm_model)],
        final_estimator=meta_learner,
        cv=5  # Use 5-fold cross-validation for stacking
    )

    # Train the ensemble
    stacking_clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = stacking_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print("Stacking Ensemble Results:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC-AUC:", roc_auc)

if __name__ == "__main__":
    create_stacking_ensemble()