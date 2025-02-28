import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    """Objective function for Optuna optimization."""
    df = pd.read_csv("/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/data/balanced_diabetes.csv")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Suggest hyperparameters for Random Forest
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
    rf_max_depth = trial.suggest_int('rf_max_depth', 5, 30)
    rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 10)
    rf_min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 4)
    rf_bootstrap = trial.suggest_categorical('rf_bootstrap', [True, False])

    # Suggest hyperparameters for LightGBM
    lgbm_num_leaves = trial.suggest_int('lgbm_num_leaves', 20, 100)
    lgbm_max_depth = trial.suggest_int('lgbm_max_depth', 5, 20)
    lgbm_learning_rate = trial.suggest_float('lgbm_learning_rate', 0.01, 0.2)
    lgbm_min_child_samples = trial.suggest_int('lgbm_min_child_samples', 20, 80)
    lgbm_subsample = trial.suggest_float('lgbm_subsample', 0.6, 1.0)
    lgbm_colsample_bytree = trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0)
    lgbm_reg_alpha = trial.suggest_float('lgbm_reg_alpha', 0, 1.0)
    lgbm_reg_lambda = trial.suggest_float('lgbm_reg_lambda', 0, 1.0)

    # Initialize models with suggested hyperparameters
    rf_model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                       min_samples_split=rf_min_samples_split,
                                       min_samples_leaf=rf_min_samples_leaf, bootstrap=rf_bootstrap, random_state=42)
    lgbm_model = lgb.LGBMClassifier(num_leaves=lgbm_num_leaves, max_depth=lgbm_max_depth,
                                     learning_rate=lgbm_learning_rate,
                                     min_child_samples=lgbm_min_child_samples,
                                     subsample=lgbm_subsample, colsample_bytree=lgbm_colsample_bytree,
                                     reg_alpha=lgbm_reg_alpha, reg_lambda=lgbm_reg_lambda, random_state=42)

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

    return accuracy

def optimize_voting_ensemble():
    """Optimizes the voting ensemble using Optuna."""
    sampler = TPESampler(seed=42) #Use a TPE sampler.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=100) #Adjust n_trials as needed

    print("Best Trial:")
    trial = study.best_trial
    print("Value (Accuracy):", trial.value)
    print("Params:", trial.params)

if __name__ == "__main__":
    optimize_voting_ensemble()