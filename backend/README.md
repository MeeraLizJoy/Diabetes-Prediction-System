# Diabetes Prediction System (Backend)

This project develops a machine learning model for predicting diabetes based on the Pima Indians Diabetes Database. It includes data preprocessing, feature engineering, model training, hyperparameter tuning, ensembling, and explainable AI (XAI) analysis.

## Project Status

* Data preprocessing, including missing value imputation and outlier removal, is complete.
* Class imbalance has been addressed using SMOTE.
* Feature engineering has been performed, creating interaction and polynomial features.
* Baseline models (Logistic Regression, Random Forest, SVM, XGBoost, LightGBM) have been trained and evaluated.
* Hyperparameter tuning (GridSearchCV, RandomizedSearchCV, and Bayesian Optimization with Optuna) has been performed for each model.
* Model ensembling (voting and stacking) has been implemented, with the voting ensemble showing the best results.
* XAI analysis using SHAP values has been conducted to understand feature importance and model behavior.
* The final voting ensemble model has been trained and saved.

## Data

The dataset used is the Pima Indians Diabetes Database, obtained from the UCI Machine Learning Repository.

* `diabetes.csv`: Original dataset.
* `preprocessed_diabetes.csv`: Dataset after initial preprocessing and final feature enfineering.
* `balanced_diabetes.csv`: Dataset after addressing class imbalance (SMOTE).

## Data Preprocessing

* Missing values were imputed using the median.
* Outliers were removed using the IQR method.
* Class imbalance was addressed using SMOTE.

## Feature Engineering

* Created interaction features (e.g., `Glucose * BMI`).
* Added polynomial features (e.g., `Glucose^2`, `BMI^2`).
* Created categorical features (e.g., `BMI_Category`, `Age_Category`) and one-hot encoded them.
* Added ratio features (Glucose/Age)

## Model Training and Evaluation

* Baseline models were trained and evaluated on the balanced dataset.
* Hyperparameter tuning was performed using GridSearchCV, RandomizedSearchCV, and Optuna.
* Model ensembling (voting and stacking) was implemented.
* The voting ensemble, with hyperparameters tuned using Optuna, achieved the best performance.

## Final Model Evaluation Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.83   |
| Precision | 0.813  |
| Recall    | 0.861  |
| F1-score  | 0.837  |
| ROC-AUC   | 0.830  |

## Hyperparameter Tuning

* Hyperparameter tuning was performed using GridSearchCV, RandomizedSearchCV, and Bayesian optimization with Optuna.
* Optuna was used to optimize the voting ensemble, resulting in improved performance.
* The best parameters of the voting ensemble are:
    * Random Forest: `{'n_estimators': 50, 'max_depth': 21, 'min_samples_split': 10, 'min_samples_leaf': 1, 'bootstrap': False}`
    * LightGBM: `{'num_leaves': 33, 'max_depth': 6, 'learning_rate': 0.03059, 'min_child_samples': 78, 'subsample': 0.963, 'colsample_bytree': 0.951, 'reg_alpha': 0.908, 'reg_lambda': 0.789}`

## Model Ensembling

* Voting and stacking ensembles were implemented.
* The voting ensemble performed better than the stacking ensemble.

## Explainable AI (XAI) Analysis

* SHAP values were calculated to understand feature importance and model behavior.
* SHAP summary and waterfall plots were generated.
* Key insights:
    * `BMI_Glucose_Interaction`, `Age`, and `Glucose` are the most important features.
    * High values of these features increase the likelihood of diabetes.
    * The waterfall plot shows how individual features contribute to specific predictions.

## Project Structure

backend/
├── data/
│   ├── diabetes.csv
│   ├── preprocessed_diabetes.csv
│   └── balanced_diabetes.csv
├── scripts/
│   ├── data_preprocessing.py
│   ├── imbalance_handling.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── hyperparameter_tuning_logistic_regression.py
│   ├── hyperparameter_tuning_random_forest.py
│   ├── hyperparameter_tuning_svm.py
│   ├── hyperparameter_tuning_xgboost.py
│   ├── hyperparameter_tuning_lightgbm.py
│   ├── model_voting_ensembling.py
│   ├── model_stacking_ensembling.py
│   ├── bayesian_tuning_voting.py
│   ├── final_model_training.py
│   └── xai_analysis.py
├── models/
│   └──final_diabetes_model.pkl
└── README.md











