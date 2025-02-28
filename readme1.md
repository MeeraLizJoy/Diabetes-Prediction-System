# Diabetes Prediction Project

## Project Overview

This project aims to develop a machine learning model to predict diabetes based on the Pima Indians Diabetes Database. The dataset contains various medical predictor variables and a target variable (Outcome) indicating whether a patient has diabetes.

## Dataset

* **Source:** National Institute of Diabetes and Digestive and Kidney Diseases
* **Description:** The dataset includes diagnostic measurements from female Pima Indian individuals aged 21 and older.
* **File:** `data/diabetes.csv`

## Initial Data Exploration (EDA)

### Dataset Summary

* **Shape:** 768 rows, 9 columns
* **Columns:**
    * `Pregnancies`: Number of pregnancies
    * `Glucose`: Glucose concentration
    * `BloodPressure`: Diastolic blood pressure (mm Hg)
    * `SkinThickness`: Triceps skin fold thickness (mm)
    * `Insulin`: 2-Hour serum insulin (mu U/ml)
    * `BMI`: Body mass index
    * `DiabetesPedigreeFunction`: Diabetes pedigree function
    * `Age`: Age (years)
    * `Outcome`: Class variable (0 or 1, where 1 indicates diabetes)
* **Data Types:** Numerical (int64 and float64)
* **Missing Values:** None (but zero values are present in some columns)

### Key Observations

* **Zero Values:** Several features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`) contain zero values, which are likely invalid and require handling.
* **Imbalanced Data:** The `Outcome` variable is imbalanced (500 instances of 0, 268 instances of 1).
* **Skewed Distributions:** Features like `Pregnancies`, `SkinThickness`, `Insulin`, and `DiabetesPedigreeFunction` are heavily skewed.
* **Outliers:** Box plots revealed outliers in several features.
* **Correlations:**
    * `Glucose` shows a strong positive correlation with `Outcome`.
    * `Age` and `Pregnancies` are strongly correlated.
    * `BMI` shows a moderate positive correlation with `Outcome`.
* **Pair Plots:** Visualized relationships between all feature pairs, confirming strong relationships between `Glucose` and `Outcome`, and also showing how the `Outcome` is distributed among all other features.

### Visualizations

* **Histograms:** Showed the distribution of each numerical feature, highlighting skewedness and zero values.
![alt text](/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/visualizations/histograms.png)
* **Box Plots:** Identified outliers and confirmed skewed distributions.
![alt text](/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/visualizations/boxplots.png)
* **Correlation Matrix:** Visualized correlations between features.
![alt text](/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/visualizations/correlationmatrix.png)
* **Pair Plots:** Displayed relationships between all pairs of features, colored by `Outcome`.
![alt text](/Users/meeralizjoy/Desktop/DiabetesPredictionSystem/backend/visualizations/paiplts.png)

### Next Steps

* **Data Preprocessing:**
    * Handle zero values (imputation, removal, etc.).
    * Address outliers.
    * Perform feature engineering (create interaction features, transform skewed features).
    * Apply feature scaling.
    * Perform feature selection.
* **Model Selection and Training:**
    * Train baseline machine learning models.
    * Implement cross-validation.
* **Documentation:** Continue to update the `README.md` with progress and findings.




## Project Status

* Data preprocessing and outlier removal have been implemented.
* Baseline models (Logistic Regression, Random Forest, SVM, XGBoost, LightGBM) have been trained and evaluated.
* Hyperparameter tuning for all models is now being performed.

## Baseline Model Evaluation Results

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC | CV Accuracy |
| ------------------- | -------- | --------- | ------ | -------- | ------- | ----------- |
| Logistic Regression | 0.755    | 0.736     | 0.802  | 0.767    | 0.754   | 0.779         |
| Random Forest       | 0.800    | 0.775     | 0.851  | 0.811    | 0.799   | 0.812         |
| SVM                 | 0.775    | 0.741     | 0.851  | 0.793    | 0.774   | 0.797         |
| XGBoost             | 0.791    | 0.761     | 0.851  | 0.804    | 0.789   | 0.792         |
| LightGBM            | 0.785    | 0.773     | 0.812  | 0/792    | 0.785   | 0.799         |

          Accuracy  Precision    Recall  F1-score   ROC-AUC  CV Accuracy
Logistic Regression     0.785   0.773585  0.811881  0.792271  0.784728      0.80125
Random Forest           0.800   0.765217  0.871287  0.814815  0.799280      0.81625
SVM                     0.720   0.731959  0.702970  0.717172  0.720172      0.72000
XGBoost                 0.790   0.756522  0.861386  0.805556  0.789279      0.80125
LightGBM                0.775   0.745614  0.841584  0.790698  0.774327      0.81625

## Hyperparameter Tuning

The following models will be tuned using `GridSearchCV` (for Logistic Regression and SVM) and `RandomizedSearchCV` (for Random Forest, XGBoost, and LightGBM):

* Logistic Regression
* Random Forest
* SVM
* XGBoost
* LightGBM

Hyperparameter grids have been defined for each model to explore a range of possible values. The best hyperparameters will be selected based on cross-validation accuracy.

## Outlier Removal

Outliers were removed from the dataset using the IQR (Interquartile Range) method. This was done to improve the robustness of the models and prevent them from being overly influenced by extreme values.

## Balancing Data
* `balanced_diabetes.csv`: Dataset after addressing class imbalance using SMOTE.


Logistic Regression Tuning Results:
Accuracy: 0.79
Precision: 0.7864077669902912
Recall: 0.801980198019802
F1-score: 0.7941176470588235
ROC-AUC: 0.7898789878987899
Best Parameters: {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}

Random Forest Tuning Results:
Accuracy: 0.81
Precision: 0.7837837837837838
Recall: 0.8613861386138614
F1-score: 0.8207547169811321
ROC-AUC: 0.8094809480948095
Best Parameters: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 20, 'bootstrap': False}

SVM Tuning Results:
Accuracy: 0.77
Precision: 0.7235772357723578
Recall: 0.8811881188118812
F1-score: 0.7946428571428571
ROC-AUC: 0.7688768876887688
Best Parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}

XGBoost Tuning Results:
Accuracy: 0.775
Precision: 0.7413793103448276
Recall: 0.8514851485148515
F1-score: 0.7926267281105991
ROC-AUC: 0.7742274227422742
Best Parameters: {'subsample': 1.0, 'reg_lambda': 0.5, 'reg_alpha': 0.1, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 1.0}

LightGBM Tuning Results:
Accuracy: 0.8
Precision: 0.7699115044247787
Recall: 0.8613861386138614
F1-score: 0.8130841121495327
ROC-AUC: 0.7993799379937994
Best Parameters: {'subsample': 0.8, 'reg_lambda': 0.5, 'reg_alpha': 0, 'num_leaves': 20, 'min_child_samples': 40, 'max_depth': -1, 'learning_rate': 0.05, 'colsample_bytree': 1.0}

Voting Ensemble Results:
Accuracy: 0.805
Precision: 0.7924528301886793
Recall: 0.8316831683168316
F1-score: 0.8115942028985508
ROC-AUC: 0.8047304730473047

Stacking Ensemble Results:
Accuracy: 0.8
Precision: 0.7747747747747747
Recall: 0.8514851485148515
F1-score: 0.8113207547169812
ROC-AUC: 0.7994799479947994

Trial 99 finished with value: 0.82 and parameters: {'rf_n_estimators': 237, 'rf_max_depth': 19, 'rf_min_samples_split': 9, 'rf_min_samples_leaf': 1, 'rf_bootstrap': False, 'lgbm_num_leaves': 52, 'lgbm_max_depth': 8, 'lgbm_learning_rate': 0.014824589444441781, 'lgbm_min_child_samples': 74, 'lgbm_subsample': 0.8682044229448131, 'lgbm_colsample_bytree': 0.9628977711788882, 'lgbm_reg_alpha': 0.7444269015363137, 'lgbm_reg_lambda': 0.7162696401534205}. Best is trial 53 with value: 0.83.
Best Trial:
Value (Accuracy): 0.83
Params: {'rf_n_estimators': 50, 'rf_max_depth': 21, 'rf_min_samples_split': 10, 'rf_min_samples_leaf': 1, 'rf_bootstrap': False, 'lgbm_num_leaves': 33, 'lgbm_max_depth': 6, 'lgbm_learning_rate': 0.030590506068458655, 'lgbm_min_child_samples': 78, 'lgbm_subsample': 0.9634328027871369, 'lgbm_colsample_bytree': 0.951133577109212, 'lgbm_reg_alpha': 0.9078880441603634, 'lgbm_reg_lambda': 0.789268400354689}

Final Voting Ensemble Results (Optuna Tuned):
Accuracy: 0.83
Precision: 0.8130841121495327
Recall: 0.8613861386138614
F1-score: 0.8365384615384616
ROC-AUC: 0.8296829682968297