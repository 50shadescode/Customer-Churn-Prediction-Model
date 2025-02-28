Overview
This project predicts customer churn using machine learning models. The dataset consists of customer information, including categorical and numerical features, and the target variable indicates whether a customer has churned or not. The project employs two models: Random Forest and XGBoost, with hyperparameter tuning for optimal performance.
Dataset
Two datasets are used:
Training Dataset: customerchurn-training.csv
Testing Dataset: customerchurn-testing.csv
Requirements

To run this project, install the required dependencies:
pip install pandas numpy scikit-learn xgboost
Stepwise Implementation

1. Import Required Libraries
The necessary libraries are imported for data manipulation, preprocessing, model training, and evaluation.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

2. Load Training & Testing Datasets

The datasets are loaded into pandas DataFrames.
df_train = pd.read_csv(r"C:\Users\cex\Desktop\Data sets\customerchurn-training.csv")
df_test = pd.read_csv(r"C:\Users\cex\Desktop\Data sets\customerchurn-testing.csv")

3. Encode Categorical Variables

Label encoding is applied to categorical variables to convert them into numerical format.
label_encoders = {}
for col in df_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le  # Store for potential deployment use
4. Define Features & Target Variables

Separate the features and the target variable (Churn).

X_train = df_train.drop(columns=['Churn'])
y_train = df_train['Churn']
X_test = df_test.drop(columns=['Churn'])
y_test = df_test['Churn']
5. Standardize Features

Feature scaling is applied using StandardScaler to improve model performance.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

6. Debugging: Check Data Integrity

Check the data structure and ensure there are no missing values in the target variable.

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("First 5 rows of X_train:\n", X_train.head())
print("Missing values in y_train:", y_train.isnull().sum())

7. Handle Missing Values

Remove rows where the target variable is missing.

df_train = df_train.dropna(subset=['Churn'])
X_train = df_train.drop(columns=['Churn'])
y_train = df_train['Churn']

8. Standardize Features Again (After Cleaning)

Reapply feature scaling to ensure consistency.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

9. Train Random Forest Model

Train a Random Forest model and evaluate its accuracy.

rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

10. Hyperparameter Tuning for XGBoost

Define a grid of hyperparameters and perform GridSearchCV for optimal tuning.

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2]
}

grid_search = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

11. Evaluate the Best XGBoost Model

Retrieve the best XGBoost model and test its performance.

best_xgb = grid_search.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test_scaled)
print("\nTuned XGBoost Accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print("\nTuned XGBoost Classification Report:\n", classification_report(y_test, y_pred_best_xgb))

12. Debugging Predictions

Check a sample of predictions to verify the model's output.

print("Checking y_pred_best_xgb:", y_pred_best_xgb[:10])
print("Checking y_test:", y_test[:10])

13. Final Comparison of Models

Compare the performance of Random Forest and XGBoost.

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nTuned XGBoost Accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("\nTuned XGBoost Classification Report:\n", classification_report(y_test, y_pred_best_xgb))

Running the Script

Execute the script in a Python environment:
python customer_churn_prediction.py




