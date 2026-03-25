# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:38:15 2026

@author: Hp
"""

# Filename: diabetes_analysis_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
FILE_PATH = r"\wsl.localhost\Ubuntu-24.04\home\boyeinstein\documents\diabetes_prediction_dataset.csv"
TARGET_COLUMN = 'diabetes'
FEATURE_COLUMNS = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose']
# Ensure all specified feature columns are present except the target
# If the dataset has more columns, they will be ignored by this selection.
# If fewer, adjustments might be needed.

# --- Data Loading and Initial Inspection ---
print("--- Loading Data ---")
try:
    df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully.")
    print(f"Shape of the dataset: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file was not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# --- Ensure correct columns are selected ---
all_columns_present = [TARGET_COLUMN] + FEATURE_COLUMNS
if not all(col in df.columns for col in all_columns_present):
    missing_cols = [col for col in all_columns_present if col not in df.columns]
    print(f"\nError: Missing columns in the dataset: {missing_cols}")
    print("Available columns:", df.columns.tolist())
    exit()

# Select only the relevant columns
try:
    df = df[all_columns_present]
    print(f"\nDataset filtered to {df.shape[1]} columns.")
except Exception as e:
    print(f"An error occurred during column selection: {e}")
    exit()

# --- Exploratory Data Analysis (EDA) ---
print("\n--- Performing Exploratory Data Analysis (EDA) ---")

print("\n--- Dataset Info ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() > 0:
    print("\nNote: Missing values detected. Basic imputation strategies could be added here if needed.")

# --- Visualizations ---
print("\n--- Generating Visualizations (will be saved as PNG files) ---")

# Plotting distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x=TARGET_COLUMN, data=df)
plt.title('Distribution of Diabetes (0: No, 1: Yes)')
plt.savefig('diabetes_distribution.png')
plt.close()
print("Saved: diabetes_distribution.png")

# Plotting distributions of numerical features
numerical_features = df.select_dtypes(include=np.number).columns.drop(TARGET_COLUMN)
if not numerical_features.empty:
    df[numerical_features].hist(bins=15, figsize=(15, 10))
    plt.suptitle('Distribution of Numerical Features')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig('numerical_features_distribution.png')
    plt.close()
    print("Saved: numerical_features_distribution.png")

# Plotting correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.savefig('correlation_matrix.png')
plt.close()
print("Saved: correlation_matrix.png")

# Plotting correlation with target
if TARGET_COLUMN in correlation_matrix.columns:
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlation_matrix[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False).index,
                y=correlation_matrix[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False).values)
    plt.title(f'{TARGET_COLUMN} Correlation with Other Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('target_correlation.png')
    plt.close()
    print("Saved: target_correlation.png")

# Plotting categorical features (example: gender, smoking_history)
categorical_features = df.select_dtypes(include='object').columns
if 'gender' in categorical_features:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='gender', hue=TARGET_COLUMN, data=df)
    plt.title('Diabetes Distribution by Gender')
    plt.savefig('gender_vs_diabetes.png')
    plt.close()
    print("Saved: gender_vs_diabetes.png")

if 'smoking_history' in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='smoking_history', hue=TARGET_COLUMN, data=df)
    plt.title('Diabetes Distribution by Smoking History')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('smoking_history_vs_diabetes.png')
    plt.close()
    print("Saved: smoking_history_vs_diabetes.png")

print("EDA complete. Visualizations saved in the script directory.")

# --- Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Separate features and target
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Identify categorical and numerical features for preprocessing
# Assuming 'gender' and 'smoking_history' are objects/strings, others numerical
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create preprocessing pipelines for numerical and categorical features
# For numerical features: scale them
# For categorical features: one-hot encode them
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep any other columns (shouldn't be any if FEATURE_COLUMNS is precise)
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")

# Fit and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after preprocessing for models that might need them
# This is a bit complex for ColumnTransformer, let's simplify for now and assume models don't strictly need names
# Note: Some models like XGBoost can handle feature names, but for a general script, this is often omitted.

print("Preprocessing complete.")

# --- Machine Learning Models (Scikit-learn & XGBoost) ---
print("\n--- Training and Evaluating Scikit-learn & XGBoost Models ---")

# Helper function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n--- Evaluating {model_name} ---")
    try:
        # Train the model
        model.fit(X_train, y_train)
        print(f"{model_name} trained successfully.")

        # Make predictions
        y_pred = model.predict(X_test)

        # Display evaluation metrics
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Optional: Cross-validation scores
        # cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        # print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    except Exception as e:
        print(f"An error occurred during {model_name} training/evaluation: {e}")

# 1. Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
evaluate_model(log_reg, X_train_processed, y_train, X_test_processed, y_test, "Logistic Regression")

# 2. Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
evaluate_model(rf_clf, X_train_processed, y_train, X_test_processed, y_test, "Random Forest Classifier")

# 3. SVC
svc_clf = SVC(probability=True, random_state=42) # probability=True is useful for voting classifiers if they use predict_proba
evaluate_model(svc_clf, X_train_processed, y_train, X_test_processed, y_test, "SVC")

# 4. Voting Classifier
# Using Logistic Regression, Random Forest, and SVC as base estimators
# Note: SVC's kernel can significantly impact performance. 'rbf' is common.
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
    ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
    ('svc', SVC(probability=True, random_state=42))
], voting='soft') # 'soft' voting uses probability estimates
evaluate_model(voting_clf, X_train_processed, y_train, X_test_processed, y_test, "Voting Classifier (Soft)")

# 5. XGBoost Classifier
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
evaluate_model(xgb_clf, X_train_processed, y_train, X_test_processed, y_test, "XGBoost Classifier")

# --- Deep Learning Models ---
print("\n--- Training and Evaluating Deep Learning Models ---")

# Prepare data for Deep Learning
# Convert processed data to numpy arrays if they aren't already (they should be from preprocessor but good to be sure)
X_train_np = np.array(X_train_processed)
X_test_np = np.array(X_test_processed)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

# Get the number of features after preprocessing
num_features = X_train_np.shape[1]

# --- TensorFlow Model ---
print("\n--- Setting up TensorFlow Model ---")

tf_model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,)), # Input layer
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

tf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

print("TensorFlow Model Architecture:")
tf_model.summary()

print("\n")