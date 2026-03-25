# -*- coding: utf-8 -*-
"""
Train and Save XGBoost Model for Deployment
This script trains the XGBoost model and saves it for use in the API
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import os

print("=" * 80)
print("TRAINING AND SAVING XGBOOST MODEL")
print("=" * 80)

# Configuration
FILE_PATH = r"C:\Users\Hp\.cache\kagglehub\datasets\iammustafatz\diabetes-prediction-dataset\versions\1\diabetes_prediction_dataset.csv"
TARGET_COLUMN = 'diabetes'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = 'models'

# Create models directory
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading data...")
try:
    df = pd.read_csv(FILE_PATH)
    print(f"[OK] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    exit()

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
print("\n[STEP 2] Preprocessing data...")

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET_COLUMN).tolist()

print(f"   Categorical columns: {categorical_cols}")
print(f"   Numerical columns: {numerical_cols}")

# Encode categorical variables
df_processed = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"   [OK] Encoded {col}")

# Separate features and target
X = df_processed.drop(TARGET_COLUMN, axis=1)
y = df_processed[TARGET_COLUMN]

feature_names = X.columns.tolist()
print(f"\n[OK] Features: {feature_names}")
print(f"[OK] Target column: {TARGET_COLUMN}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[OK] Train set: {X_train.shape[0]} samples")
print(f"[OK] Test set: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"[OK] Features scaled using StandardScaler")

# ============================================================================
# 3. TRAIN XGBOOST MODEL
# ============================================================================
print("\n[STEP 3] Training XGBoost model...")

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0
)

xgb_model.fit(X_train_scaled, y_train)
print("[OK] XGBoost model trained successfully")

# ============================================================================
# 4. EVALUATE MODEL
# ============================================================================
print("\n[STEP 4] Evaluating model on test set...")

y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"[METRICS]")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")

# ============================================================================
# 5. SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n[STEP 5] Saving model artifacts...")

# Save the trained model
model_path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
joblib.dump(xgb_model, model_path)
print(f"[OK] Model saved to: {model_path}")

# Save the scaler
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"[OK] Scaler saved to: {scaler_path}")

# Save the label encoders
encoders_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
joblib.dump(label_encoders, encoders_path)
print(f"[OK] Label encoders saved to: {encoders_path}")

# Save feature names
feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
joblib.dump(feature_names, feature_names_path)
print(f"[OK] Feature names saved to: {feature_names_path}")

# Save model metadata
metadata = {
    'model_type': 'XGBoost',
    'n_features': len(feature_names),
    'feature_names': feature_names,
    'categorical_features': categorical_cols,
    'numerical_features': numerical_cols,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'test_size': TEST_SIZE,
    'random_state': RANDOM_STATE
}

metadata_path = os.path.join(MODEL_DIR, 'model_metadata.pkl')
joblib.dump(metadata, metadata_path)
print(f"[OK] Model metadata saved to: {metadata_path}")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE")
print("=" * 80)
print(f"\nModel artifacts saved in: {os.path.abspath(MODEL_DIR)}/")
print("\nFiles created:")
print(f"  - xgboost_model.pkl")
print(f"  - scaler.pkl")
print(f"  - label_encoders.pkl")
print(f"  - feature_names.pkl")
print(f"  - model_metadata.pkl")
print("\nNext: Deploy the API using main.py")
print("=" * 80)
