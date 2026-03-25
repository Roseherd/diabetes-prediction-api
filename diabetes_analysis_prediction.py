# -*- coding: utf-8 -*-
"""
Diabetes Prediction Analysis - Comprehensive ML & DL Pipeline
Created on Sat Mar 21 2026

This script performs:
1. Exploratory Data Analysis (EDA)
2. 4 Machine Learning Models:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - XGBoost
3. 2 Deep Learning Models:
   - TensorFlow (Keras Sequential)
   - PyTorch Neural Network

@author: Hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, 
    roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
)
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
FILE_PATH = r"C:\Users\Hp\.cache\kagglehub\datasets\iammustafatz\diabetes-prediction-dataset\versions\1\diabetes_prediction_dataset.csv"
TARGET_COLUMN = 'diabetes'
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS_TF = 100
EPOCHS_PYTORCH = 100
BATCH_SIZE = 32

# ============================================================================
# 1. DATA LOADING AND INITIAL INSPECTION
# ============================================================================
print("=" * 80)
print("DIABETES PREDICTION ANALYSIS - COMPREHENSIVE ML & DL PIPELINE")
print("=" * 80)
print("\n--- LOADING DATA ---")

try:
    df = pd.read_csv(FILE_PATH)
    print(f"[OK] Data loaded successfully from: {FILE_PATH}")
    print(f"[OK] Dataset shape: {df.shape}")
    print(f"\n--- First 5 Rows ---")
    print(df.head())
    print(f"\n--- Data Types ---")
    print(df.dtypes)
except FileNotFoundError:
    print(f"[ERROR] File not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"[ERROR] Error loading file: {e}")
    exit()

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])
else:
    print("[OK] No missing values found")

print("\n--- Target Variable Distribution ---")
print(df[TARGET_COLUMN].value_counts())
print(f"\nTarget distribution (%):\n{df[TARGET_COLUMN].value_counts(normalize=True) * 100}")

# ============================================================================
# 3. DATA VISUALIZATION
# ============================================================================
print("\n--- Generating Visualizations ---")

# Create output directory structure for visualizations
import os
os.makedirs('visualizations', exist_ok=True)

# 3.1 Target Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=TARGET_COLUMN, data=df, palette='viridis')
plt.title('Distribution of Diabetes Target Variable', fontsize=14, fontweight='bold')
plt.xlabel('Diabetes (0: No, 1: Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visualizations/01_target_distribution.png', dpi=300)
plt.close()
print("[OK] Saved: visualizations/01_target_distribution.png")

# 3.2 Numerical Features Distribution
numerical_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET_COLUMN)
if len(numerical_cols) > 0:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        if idx < len(axes):
            axes[idx].hist(df[col], bins=30, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
    
    # Hide unused subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/02_numerical_features_distribution.png', dpi=300)
    plt.close()
    print("[OK] Saved: visualizations/02_numerical_features_distribution.png")

# 3.3 Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of All Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/03_correlation_matrix.png', dpi=300)
plt.close()
print("[OK] Saved: visualizations/03_correlation_matrix.png")

# 3.4 Target Correlation
plt.figure(figsize=(10, 6))
target_corr = correlation_matrix[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
sns.barplot(x=target_corr.values, y=target_corr.index, palette='RdYlGn')
plt.title(f'Feature Correlation with Target ({TARGET_COLUMN})', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('visualizations/04_target_correlation.png', dpi=300)
plt.close()
print("[OK] Saved: visualizations/04_target_correlation.png")

# 3.5 Categorical Features vs Target
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if len(categorical_cols) > 0:
    fig, axes = plt.subplots(1, len(categorical_cols), figsize=(15, 4))
    if len(categorical_cols) == 1:
        axes = [axes]
    
    # Create a copy with target as string for visualization
    df_for_viz = df.copy()
    df_for_viz[TARGET_COLUMN] = df_for_viz[TARGET_COLUMN].astype(str)
    
    for idx, col in enumerate(categorical_cols):
        sns.countplot(x=col, hue=TARGET_COLUMN, data=df_for_viz, ax=axes[idx], palette='husl')
        axes[idx].set_title(f'{col} vs {TARGET_COLUMN}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('visualizations/05_categorical_vs_target.png', dpi=300)
    plt.close()
    print("[OK] Saved: visualizations/05_categorical_vs_target.png")

# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET_COLUMN).tolist()

print(f"\n--- Categorical Columns: {categorical_cols}")
print(f"--- Numerical Columns: {numerical_cols}")

# Create a copy for preprocessing
df_processed = df.copy()

# Encode categorical variables using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"[OK] Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Separate features and target
X = df_processed.drop(TARGET_COLUMN, axis=1)
y = df_processed[TARGET_COLUMN]

feature_names = X.columns.tolist()
print(f"\n--- Feature Names: {feature_names}")
print(f"--- Total Features: {len(feature_names)}")

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n[OK] Data split (80-20):")
print(f"   - Training set: {X_train.shape[0]} samples")
print(f"   - Testing set: {X_test.shape[0]} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"[OK] Features scaled using StandardScaler")

# Convert to numpy arrays for DL models
X_train_np = np.array(X_train_scaled)
X_test_np = np.array(X_test_scaled)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

n_features = X_train_scaled.shape[1]
print(f"[OK] Number of features for DL models: {n_features}")

# ============================================================================
# 5. HELPER FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    filename = f"visualizations/cm_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    return cm

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_names=None):
    """Train and evaluate a model"""
    print(f"\n--- {model_name} ---")
    try:
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"[OK] Accuracy:  {accuracy:.4f}")
        print(f"[OK] Precision: {precision:.4f}")
        print(f"[OK] Recall:    {recall:.4f}")
        print(f"[OK] F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"[OK] ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, model_name)
        
        # Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:10]  # Top 10
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            plt.bar(range(len(indices)), importance[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.ylabel('Importance')
            plt.tight_layout()
            filename = f"visualizations/fi_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"[OK] Feature importance plot saved")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'model': model
        }
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return None

# ============================================================================
# 6. MACHINE LEARNING MODELS (SCIKIT-LEARN & XGBOOST)
# ============================================================================
print("\n" + "=" * 80)
print("MACHINE LEARNING MODELS")
print("=" * 80)

ml_results = {}

# Model 1: Logistic Regression
lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
ml_results['Logistic Regression'] = evaluate_model(
    lr_model, X_train_scaled, y_train, X_test_scaled, y_test, 
    'Logistic Regression', feature_names
)

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
ml_results['Random Forest'] = evaluate_model(
    rf_model, X_train_scaled, y_train, X_test_scaled, y_test, 
    'Random Forest', feature_names
)

# Model 3: Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
ml_results['SVM'] = evaluate_model(
    svm_model, X_train_scaled, y_train, X_test_scaled, y_test, 
    'Support Vector Machine', feature_names
)

# Model 4: XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    use_label_encoder=False
)
ml_results['XGBoost'] = evaluate_model(
    xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, 
    'XGBoost', feature_names
)

# ============================================================================
# ============================================================================
# 7. DEEP LEARNING - PYTORCH (TensorFlow skipped due to environment issues)
# ============================================================================
print("\n" + "=" * 80)
print("DEEP LEARNING - PYTORCH")
print("=" * 80)
print("\nNote: TensorFlow has environment compatibility issues and has been skipped.")
print("Using PyTorch for deep learning instead.\n")

# ============================================================================
# 8. DEEP LEARNING - PYTORCH (CONTINUED)
# ============================================================================
print("=" * 80)

# Define PyTorch Neural Network
class DiabetesNN(nn.Module):
    def __init__(self, input_size):
        super(DiabetesNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

# Custom Dataset
class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

print("\n--- Building PyTorch Neural Network ---")

# Initialize model, optimizer, and loss function
pytorch_model = DiabetesNN(n_features)
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

print("PyTorch Model Architecture:")
print(pytorch_model)

# Create data loaders
train_dataset = DiabetesDataset(X_train_np, y_train_np)
test_dataset = DiabetesDataset(X_test_np, y_test_np)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\n--- Training PyTorch Model ---")

train_losses = []
val_losses = []

for epoch in range(EPOCHS_PYTORCH):
    # Training
    pytorch_model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = pytorch_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    pytorch_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = pytorch_model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS_PYTORCH} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("[OK] PyTorch training complete")

# Evaluate PyTorch Model
print("\n--- Evaluating PyTorch Model ---")

pytorch_model.eval()
with torch.no_grad():
    y_pred_pytorch_proba = pytorch_model(torch.FloatTensor(X_test_np)).numpy()
    y_pred_pytorch = (y_pred_pytorch_proba > 0.5).astype(int).flatten()

accuracy_pytorch = accuracy_score(y_test_np, y_pred_pytorch)
precision_pytorch = precision_score(y_test_np, y_pred_pytorch)
recall_pytorch = recall_score(y_test_np, y_pred_pytorch)
f1_pytorch = f1_score(y_test_np, y_pred_pytorch)
roc_auc_pytorch = roc_auc_score(y_test_np, y_pred_pytorch_proba)

print(f"[OK] Accuracy:  {accuracy_pytorch:.4f}")
print(f"[OK] Precision: {precision_pytorch:.4f}")
print(f"[OK] Recall:    {recall_pytorch:.4f}")
print(f"[OK] F1-Score:  {f1_pytorch:.4f}")
print(f"[OK] ROC-AUC:   {roc_auc_pytorch:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test_np, y_pred_pytorch))

plot_confusion_matrix(y_test_np, y_pred_pytorch, 'PyTorch Model')

# Training History Plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('PyTorch Model - Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/pytorch_training_history.png', dpi=300)
plt.close()
print("[OK] Training history plot saved")

dl_results_pytorch = {
    'accuracy': accuracy_pytorch,
    'precision': precision_pytorch,
    'recall': recall_pytorch,
    'f1': f1_pytorch,
    'roc_auc': roc_auc_pytorch,
    'model': pytorch_model
}

# ============================================================================
# 9. MODEL COMPARISON AND SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

# Compile results
all_results = {
    'Logistic Regression': ml_results['Logistic Regression'],
    'Random Forest': ml_results['Random Forest'],
    'SVM': ml_results['SVM'],
    'XGBoost': ml_results['XGBoost'],
    'PyTorch': dl_results_pytorch
}

# Create comparison dataframe
comparison_data = []
for model_name, results in all_results.items():
    if results:
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1'],
            'ROC-AUC': results.get('roc_auc', 'N/A')
        })

comparison_df = pd.DataFrame(comparison_data)
print("\n")
print(comparison_df.to_string(index=False))

# Save comparison to CSV
comparison_df.to_csv('model_comparison.csv', index=False)
print("\n[OK] Model comparison saved to: model_comparison.csv")

# Visualization: Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    models = comparison_df['Model'].tolist()
    values = comparison_df[metric].tolist()
    
    bars = ax.bar(models, values, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300)
plt.close()
print("[OK] Model comparison plot saved: visualizations/model_comparison.png")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"\n[OK] All visualizations saved in: visualizations/")
print(f"[OK] Model comparison saved to: model_comparison.csv")
print(f"\nBest Model: {comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']} "
      f"(Accuracy: {comparison_df['Accuracy'].max():.4f})")

print("\n--- Visualization Files Generated ---")
import glob
viz_files = sorted(glob.glob('visualizations/*.png'))
for i, file in enumerate(viz_files, 1):
    print(f"  {i}. {file}")

print("\n" + "=" * 80)
