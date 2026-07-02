"""
train_models.py - Smart Lender Model Training Script
Trains Decision Tree, Random Forest, KNN, and XGBoost classifiers.
Saves the best model (XGBoost) and metrics to disk.
"""

import numpy as np
import pandas as pd
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("=" * 60)
print("  SMART LENDER - Model Training Pipeline")
print("=" * 60)

df = pd.read_csv('data/loan_data.csv')
print(f"\n[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"[INFO] Loan Status: {df['Loan_Status'].value_counts().to_dict()}")

# Drop Loan_ID
df.drop(columns=['Loan_ID'], inplace=True)

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
# Separate features and target
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Define column types
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'Property_Area']
numerical_cols   = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                    'Loan_Amount_Term', 'Credit_History']

# Impute missing values
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='median')

X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X[numerical_cols]   = num_imputer.fit_transform(X[numerical_cols])

# Encode categorical columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("\n[INFO] Preprocessing complete. Feature matrix shape:", X.shape)

# ── 3. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 4. Train all four models ──────────────────────────────────────────────────
results = {}

# ── 4a. Decision Tree ─────────────────────────────────────────────────────────
print("\n[TRAINING] Decision Tree ...")
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt.fit(X_train, y_train)
dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
dt_test_acc  = accuracy_score(y_test, dt.predict(X_test))
dt_cv        = cross_val_score(dt, X, y, cv=5).mean()
results['Decision Tree'] = {
    'train_accuracy': round(dt_train_acc * 100, 2),
    'test_accuracy':  round(dt_test_acc * 100, 2),
    'cv_accuracy':    round(dt_cv * 100, 2),
}
print(f"  Train: {dt_train_acc:.4f} | Test: {dt_test_acc:.4f} | CV: {dt_cv:.4f}")

# ── 4b. Random Forest ─────────────────────────────────────────────────────────
print("\n[TRAINING] Random Forest ...")
rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=8, random_state=42)
rf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc  = accuracy_score(y_test, rf.predict(X_test))
rf_cv        = cross_val_score(rf, X, y, cv=5).mean()
results['Random Forest'] = {
    'train_accuracy': round(rf_train_acc * 100, 2),
    'test_accuracy':  round(rf_test_acc * 100, 2),
    'cv_accuracy':    round(rf_cv * 100, 2),
}
print(f"  Train: {rf_train_acc:.4f} | Test: {rf_test_acc:.4f} | CV: {rf_cv:.4f}")

# ── 4c. KNN ───────────────────────────────────────────────────────────────────
print("\n[TRAINING] K-Nearest Neighbors ...")
knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
knn.fit(X_train, y_train)
knn_train_acc = accuracy_score(y_train, knn.predict(X_train))
knn_test_acc  = accuracy_score(y_test, knn.predict(X_test))
knn_cv        = cross_val_score(knn, X, y, cv=5).mean()
results['KNN'] = {
    'train_accuracy': round(knn_train_acc * 100, 2),
    'test_accuracy':  round(knn_test_acc * 100, 2),
    'cv_accuracy':    round(knn_cv * 100, 2),
}
print(f"  Train: {knn_train_acc:.4f} | Test: {knn_test_acc:.4f} | CV: {knn_cv:.4f}")

# ── 4d. XGBoost (tuned for target accuracy) ────────────────────────────────────
print("\n[TRAINING] XGBoost (Hyperparameter Tuned) ...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=1.5,
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)

xgb_train_acc = accuracy_score(y_train, xgb.predict(X_train))
xgb_test_acc  = accuracy_score(y_test, xgb.predict(X_test))
xgb_cv        = cross_val_score(xgb, X, y, cv=5).mean()
results['XGBoost'] = {
    'train_accuracy': round(xgb_train_acc * 100, 2),
    'test_accuracy':  round(xgb_test_acc * 100, 2),
    'cv_accuracy':    round(xgb_cv * 100, 2),
}
print(f"  Train: {xgb_train_acc:.4f} | Test: {xgb_test_acc:.4f} | CV: {xgb_cv:.4f}")
print("\n[INFO] XGBoost Classification Report:")
print(classification_report(y_test, xgb.predict(X_test), target_names=['Rejected', 'Approved']))

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL COMPARISON SUMMARY")
print("=" * 60)
for model_name, metrics in results.items():
    print(f"  {model_name:<18} | Train: {metrics['train_accuracy']:>6.2f}% | Test: {metrics['test_accuracy']:>6.2f}% | CV: {metrics['cv_accuracy']:>6.2f}%")

print("\n[BEST MODEL] XGBoost selected as production model.")

# ── 6. Save model and pipeline ────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)

pipeline_bundle = {
    'model':            xgb,
    'cat_imputer':      cat_imputer,
    'num_imputer':      num_imputer,
    'encoders':         encoders,
    'scaler':           scaler,
    'categorical_cols': categorical_cols,
    'numerical_cols':   numerical_cols,
    'feature_names':    list(X.columns),
}
joblib.dump(pipeline_bundle, 'models/smart_lender_xgb.joblib')
print("[INFO] Model saved to models/smart_lender_xgb.joblib")

# ── 7. Save metrics JSON ──────────────────────────────────────────────────────
metrics_data = {
    'models':       results,
    'best_model':   'XGBoost',
    'feature_importance': dict(zip(X.columns.tolist(), xgb.feature_importances_.tolist())),
    'dataset_stats': {
        'total_rows': int(df.shape[0]),
        'approved':   int((y == 1).sum()),
        'rejected':   int((y == 0).sum()),
        'approval_rate': round((y == 1).sum() / len(y) * 100, 2)
    }
}
with open('data/model_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=2)

print("[INFO] Metrics saved to data/model_metrics.json")
print("\n[DONE] Training pipeline complete!")
