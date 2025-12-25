
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# === Load your dataset ===
df = pd.read_csv("datasets/thyroid.csv")  # Replace with your actual CSV file
df.replace('?', pd.NA, inplace=True)

# Drop rows with any missing values in the relevant columns
df.dropna(subset=['T3', 'TT4', 'TSH', 'age'], inplace=True)

# === Keep only required columns ===
features = ['age', 'T3', 'TT4', 'TSH']
target = 'binaryClass'  # This column should be 0 or 1

df = df[features + [target]]
df = df[df['binaryClass'].isin(['N', 'P'])]  # Filter valid labels
df['binaryClass'] = df['binaryClass'].map({'N': 0, 'P': 1})

# === Handle missing values if any ===
df.dropna(inplace=True)

# === Train-Test Split ===
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Feature Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Helper Function to Evaluate Models ===
# === Updated Helper Function to Evaluate Models with Bias-Variance Check ===
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    # Predict and calculate probabilities
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred_test

    # Accuracy, Precision, Recall, F1 Score, ROC AUC for Test Data
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_proba_test)

    # Accuracy for Train Data
    accuracy_train = accuracy_score(y_train, y_pred_train)

    # Bias-Variance Check
    bias_variance_balance = "✅ Good bias-variance balance" if accuracy_train - accuracy_test < 0.1 else "⚠️ Potential overfitting or underfitting detected."

    # Print Evaluation Results
    print(f"\n--- {name} Evaluation ---")
    print(f"Accuracy (Test)       : {accuracy_test:.4f}")
    print(f"F1 Score (Test)       : {f1_test:.4f}")
    print(f"Precision (Test)      : {precision_test:.4f}")
    print(f"Recall (Test)         : {recall_test:.4f}")
    print(f"ROC AUC Score (Test)  : {roc_auc_test:.4f}")

    
    # Bias-Variance Message
    print(f"Bias-Variance Check: {bias_variance_balance}\n")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Positive'], yticklabels=['Normal', 'Positive'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_test:.2f})", color='b')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - ROC Curve')
    plt.legend()
    plt.show()


# === Model 1: Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_scaled, y_train)
evaluate_model("Random Forest", rf, X_train_scaled, y_train, X_test_scaled, y_test)

importances = pd.Series(rf.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', title='Random Forest - Feature Importances')
plt.show()

# === Cross-Validation for Random Forest ===
scores = cross_val_score(rf, scaler.fit_transform(X), y, cv=5, scoring='f1')
#joblib.dump(rf, 'thyroidmodel.pkl')

print("F1 Scores for each fold:", scores)
print("Average F1 Score:", scores.mean())

