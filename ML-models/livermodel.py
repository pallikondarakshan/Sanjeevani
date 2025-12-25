
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
warnings.filterwarnings(action='ignore', category=UserWarning, module='lightgbm')

# === Load & Clean Data ===
df = pd.read_csv("datasets/indianliver.csv", encoding="unicode_escape")
df.columns = df.columns.str.strip()

df.rename(columns={
    "Alkaline_Phosphotase": "ALP",
    "Alamine_Aminotransferase": "ALT",
    "Aspartate_Aminotransferase": "AST",
    "Total_Protiens": "Total_Proteins",
    "Albumin_and_Globulin_Ratio": "A_G_Ratio"
}, inplace=True)

# === Preprocessing ===
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Label"] = df["Dataset"].apply(lambda x: 1 if x == 1 else 0)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# === Remove Outliers ===
z_features = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "ALP", "ALT", "AST", "Total_Proteins", "Albumin", "A_G_Ratio"
]
z_scores = stats.zscore(df[z_features])
df = df[(np.abs(z_scores) < 3).all(axis=1)]

# === Features and Target ===
features = z_features
X = df[features]
y = df["Label"]

# === Apply SMOTE ===
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)


def evaluate_model(name, model, X_test, y_test, X_train=None, y_train=None, X_all=None, y_all=None, cv=5):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    print(f"\n--- {name} Evaluation ---")
    print("Accuracy       :", accuracy_score(y_test, y_pred))
    print("F1 Score       :", f1_score(y_test, y_pred))
    print("Precision      :", precision_score(y_test, y_pred))
    print("Recall         :", recall_score(y_test, y_pred))
    if hasattr(model, "predict_proba"):
        print("ROC AUC Score  :", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} - ROC Curve')
        plt.legend()
        plt.show()

    # Bias-Variance Check (optional if training data given)
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        print("\n--- Bias-Variance Check ---")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy : {test_acc:.4f}")
        if train_acc < 0.7 and test_acc < 0.7:
            print("⚠️ Likely underfitting (high bias)")
        elif train_acc > 0.9 and test_acc < 0.7:
            print("⚠️ Likely overfitting (high variance)")
        else:
            print("✅ Good bias-variance balance")

    # Cross-validation (optional if full dataset given)
    if X_all is not None and y_all is not None:
        cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='f1')
        print(f"\nCross-Validation F1 Scores: {cv_scores}")
        print(f"Average CV F1 Score       : {cv_scores.mean():.4f}")

# === Randomized Search for Hyperparameter Tuning ===
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define a reduced or randomized parameter space
param_dist = {
    'n_estimators': randint(100, 300),
    'learning_rate': uniform(0.01, 0.1),
    'num_leaves': randint(20, 50),
    'max_depth': randint(3, 7),
    'min_child_samples': randint(10, 40),
    'subsample': uniform(0.8, 0.2),
    'colsample_bytree': uniform(0.8, 0.2)
}

# Initialize the model
lgbm_model = LGBMClassifier(random_state=42, verbose=-1)

# Randomized Search (reduce n_iter to control speed)
random_search = RandomizedSearchCV(
    estimator=lgbm_model,
    param_distributions=param_dist,
    n_iter=50,              # Try 50 combinations instead of 648
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Extract best model and scores
best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters:", best_params)
print("Best F1 Score:", best_score)

# Evaluate and save
best_lgbm_model = random_search.best_estimator_
evaluate_model("Tuned LightGBM", best_lgbm_model, X_test, y_test, X_train, y_train)


# Cross-validation F1 score
scores_tuned_lgbm = cross_val_score(best_lgbm_model, X_resampled, y_resampled, cv=5, scoring='f1')
print("Tuned LightGBM CV F1 Scores:", scores_tuned_lgbm)
print("Average Tuned LightGBM CV F1 Score:", scores_tuned_lgbm.mean())

# === Save the Best Model ===
#joblib.dump(best_lgbm_model, 'livermodel.pkl')
print("\n✅ Best tuned LightGBM model saved as best_liver_model.pkl")
