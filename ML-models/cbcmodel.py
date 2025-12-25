import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    multilabel_confusion_matrix,accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, hamming_loss, precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("datasets\cbc.csv")
df.replace('?', pd.NA, inplace=True)
df.dropna(subset=['WBC', 'RBC', 'HGB', 'MCV', 'MCH', 'MCHC', 'PLT'], inplace=True)

df['WBC'] = df['WBC'] * 1000
df['PLT'] = df['PLT'] * 1000

columns_needed = ['WBC', 'RBC', 'HGB', 'MCV', 'MCH', 'MCHC', 'PLT']
df = df[columns_needed].copy()

z_scores = np.abs((df - df.mean()) / df.std())
df = df[(z_scores < 3).all(axis=1)]

def generate_labels(row):
    labels = {
        "Bacterial_Infection": int(row['WBC'] > 11000),
        "Iron_Def_Anemia": int(row['HGB'] < 12 and row['MCV'] < 80 and row['MCH'] < 27 and row['MCHC'] < 32),
        "Thrombocytosis": int(row['PLT'] > 450000),
    }
    labels["Normal"] = int(sum(labels.values()) == 0)
    return pd.Series(labels)

labels_df = df.apply(generate_labels, axis=1)
X = df
Y = labels_df
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
def evaluate_model(name, model, X_test, y_test, X_all, y_all, cv):
    y_pred = model.predict(X_test)
    y_pred = rf_model.predict(X_test)
    subset_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {subset_accuracy:.4f}")

    
    metrics = {
    "Hamming Loss": hamming_loss(y_test, y_pred),
    "Micro Precision": precision_score(y_test, y_pred, average='micro', zero_division=0),
    "Micro Recall": recall_score(y_test, y_pred, average='micro', zero_division=0),
    "Micro F1 Score": f1_score(y_test, y_pred, average='micro', zero_division=0),
    "Macro Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
    "Macro Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
    "Macro F1 Score": f1_score(y_test, y_pred, average='macro', zero_division=0),
}

    print(f"\n--- {name} Multi-Label Evaluation Summary ---")
    print(pd.DataFrame.from_dict(metrics, orient='index', columns=["Score"]).round(4))

    cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='f1_micro')
    print(f"\n{name} Cross-Validation Micro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\nDetailed Classification Report:\n")
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    print(report_df)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    report_df.iloc[:-3][["precision", "recall", "f1-score"]].plot(kind='bar', ax=ax)
    ax.set_title("Precision, Recall, F1 per Label")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    mcm = multilabel_confusion_matrix(y_test, y_pred)
    labels = y_test.columns

    for idx, cm in enumerate(mcm):
        plt.figure(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=["Not", "Yes"], yticklabels=["Not", "Yes"])
        plt.title(f"Confusion Matrix: {labels[idx]}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()



cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest
rf_model = MultiOutputClassifier(RandomForestClassifier(random_state=42, class_weight='balanced'))
rf_model.fit(X_train, y_train)
evaluate_model("Random Forest", rf_model, X_test, y_test, X, Y, cv)

# joblib.dump(rf_model,'cbcmodel.pkl')

