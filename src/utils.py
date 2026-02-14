import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

def preprocess_data(X_train, X_val, X_test, mode='zscore'):
    if mode == 'zscore':
        scaler = StandardScaler()
    elif mode == 'minmax':
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled
    
def evaluate_model(y_true, y_pred):
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm

def print_eval(version, metrics, cm):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred-Normal", "Pred-Trojan"],
                yticklabels=["True-Normal", "True-Trojan"])

    plt.title(
        f"{version}\n"
        f"Acc: {metrics['accuracy']:.4f} | "
        f"Pre: {metrics['precision']:.4f} | "
        f"Rec: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f}",
        fontsize=10,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()