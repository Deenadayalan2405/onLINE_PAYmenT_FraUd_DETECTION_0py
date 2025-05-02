import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# Load dataset
data = pd.read_csv(r"C:\Users\DEENADAYALAN\OneDrive\Desktop\Online_Payment Fraud_detection\Data_set\creditcard.csv")

# Feature & label split
X = data.drop('Class', axis=1)
y = data['Class']

# Normalize 'Time' and 'Amount'
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- PLOT 1: Confusion Matrix ---
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig(r"C:\Users\DEENADAYALAN\OneDrive\Desktop\Online_Payment Fraud_detection\ouput\confusion_matrix.png")
plt.close()

# --- PLOT 2: Classification Report Metrics (bar graph) ---
metrics_df = pd.DataFrame(report).transpose()
metrics_df = metrics_df.drop(index='accuracy')

metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6))
plt.title("Precision, Recall, and F1-Score")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:\Users\DEENADAYALAN\OneDrive\Desktop\Online_Payment Fraud_detection\ouput\metrics_bar_chart.png")
plt.close()

# --- PLOT 3: Feature Importance ---
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[-10:]  # Top 10 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(r"C:\Users\DEENADAYALAN\OneDrive\Desktop\Online_Payment Fraud_detection\ouput\feature_importance.png")
plt.close()
