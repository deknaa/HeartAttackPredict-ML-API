import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from preprocess_data import load_and_clean_data

# Load data
X_train, X_test, y_train, y_test = load_and_clean_data("data/Heart-Attack-Data-Set.csv")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Save model
joblib.dump(model, 'models/heart-attack-model.pkl')

# Print evaluation
print(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# -----------------------------
# 1. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# -----------------------------
# 2. ROC Curve
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig("outputs/roc_curve.png")
plt.show()

# -----------------------------
# 3. Precision-Recall Curve (Opsional)
# -----------------------------
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color="green")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig("outputs/precision_recall_curve.png")
plt.show()

# -----------------------------
# penjelasan hasil visual
# -----------------------------
#Confusion Matrix: memperlihatkan jumlah prediksi benar dan salah dalam format visual.
#ROC Curve: menunjukkan kemampuan model dalam membedakan kelas 0 dan 1 berdasarkan probabilitas. Semakin mendekati pojok kiri atas, semakin bagus.
#AUC Score: Area di bawah ROC Curve. Di plot, ini tampil sebagai legenda (misalnya: AUC = 0.92).
#Precision-Recall Curve: berguna ketika data tidak seimbang, tapi tetap bagus untuk analisis klasifikasi biner.
