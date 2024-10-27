import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess_data import load_and_clean_data

#load dataset
X_train, X_test, y_train, y_test = load_and_clean_data("data/Heart-Attack-Data-Set.csv")

# membuat dan melatih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# simpan model ke dalam file .pkl
joblib.dump(model, 'models/heart-attack-model.pkl')

# evaluasi model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")