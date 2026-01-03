import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Load dataset
df = pd.read_csv("data/DiseaseAndSymptoms.csv")

# First column is Disease
disease_col = df.columns[0]
symptom_cols = df.columns[1:]

# Convert symptoms into list (IMPORTANT FIX)
df["symptom_list"] = df[symptom_cols].apply(
    lambda row: [str(symptom).strip().lower()
                 for symptom in row
                 if pd.notna(symptom)],
    axis=1
)

X_symptoms = df["symptom_list"]
y = df[disease_col]

# Encode symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(X_symptoms)

print("Number of symptoms:", X.shape[1])  # sanity check

# Train model
model = OneVsRestClassifier(
    LogisticRegression(max_iter=3000)
)
model.fit(X, y)

# Save model & encoder
joblib.dump(model, "model.pkl")
joblib.dump(mlb, "encoder.pkl")

print("✅ Model trained successfully")
