import pandas as pd
import joblib
import numpy as np

# Load model & encoder
model = joblib.load("model.pkl")
mlb = joblib.load("encoder.pkl")

# Load dataset
df = pd.read_csv("data/DiseaseAndSymptoms.csv")

disease_col = df.columns[0]
symptom_cols = df.columns[1:]

# Convert symptoms into list
df["symptom_list"] = df[symptom_cols].apply(
    lambda row: [str(symptom).strip().lower()
                 for symptom in row if pd.notna(symptom)],
    axis=1
)

X = mlb.transform(df["symptom_list"])
y_true = df[disease_col].values

top1_correct = 0
top3_correct = 0

for i in range(len(X)):
    probs = model.predict_proba(X[i].reshape(1, -1))[0]
    top_indices = np.argsort(probs)[-3:][::-1]
    top_diseases = model.classes_[top_indices]

    if y_true[i] == top_diseases[0]:
        top1_correct += 1

    if y_true[i] in top_diseases:
        top3_correct += 1

accuracy = top1_correct / len(X)
recall_at_3 = top3_correct / len(X)

print("📊 MODEL PERFORMANCE")
print(f"Top-1 Accuracy  : {accuracy:.2%}")
print(f"Recall@3        : {recall_at_3:.2%}")
