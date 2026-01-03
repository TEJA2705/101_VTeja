import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("data/DiseaseAndSymptoms.csv")

disease_col = df.columns[0]
symptom_cols = df.columns[1:]

# Convert symptoms to list
df["symptom_list"] = df[symptom_cols].apply(
    lambda row: [str(s).strip().lower() for s in row if pd.notna(s)],
    axis=1
)

# Encode symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptom_list"])

# Reduce dimensionality → disease embeddings
svd = TruncatedSVD(n_components=50, random_state=42)
X_embedded = svd.fit_transform(X)

# Store disease embeddings
disease_embeddings = {
    disease: X_embedded[i]
    for i, disease in enumerate(df[disease_col])
}

# Save everything
joblib.dump(disease_embeddings, "disease_embeddings.pkl")
joblib.dump(mlb, "encoder_embed.pkl")
joblib.dump(svd, "svd_model.pkl")

print("✅ Disease embeddings created and saved")
