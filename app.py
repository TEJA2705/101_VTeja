import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# Load models
# ===============================
model = joblib.load("model.pkl")
mlb = joblib.load("encoder.pkl")

disease_embeddings = joblib.load("disease_embeddings.pkl")
mlb_embed = joblib.load("encoder_embed.pkl")
svd = joblib.load("svd_model.pkl")

precaution_df = pd.read_csv("data/Disease precaution.csv")

# ===============================
# 🚨 Red-Flag Rules
# ===============================
RED_FLAG_RULES = {
    "Possible Cardiac Emergency": {
        "symptoms": {"chest_pain", "breathlessness", "sweating"},
        "advice": "Seek emergency medical care immediately."
    },
    "Possible Neurological Emergency": {
        "symptoms": {"loss_of_consciousness", "seizures", "confusion"},
        "advice": "Urgent neurological evaluation is recommended."
    }
}

# ===============================
# UI + CSS
# ===============================
st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="centered")

st.markdown("""
<style>
.card {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    background-color: #f9f9f9;
    border-left: 6px solid #4CAF50;
}
.high { border-left: 6px solid #e53935; }
.medium { border-left: 6px solid #fb8c00; }
.low { border-left: 6px solid #43a047; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>🩺 AI Symptom Checker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Smart & safe preliminary health insights</p>", unsafe_allow_html=True)



# ===============================
# Tabs
# ===============================
tab1, tab2, tab3 = st.tabs(["🧾 Symptoms", "📊 ML Results", "🧠 Embedding Results"])

# ===============================
# TAB 1 – Symptoms
# ===============================
with tab1:
    st.subheader("Select symptoms you are experiencing")
    selected_symptoms = st.multiselect(
        "You can select multiple symptoms",
        sorted(mlb.classes_)
    )

    analyze = st.button("🔍 Analyze Symptoms", use_container_width=True)

# ===============================
# Prediction Logic
# ===============================
if analyze:
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
        st.stop()

    selected_set = set(selected_symptoms)

    # 🚨 Red-flag check
    for rule, info in RED_FLAG_RULES.items():
        if info["symptoms"].issubset(selected_set):
            st.error(f"🚨 **{rule}**\n\n{info['advice']}")
            st.stop()

    # ML prediction
    X_input = mlb.transform([selected_symptoms])
    probs = model.predict_proba(X_input)[0]
    top_indices = np.argsort(probs)[-3:][::-1]

    # Embedding prediction
    X_embed = svd.transform(mlb_embed.transform([selected_symptoms]))
    similarity_scores = []

    for disease, emb in disease_embeddings.items():
        score = cosine_similarity(X_embed, emb.reshape(1, -1))[0][0]
        similarity_scores.append((disease, score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # ===============================
    # TAB 2 – ML Results
    # ===============================
    with tab2:
        st.subheader("Top 3 Conditions (ML Model)")
        for idx in top_indices:
            disease = model.classes_[idx]
            confidence = probs[idx]

            level = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"

            st.markdown(f"""
            <div class="card {level}">
                <h3>🦠 {disease}</h3>
                <p><b>Confidence:</b> {confidence*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            prec = precaution_df[precaution_df["Disease"] == disease]
            if not prec.empty:
                with st.expander("🛡 Precautions"):
                    for col in prec.columns[1:]:
                        if pd.notna(prec[col].values[0]):
                            st.write(f"• {prec[col].values[0]}")

    # ===============================
    # TAB 3 – Embedding Results
    # ===============================
    with tab3:
        st.subheader("Top Conditions (Similarity-Based)")
        for disease, score in similarity_scores[:3]:
            st.markdown(f"""
            <div class="card">
                <h3>🧬 {disease}</h3>
                <p><b>Similarity Score:</b> {score:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.success("✅ Analysis complete. Consult a medical professional if symptoms persist.")
