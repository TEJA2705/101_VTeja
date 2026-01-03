PROBLEM STATEMENT: Symptom Checker (Tabular → Preliminary Dx)
Suggest top-3 likely conditions from symptom inputs (not diagnostic).
Sample data: Disease & Symptoms –
https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset
Outcome: Multilabel or top-k ranking model + LLM explanation template.
Stack: Python, scikit-learn, Streamlit, Pandas.

OUR APPROACH:

An ML-based Symptom Checker that ranks diseases based on symptom similarity and displays confidence scores with preventive guidance through a web interface.

📊 Dataset

DiseaseAndSymptoms.csv – Disease–symptom mapping

Disease precaution.csv – Disease-wise precautions

⚙️ Execution Flow

Extract disease labels and symptom lists

Encode symptoms using MultiLabelBinarizer

Train One-Vs-Rest Logistic Regression model

Predict disease probabilities and rank Top-3 conditions

Display results with precautions and disclaimer via Streamlit UI

📈 Evaluation

Top-1 Accuracy

Recall@3 (prioritized to reduce missed conditions)

🛡 Ethics

Not a diagnostic tool

Clear medical disclaimer

Encourages professional consultation

🧩 Tech Stack

Python, scikit-learn, Streamlit, Pandas
