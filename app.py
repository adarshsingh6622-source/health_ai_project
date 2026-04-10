import streamlit as st
import pandas as pd
import joblib
import os
import requests
import numpy as np

from pathlib import Path
from tensorflow.keras.models import load_model

# ================= API =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================= PATH =================
BASE_DIR = Path(__file__).resolve().parent

# ================= LOAD MODEL =================
model = load_model(BASE_DIR / "model/health_model_dl.h5")
vectorizer = joblib.load(BASE_DIR / "model/vectorizer_dl.pkl")
le = joblib.load(BASE_DIR / "model/label_encoder.pkl")

# ================= LOAD DATA =================
df = pd.read_csv(BASE_DIR / "data/Original_Dataset.csv")
df.columns = df.columns.str.strip()

desc_df = pd.read_csv(BASE_DIR / "data/Disease_Description.csv")
desc_df.columns = desc_df.columns.str.strip()

doc_df = pd.read_csv(BASE_DIR / "data/Doctor_Versus_Disease.csv")
doc_df.columns = doc_df.columns.str.strip()

# ================= SYMPTOMS =================
symptom_cols = [col for col in df.columns if col != "Disease"]

all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())

all_symptoms = sorted([str(s).strip() for s in all_symptoms if str(s).strip() != ""])

# ================= FUNCTIONS =================

def get_description(disease):
    disease = disease.strip().lower()
    desc_df["Disease"] = desc_df["Disease"].astype(str).str.strip().str.lower()
    row = desc_df[desc_df["Disease"] == disease]
    if not row.empty:
        return row.iloc[0, 1]
    return "No description available"

def get_doctor(disease):
    disease = disease.strip().lower()
    doc_df.iloc[:, 0] = doc_df.iloc[:, 0].astype(str).str.strip().str.lower()
    row = doc_df[doc_df.iloc[:, 0] == disease]
    if not row.empty:
        return row.iloc[0, 1]
    return "Consult general physician"

def ask_groq(prompt):
    if not GROQ_API_KEY:
        return "API key not set"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        res = requests.post(url, headers=headers, json=data)
        result = res.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI Error: {str(e)}"

# ================= UI =================

st.title("🧠 AI Health Analyzer ")

selected_symptoms = st.multiselect("Select your symptoms", all_symptoms)

if st.button("Predict"):

    if not selected_symptoms:
        st.warning("Please select at least one symptom")

    else:
        # ================= SMART PREDICTION =================

        matches = []

        for i, row in df.iterrows():
            symptoms = row.drop("Disease").values
            symptoms = [str(s).strip() for s in symptoms if str(s) != "nan"]

            score = len(set(selected_symptoms) & set(symptoms))

            if score > 0:
                matches.append((row["Disease"], score))

        if matches:
            matches = sorted(matches, key=lambda x: x[1], reverse=True)
            pred = matches[0][0]
            confidence = matches[0][1] / len(selected_symptoms)
        else:
            # fallback to Deep Learning
            input_text = " ".join(selected_symptoms)
            input_vec = vectorizer.transform([input_text]).toarray()

            probs = model.predict(input_vec)
            pred_index = np.argmax(probs)
            confidence = np.max(probs)

            pred = le.inverse_transform([pred_index])[0]

        # ================= OUTPUT =================

        st.success(f"Predicted Disease: {pred}")
       

        # ================= DESCRIPTION =================
        st.subheader("Description")
        st.write(get_description(pred))

        # ================= DOCTOR =================
        st.subheader("Recommended Doctor")
        st.write(get_doctor(pred))

        # ================= AI EXPLANATION =================
        st.subheader("AI Explanation")

        prompt1 = f"""
        Disease: {pred}
        Explain:
        - What is it
        - Causes
        - Is it serious
        """

        st.write(ask_groq(prompt1))

        # ================= AI ADVICE =================
        st.subheader("AI Advice")

        prompt2 = f"""
        Disease: {pred}
        Give:
        - Home remedies
        - Diet
        - When to see doctor
        """

        st.write(ask_groq(prompt2))
