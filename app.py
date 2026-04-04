import streamlit as st
import pandas as pd
import joblib
import os
import requests
from pathlib import Path

# ------------------ API KEY ------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------ PATH ------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------ LOAD MODEL ------------------
model = joblib.load(BASE_DIR / "model/health_model.pkl")
vectorizer = joblib.load(BASE_DIR / "model/vectorizer.pkl")

# ------------------ LOAD DATA ------------------
df = pd.read_csv(BASE_DIR / "data/Original_Dataset.csv")
df.columns = df.columns.str.strip()

desc_df = pd.read_csv(BASE_DIR / "data/Disease_Description.csv")
desc_df.columns = desc_df.columns.str.strip()

doc_df = pd.read_csv(BASE_DIR / "data/Doctor_Versus_Disease.csv")
doc_df.columns = doc_df.columns.str.strip()

# ------------------ EXTRACT SYMPTOMS ------------------
symptom_cols = [col for col in df.columns if col != "Disease"]

all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())

all_symptoms = sorted([str(s).strip() for s in all_symptoms if str(s).strip() != ""])

# ------------------ FUNCTIONS ------------------

def get_description(disease):
    disease = disease.strip()
    row = desc_df[desc_df.iloc[:,0].str.strip() == disease]
    if not row.empty:
        return row.iloc[0][1]
    return "No description available"

def get_doctor(disease):
    disease = disease.strip()
    row = doc_df[doc_df.iloc[:,0].str.strip() == disease]
    if not row.empty:
        return row.iloc[0][1]
    return "Consult general physician"

def ask_groq(prompt):
    if not GROQ_API_KEY:
        return " API key not set"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        res = requests.post(url, headers=headers, json=data)
        result = res.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return str(result)

    except Exception as e:
        return str(e)

# ------------------ UI ------------------

st.title(" AI Health Analyzer")

selected_symptoms = st.multiselect(
    "Select your symptoms",
    all_symptoms
)
# -------SHOW SELECTED ---------
if selected_symptoms:
    st.markdown("### Selected Symptoms:")
    st.success(" | ".join(selected_symptoms))

if st.button("Predict"):

    if not selected_symptoms:
        st.warning("Please select at least one symptom")

    else:
        input_text = " ".join(selected_symptoms)
        input_vec = vectorizer.transform([input_text])

        pred = model.predict(input_vec)[0]

        # -------- RESULT --------
        st.success(f"Predicted Disease: {pred}")

        # -------- DESCRIPTION --------
        st.subheader(" Description")
        st.write(get_description(pred))

        # -------- DOCTOR --------
        st.subheader(" Recommended Doctor")
        st.write(get_doctor(pred))

        # -------- AI EXPLANATION --------
        prompt1 = f"""
        Disease: {pred}

        Explain:
        - What is it
        - Causes
        - Is it serious
        """

        st.subheader(" AI Explanation")
        st.write(ask_groq(prompt1))

        # -------- AI ADVICE --------
        prompt2 = f"""
        Disease: {pred}

        Give advice:
        - Home remedies
        - Diet
        - When to see doctor
        """

        st.subheader(" AI Advice")
        st.write(ask_groq(prompt2))
