import gradio as gr
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
    try:
        disease = disease.strip().lower()
        desc_df["Disease"] = desc_df["Disease"].astype(str).str.strip().str.lower()
        row = desc_df[desc_df["Disease"] == disease]
        return row.iloc[0, 1] if not row.empty else "No description available"
    except:
        return "No description available"

def get_doctor(disease):
    try:
        disease = disease.strip().lower()
        doc_df.iloc[:, 0] = doc_df.iloc[:, 0].astype(str).str.strip().str.lower()
        row = doc_df[doc_df.iloc[:, 0] == disease]
        return row.iloc[0, 1] if not row.empty else "Consult general physician"
    except:
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
        res = requests.post(url, headers=headers, json=data, timeout=10)
        result = res.json()
        return result["choices"][0]["message"]["content"]
    except:
        return "AI service temporarily unavailable"

# ================= MAIN FUNCTION =================

def predict(symptoms):
    try:
        if not symptoms:
            return "Select symptoms", "", "", "", ""

        # ===== SMART MATCHING =====
        matches = []

        for i, row in df.iterrows():
            row_symptoms = row.drop("Disease").values
            row_symptoms = [str(s).strip() for s in row_symptoms if str(s) != "nan"]

            score = len(set(symptoms) & set(row_symptoms))

            if score > 0:
                matches.append((row["Disease"], score))

        if matches:
            matches = sorted(matches, key=lambda x: x[1], reverse=True)
            pred = matches[0][0]
        else:
            # ===== DL fallback =====
            input_text = " ".join(symptoms)
            input_vec = vectorizer.transform([input_text]).toarray()

            probs = model.predict(input_vec)
            pred_index = np.argmax(probs)
            pred = le.inverse_transform([pred_index])[0]

        # ===== EXTRA INFO =====
        description = get_description(pred)
        doctor = get_doctor(pred)

        # ===== AI =====
        explanation = ask_groq(f"Disease: {pred}. Explain simply.")
        advice = ask_groq(f"Disease: {pred}. Give home remedies, diet and when to see doctor.")

        return pred, description, doctor, explanation, advice

    except Exception as e:
        return "Error", str(e), "", "", ""

# ================= UI =================

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Health Analyzer ")

    symptoms_input = gr.Dropdown(all_symptoms, multiselect=True, label="Select Symptoms")

    btn = gr.Button("Predict")

    disease = gr.Textbox(label="Predicted Disease")
    description = gr.Textbox(label="Description")
    doctor = gr.Textbox(label="Recommended Doctor")
    explanation = gr.Textbox(label="AI Explanation")
    advice = gr.Textbox(label="AI Advice")

    btn.click(
        predict,
        inputs=symptoms_input,
        outputs=[disease, description, doctor, explanation, advice]
    )

demo.launch()