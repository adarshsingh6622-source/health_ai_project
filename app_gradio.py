import gradio as gr
import pandas as pd
import joblib
import os
import requests
from pathlib import Path

# ---------------- API KEY ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- PATH ----------------
BASE_DIR = Path(__file__).resolve().parent

# ---------------- LOAD MODEL ----------------
model = joblib.load(BASE_DIR / "model/health_model.pkl")
vectorizer = joblib.load(BASE_DIR / "model/vectorizer.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv(BASE_DIR / "data/Original_Dataset.csv")
df.columns = df.columns.str.strip()

desc_df = pd.read_csv(BASE_DIR / "data/Disease_Description.csv")
desc_df.columns = desc_df.columns.str.strip()

doc_df = pd.read_csv(BASE_DIR / "data/Doctor_Versus_Disease.csv")
doc_df.columns = doc_df.columns.str.strip()

# ---------------- EXTRACT SYMPTOMS ----------------
symptom_cols = [col for col in df.columns if col != "Disease"]

all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())

all_symptoms = sorted([str(s).strip() for s in all_symptoms if str(s).strip() != ""])

# ---------------- FUNCTIONS ----------------
def get_description(disease):
    disease = disease.strip()
    row = desc_df[desc_df.iloc[:, 0].str.strip() == disease]
    if not row.empty:
        return row.iloc[0][1]
    return "No description available"

def get_doctor(disease):
    disease = disease.strip()
    row = doc_df[doc_df.iloc[:, 0].str.strip() == disease]
    if not row.empty:
        return row.iloc[0][1]
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

        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return str(result)

    except Exception as e:
        return str(e)

# ---------------- MAIN FUNCTION ----------------
def predict(symptoms):
    if not symptoms:
        return "Please select symptoms", "", "", "", ""

    input_text = " ".join(symptoms)
    input_vec = vectorizer.transform([input_text])

    pred = model.predict(input_vec)[0]

    description = get_description(pred)
    doctor = get_doctor(pred)

    # AI Explanation
    prompt1 = f"""
    Disease: {pred}
    Explain:
    - What is it
    - Causes
    - Is it serious
    """
    explanation = ask_groq(prompt1)

    # AI Advice
    prompt2 = f"""
    Disease: {pred}
    Give advice:
    - Home remedies
    - Diet
    - When to see doctor
    """
    advice = ask_groq(prompt2)

    return pred, description, doctor, explanation, advice

# ---------------- GRADIO UI ----------------
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 AI Health Analyzer")

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