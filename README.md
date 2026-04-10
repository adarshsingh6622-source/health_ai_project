🧠 AI Health Analyzer

An AI-powered health prediction system that analyzes user symptoms and predicts possible diseases using Machine Learning, Data Science, and Deep Learning techniques, along with AI-generated explanations and medical advice.

🚀 Features
🩺 Disease prediction based on symptoms
🤖 AI-powered explanation of the disease
💊 Personalized health advice (diet, remedies, precautions)
👨‍⚕️ Doctor recommendation based on predicted disease
📊 Confidence-based prediction (ML model)
🌐 Interactive UI using Gradio
🧠 Technologies Used
🔹 Data Science
 . Data cleaning and preprocessing
 . Handling missing values and text normalization
 . Feature engineering from symptom data
🔹 Machine Learning
 . TF-IDF Vectorization for text feature extraction
 . Classification model for disease prediction
 . Label Encoding for categorical output
🔹 Deep Learning
 . Neural Network (Dense layers) for improved prediction
 . Activation functions (ReLU, Softmax)
 . Model training using TensorFlow/Keras
🔹 AI Integration
. Groq API for:
  . Disease explanation
  . Health advice
  . Smart medical insights
📂 Project Structure
health_ai_project/
│
├── app.py                # Main Gradio App
├── app_gradio.py         # Alternative UI (optional)
├── requirements.txt
│
├── data/
│   ├── Disease_Description.csv
│   ├── Doctor_Versus_Disease.csv
│   └── Original_Dataset.csv
│
├── model/
│   ├── health_model_v2.pkl
│   ├── vectorizer_v2.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── train.py
│   └── analysis.py
⚙️ How It Works
1. User selects symptoms
2. Symptoms are converted into numerical features using TF-IDF
3. ML/DL model predicts the disease
4. System retrieves:
 . Disease description
 . Recommended doctor
5. AI (Groq API) generates:
 . Explanation
 . Advice
🛠️ Installation
git clone <your-repo-link>
cd health_ai_project
pip install -r requirements.txt
▶️ Run the App
python app.py

or (if using Gradio):

python app_gradio.py
⚠️ Important Note

Model files are required for this app to run:

. health_model_v2.pkl
. vectorizer_v2.pkl
. label_encoder.pkl

Make sure these files are present inside the model/ folder.

📈 Future Improvements
. Improve model accuracy using larger datasets
. Add real-time doctor consultation
. Deploy mobile application
. Integrate more advanced deep learning models
👨‍💻 Author

Adarsh Singh
Aspiring Data Scientist | Machine Learning Enthusiast 🚀





