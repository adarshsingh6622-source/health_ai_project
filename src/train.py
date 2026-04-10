import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
df = pd.read_csv("data/Original_Dataset.csv")
df.columns = df.columns.str.strip()

# Combine symptoms
df = df.fillna("")
df["combined"] = df.apply(lambda row: " ".join(map(str, row.values)), axis=1)

# Cleaning
df["combined"] = df["combined"].str.replace("nan", "", regex=False)
df["combined"] = df["combined"].str.replace(",", "", regex=False)
df = df[df["combined"].str.strip() != ""]

# Features & target
X = df["combined"]
y = df["Disease"]

# Vectorize
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X).toarray()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_encoded, test_size=0.2, random_state=42
)

# Deep Learning model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=16)

# Save
os.makedirs("model", exist_ok=True)
model.save("model/health_model_dl.h5")
joblib.dump(vectorizer, "model/vectorizer_dl.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("✅ Model trained successfully")

