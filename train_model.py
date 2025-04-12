import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 📌 Load the OLD dataset
df_old = pd.read_csv("dataset/spam_assassin.csv", encoding="ISO-8859-1")

# 📌 Load the NEW dataset
df_new = pd.read_csv("dataset/large_balanced_emails.csv", encoding="ISO-8859-1")

# 📌 Fix Column Names for Both Datasets
if "spamORham" in df_old.columns and "Message" in df_old.columns:
    df_old = df_old.rename(columns={"spamORham": "label", "Message": "email_text"})

if "Sender" in df_new.columns and "Subject" in df_new.columns and "Message" in df_new.columns:
    df_new = df_new.rename(columns={"Message": "email_text", "Label": "label"})
else:
    raise ValueError("New dataset does not have expected columns like 'Sender', 'Subject', 'Message', 'Label'.")

# 📌 Ensure correct label mapping for both datasets
df_old["label"] = df_old["label"].map({"ham": 0, "spam": 1})
df_new["label"] = df_new["label"].map({"ham": 0, "spam": 1})

# 📌 Drop missing values
df_old.dropna(subset=["label", "email_text"], inplace=True)
df_new.dropna(subset=["label", "email_text"], inplace=True)

# 📌 Combine Old + New Data Without Affecting Old Data
df_combined = pd.concat([df_old, df_new], ignore_index=True)

# **TF-IDF Feature Extraction**
vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b")
X = vectorizer.fit_transform(df_combined["email_text"]).toarray()
y = df_combined["label"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **LSTM Model (Fixed Input Shape)**
vocab_size = X.shape[1]
model = Sequential([
    Dense(128, activation="relu", input_shape=(vocab_size,)),
    Dense(64, activation="relu"),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# **Train Model on Combined Data**
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# **Save New Model Separately**
model.save("model/spam_model_v3.keras")  # 📌 Save as v3 so old model is not affected

# **Save TF-IDF Vectorizer**
with open("model/tfidf_vectorizer_v3.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("✅ New Model Trained with Old + New Data! Saved as 'spam_model_v3.keras' ✅")
