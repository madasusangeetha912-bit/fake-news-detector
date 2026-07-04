import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Current folder details
print("Current Folder:", os.getcwd())
print("Files in Folder:", os.listdir())

# Check dataset
if not os.path.exists("news.csv"):
    print("❌ Error: news.csv file not found!")
    exit()

# Load dataset
data = pd.read_csv("news.csv")

# Remove missing values
data = data.dropna()

# Input and Output
X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model saved successfully!")