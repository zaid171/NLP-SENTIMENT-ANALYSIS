import streamlit as st
#import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model + vectorizer
model = joblib.load("/content/sentiment_model.pkl")
vectorizer = joblib.load("/content/vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

st.title("📊 AI Echo - Sentiment Analysis")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        st.success(f"Predicted Sentiment: {prediction}")
