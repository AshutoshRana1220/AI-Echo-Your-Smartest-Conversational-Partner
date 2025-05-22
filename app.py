import streamlit as st
import xgboost as xgb
import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer

# --- Load Saved CountVectorizer and XGBoost Model ---
cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
model = xgb.Booster()
model.load_model('xgboost_sentiment_model.json')

# --- Function to preprocess and predict ---
def predict_sentiment(text):
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    X = cv.transform([text])
    dmatrix = xgb.DMatrix(X)
    pred = int(model.predict(dmatrix)[0])

    if pred == 2:
      return "Positive 😊"
    elif pred == 1:
      return "Neutral 😐"
    else:
      return "Negative 😞"


# --- Streamlit UI ---
st.set_page_config(page_title="ChatGPT Review Sentiment Analyzer", layout="centered")

st.title("🧠 ChatGPT Review Sentiment Analyzer")
st.markdown("Analyze the sentiment of user reviews about ChatGPT")

user_input = st.text_area("Enter your review below 👇", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")

st.markdown("---")
st.markdown("💡 **Model Info**: XGBoost (multi-class classification)\n\n🔢 **Sentiment Labels**: 0 = Negative, 1 = Neutral, 2 = Positive")

