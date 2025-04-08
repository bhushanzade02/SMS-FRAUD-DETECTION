import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Preprocessing function
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = re.findall(r'\b\w+\b', text)
    text = [word for word in text if word not in ENGLISH_STOP_WORDS]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            color:white;
            font-size: 48px;
            font-weight: bold;
        }
        .result-box {
            background-color: #f1f3f6;
            padding: 1.5em;
            border-radius: 10px;
            font-size: 24px;
            text-align: center;
            margin-top: 20px;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #888;
            margin-top: 2em;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title"> SMS SPAM DETECTOR</div>', unsafe_allow_html=True)
st.write(" Detect whether a message is **Spam** or **Not Spam** using a trained ML model.")

# Text input
input_sms = st.text_area(" Enter your message here:", height=150)

# Prediction
if st.button('🚀 Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0][1]

        if result == 1:
            st.markdown('<div class="result-box" style="color:red;">🚫 Spam Message</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="color:green;">✅ Not Spam</div>', unsafe_allow_html=True)

        st.info(f" Model confidence: **{proba:.2f}**")

# Footer
st.markdown('<div class="footer">Made with ❤️ by Bhushan Zade</div>', unsafe_allow_html=True)
