# app.py
import os
import pickle
import string
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# --- robust paths (works when run from anywhere) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# --- ensure nltk resources exist (download into repo folder if needed) ---
nltk_data_dir = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

def ensure_nltk_resources():
    # resource pairs: (download_name, lookup_path)
    resources = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]
    for pkg, lookup in resources:
        try:
            nltk.data.find(lookup)
        except LookupError:
            # download quietly into nltk_data dir
            try:
                nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
            except Exception:
                # fallback: try download without download_dir (some NLTK versions)
                nltk.download(pkg, quiet=True)

ensure_nltk_resources()

# --- helper function ---
def transform_text(text):
    if not text:
        return ""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    stems = [ps.stem(t) for t in tokens]
    return " ".join(stems)

# --- load models (with friendly error messages) ---
if not os.path.exists(VECT_PATH) or not os.path.exists(MODEL_PATH):
    st.error("Model files not found. Make sure models/vectorizer.pkl and models/model.pkl are in the repo.")
    raise SystemExit

with open(VECT_PATH, "rb") as f:
    tfidf = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Streamlit UI ---
st.title("Email / SMS Spam Classifier")

input_sms = st.text_input("Enter the message to classify:")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please type a message to classify.")
    else:
        transformed = transform_text(input_sms)
        vector_input = tfidf.transform([transformed])  # correct shape: 1 x n_features
        pred = model.predict(vector_input)[0]          # do NOT wrap vector_input in []
        if int(pred) == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
