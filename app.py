import streamlit as st
import os
import re
import pickle
import tempfile
import whisper
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- SETTINGS ----------------
MAX_LEN = 50
THRESHOLD = 0.3

st.set_page_config(
    page_title="Scam Call Detector",
    page_icon="🚨",
    layout="centered"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_scam_model():
    try:
        model_path = os.path.join(os.getcwd(), "scam_model_lstm.keras")
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        tokenizer_path = os.path.join(os.getcwd(), "tokenizer.pkl")
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error(f"Tokenizer Loading Error: {e}")
        return None

# ---------------- LOAD OBJECTS ----------------
whisper_model = load_whisper_model()
model = load_scam_model()
tokenizer = load_tokenizer()

# ---------------- TEXT CLEAN ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# ---------------- UI ----------------
st.title("🚨 Scam Call Detection System")
st.markdown("**Whisper + LSTM based Audio Scam Detection**")
st.markdown("### Upload Call Recording (MP3 / WAV)")

audio_file = st.file_uploader(
    "Upload audio file",
    type=["mp3", "wav"]
)

transcript = ""

# ---------------- TRANSCRIPTION ----------------
if audio_file is not None:

    st.audio(audio_file)

    with tempfile.NamedTe
