import streamlit as st
import os
import re
import pickle
import tempfile
import whisper
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 50
THRESHOLD = 0.3

st.set_page_config(
    page_title="Scam Call Detector",
    page_icon="🚨",
    layout="centered"
)

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_scam_model():
    return tf.keras.models.load_model(
        "scam_model_lstm.keras",
        compile=False
    )

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

whisper_model = load_whisper_model()
model = load_scam_model()
tokenizer = load_tokenizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

st.title("🚨 Scam Call Detection System")
st.markdown("**Whisper + LSTM based Audio Scam Detection**")
st.markdown("### Upload Call Recording (MP3 / WAV)")

audio_file = st.file_uploader(
    "Upload audio file",
    type=["mp3", "wav"]
)

transcript = ""

if audio_file is not None:
    st.audio(audio_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.getbuffer())
        temp_audio_path = tmp.name

    tmp.close()

    with st.spinner("🎧 Transcribing audio using Whisper..."):
        transcript = whisper_model.transcribe(temp_audio_path)["text"]

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    st.subheader("📝 Transcribed Text")
    st.write(transcript)

if st.button("🔍 Predict Scam"):
    if transcript.strip() == "":
        st.warning("Please upload an audio file first.")
    else:
        cleaned_text = clean_text(transcript)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded_seq = pad_sequences(seq, maxlen=MAX_LEN)
        probability = float(model.predict(padded_seq)[0][0])

        st.subheader("📊 Prediction Result")

        if probability >= THRESHOLD:
            st.error("🚨 SCAM CALL DETECTED")
        else:
            st.success("✅ NOT A SCAM CALL")

        st.metric("Scam Probability", f"{probability:.2f}")

st.markdown("---")
st.caption("Model: LSTM | Speech-to-Text: Whisper | Threshold = 0.3")
