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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Scam Call Detector",
    page_icon="🚨",
    layout="centered"
)

# ---------------- LOAD WHISPER MODEL ----------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# ---------------- LOAD SCAM MODEL ----------------
@st.cache_resource
def load_scam_model():
    try:
        model_path = "scam_model_lstm.keras"
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None

# ---------------- LOAD TOKENIZER ----------------
@st.cache_resource
def load_tokenizer():
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error(f"Tokenizer Loading Error: {e}")
        return None

# ---------------- LOAD OBJECTS ----------------
whisper_model = load_whisper_model()
model = load_scam_model()
tokenizer = load_tokenizer()

# ---------------- TEXT CLEAN FUNCTION ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# ---------------- UI ----------------
st.title("🚨 Scam Call Detection System")
st.markdown("Whisper + LSTM based Audio Scam Detection")

st.markdown("### Upload Call Recording (Prefer WAV format)")

audio_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3"]
)

transcript = ""

# ---------------- AUDIO PROCESS ----------------
if audio_file is not None:

    st.audio(audio_file)

    # Save temp file
    suffix = ".wav" if audio_file.type == "audio/wav" else ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.getbuffer())
        temp_audio_path = tmp.name

    # Transcribe safely
    try:
        with st.spinner("Transcribing audio..."):
            result = whisper_model.transcribe(
                temp_audio_path,
                fp16=False
            )
            transcript = result["text"]

    except Exception as e:
        st.error("Audio processing failed. Try WAV file.")
        st.stop()

    # Remove temp file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    st.subheader("Transcribed Text")
    st.write(transcript)

# ---------------- PREDICTION ----------------
if st.button("Predict Scam"):

    if model is None or tokenizer is None:
        st.error("Model or Tokenizer not loaded properly")

    elif transcript.strip() == "":
        st.warning("Upload audio first")

    else:
        try:
            cleaned_text = clean_text(transcript)
            seq = tokenizer.texts_to_sequences([cleaned_text])
            padded_seq = pad_sequences(seq, maxlen=MAX_LEN)

            probability = float(model.predict(padded_seq)[0][0])

            st.subheader("Prediction Result")

            if probability >= THRESHOLD:
                st.error("🚨 SCAM CALL DETECTED")
            else:
                st.success("✅ NOT A SCAM CALL")

            st.metric("Scam Probability", f"{probability:.2f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Model: LSTM | Speech-to-Text: Whisper")
