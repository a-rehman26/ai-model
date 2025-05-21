import streamlit as st
import librosa
import numpy as np
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import joblib
import av  # Required for audio processing

# Streamlit app title
st.title("🎙️ Gender Voice Detection with Audio Upload & Live Stream")

# --- Load pre-trained model (assumed to be stored in 'gender_model.pkl') ---
try:
    model = joblib.load("gender_model.pkl")  # Load the model
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error("Model file 'gender_model.pkl' not found. Please upload or train a model.")
    model = None

# ------------------------- Feature Extraction -------------------------
def extract_features(audio_input, sr=None):
    """
    Extracts pitch (spectral centroid), MFCC, and Chroma features from audio.
    Supports both uploaded files and raw numpy audio from live stream.
    """
    y = None

    if isinstance(audio_input, np.ndarray):
        y = audio_input
        if sr is None:
            st.error("Sample rate (sr) must be provided for live audio.")
            return 0.0, 0.0, 0.0
    else:
        try:
            y, sr = librosa.load(audio_input, sr=sr)
        except Exception as e:
            st.error(f"Error loading audio: {e}")
            return 0.0, 0.0, 0.0

    if y is None or len(y) < sr * 0.1:
        st.warning("Audio is too short or invalid.")
        return 0.0, 0.0, 0.0

    try:
        pitch = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return 0.0, 0.0, 0.0

    mean_pitch = np.mean(pitch)
    mean_mfcc = np.mean(mfcc)
    mean_chroma = np.mean(chroma)
    
    return mean_pitch, mean_mfcc, mean_chroma

# ------------------------- Gender Prediction using ML Model -------------------------
def predict_gender(features):
    """
    Predict gender using the pre-trained ML model based on extracted features.
    """
    if model is None:
        st.error("Model is not available. Cannot predict gender.")
        return "Model unavailable"

    pitch, mfcc, chroma = features
    feature_array = np.array([[pitch, mfcc, chroma]])  # Combine features into a 2D array

    # Predict gender using the model
    prediction = model.predict(feature_array)
    return "Male" if prediction[0] == 0 else "Female"  # Assuming 0 = Male, 1 = Female

# ------------------------- Live Audio Processor -------------------------
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame):
        audio_data = frame.to_ndarray()
        sample_rate = frame.sample_rate
        features = extract_features(audio_data, sr=sample_rate)
        gender = predict_gender(features)
        st.write(f"🎤 Live Prediction: **{gender}**")
        return frame

# ------------------------- UI: File Upload -------------------------
st.subheader("📁 Upload an Audio File")
audio_file = st.file_uploader("Upload WAV, MP3, or FLAC audio", type=["wav", "mp3", "flac"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    st.write("Processing uploaded audio...")
    features = extract_features(audio_file)
    if features != (0.0, 0.0, 0.0):
        gender = predict_gender(features)
        st.success(f"✅ Predicted Gender: **{gender}**")
    else:
        st.error("Failed to process the audio.")

# ------------------------- UI: Live Audio -------------------------
st.subheader("🎙️ Live Microphone Input")
st.info("Allow microphone access in your browser to use this feature.")

webrtc_streamer(
    key="gender-detection",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False}
)

# ------------------------- Footer -------------------------
st.markdown("""
---
💡 *This app uses an ML model to predict gender based on voice features like pitch, MFCC, and Chroma.*
If you need help training the model or have any issues, feel free to reach out.
""")
