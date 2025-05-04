import streamlit as st
from st_audiorec import st_audiorec
import librosa
import numpy as np
import joblib
from datetime import datetime
import io
import base64

# Load the pre-trained SVM model
model_path = "svm_accent_model.pkl"
clf = joblib.load(model_path)

# Function to create a download link for Streamlit
def create_download_link(data, filename):
    # Encode the data in base64 for download
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/wav;base64,{b64}" download="{filename}">Download</a>'
    return href
# Streamlit App Layout
st.title("Record and Download Audio")

# Record Audio with `st_audiorec`
st.markdown("### Step 1: Record Your Audio")
wav_audio_data = st_audiorec()  # Create the audio recorder instance

# Provide a download link once the recording is complete
if wav_audio_data is not None:
    st.markdown("### Step 2: Download Your Audio")
    # Create a unique filename for the audio file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audio.wav"
    
    # Create the download link
    download_link = create_download_link(wav_audio_data, filename)
    st.markdown(download_link, unsafe_allow_html=True)

# Step 3: File Uploader to Predict Accent
st.markdown("### Step 3: Upload Your Audio for Prediction")
uploaded_file = st.file_uploader("Upload the downloaded audio file (WAV format)", type=["wav"])

# If there's a file to process, predict the accent
if uploaded_file is not None:
    try:
        # Load the audio data with librosa
        audio_data, sr = librosa.load(uploaded_file, sr=None)

        # Check if the audio is too long
        if len(audio_data) > sr * 10:
            raise ValueError("The uploaded audio is too long for prediction.")

        # Predict the accent with the pre-trained SVM model
        accent = predict_accent(clf, audio_data, sr)

        # Display the predicted accent
        st.write(f"Predicted Accent: {accent[0]}")
    
    except ValueError as ve:
        st.error(f"Error: {str(ve)}")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
