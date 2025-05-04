import streamlit as st
import librosa
import numpy as np
import joblib
from st_audiorec import st_audiorec
from datetime import datetime
import io
import base64

# Load the pre-trained model
model_path = "svm_accent_model.pkl"  # Path to your trained SVM model
clf = joblib.load(model_path)

# Function to create a download link for Streamlit
def create_download_link(data, filename):
    # Encode the data in base64 for download
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/wav;base64,{b64}" download="{filename}">Download</a>'
    return href

# Feature Extraction Function
def extract_features(audio_data, sr, start=1, end=19):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr)
    feature_vector = mfcc.T[1][start:end]
    for frame in range(10, 50):
        feature_vector = np.concatenate((feature_vector, mfcc.T[frame][start:end]))
    return feature_vector

# Function to Predict Accent
def predict_accent(model, audio_data, sr):
    feature_vector = extract_features(audio_data, sr)
    prediction = model.predict([feature_vector])
    return prediction

# Streamlit App Layout
st.title("Accent Prediction from Audio")

# File uploader to accept audio files
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

# Prediction Button
if uploaded_file is not None:
    st.write("Processing the uploaded file...")
    try:
        # Load the audio data
        audio_data, sr = librosa.load(uploaded_file, sr=None)
        
        # Predict the accent
        accent = predict_accent(clf, audio_data, sr)
        
        # Display the predicted accent
        st.write(f"Predicted Accent: {accent[0]}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
