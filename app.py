from fastcore.all import *
from fastai.vision.all import *
import streamlit as st
from PIL import Image
import requests
import os

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

@st.cache_resource
def download_and_load_model():
    # Replace 'your_google_drive_file_id' with your actual Google Drive file ID
    google_drive_file_id = '1aypQw2s9Qge_FPjUbj7WqwujqOT8etA4'
    model_path = "export.pkl"

    # Download the model if it doesn't exist
    if not os.path.exists(model_path):
        try:
            download_file_from_google_drive(google_drive_file_id, model_path)
            st.write("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Error downloading the model: {e}")
            return None
    
    # Load the model
    try:
        learn = load_learner(model_path)
        return learn
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

st.title("Plant Disease Detection")

# Description paragraph
description = """
This project is trained on tomato, soybean, grape, apple, cassava, coffee, chilli, corn, cherry, guava, cucumber, lemon, 
mango, jamun, peach, pepper bell, rice, potato, sugarcane, strawberry, tea, wheat, pomegranate plants disease dataset.
"""
st.write(description)

st.write("Loading model...")
learn_inf = download_and_load_model()
if learn_inf is not None:
    st.write("Model loaded successfully!")
else:
    st.stop()  # Stop execution if model loading failed

# Classifier
def classify_img(data):
    img = PILImage.create(data)
    # Resize the image to 256x256
    img = img.resize((256, 256))
    # Disable progress bar
    with learn_inf.no_bar():
        pred, pred_idx, probs = learn_inf.predict(img)
    return pred, probs[pred_idx]

# Image uploader
bytes_data = None
uploaded_image = st.file_uploader("Upload image")
if uploaded_image:
    bytes_data = uploaded_image
    st.image(bytes_data, caption="Uploaded image")

# Classification
if bytes_data:
    classify = st.button("CLASSIFY!")
    if classify:
        label, confidence = classify_img(bytes_data)
        if confidence < 0.1:
            st.write("Not within trained class")
        else:
            st.write(f"It is a {label}! ({confidence:.04f})")
