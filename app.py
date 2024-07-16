from fastcore.all import *
from fastai.vision.all import *
import streamlit as st
from PIL import Image
import requests
import os

# Function to download and load the model
@st.cache(allow_output_mutation=True)
def download_and_load_model():
    model_url = "https://huggingface.co/maureenmugo/Plant_disease_classification/resolve/main/export.pkl"
    model_path = "export.pkl"

    # Download the model if it doesn't exist
    if not os.path.exists(model_path):
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    
    # Load the model
    learn = load_learner(model_path)
    return learn

# Function to resize and classify image
def classify_img(data):
    img = PILImage.create(data)
    # Resize image to 256x256
    img = img.resize((256, 256))
    # Disable progress bar
    with learn_inf.no_bar():
        pred, pred_idx, probs = learn_inf.predict(img)
    return pred, probs[pred_idx]

# Streamlit app
st.title("Plant Disease Detection")

# Description paragraph
description = """
This project is trained on tomato, soybean, grape, apple, cassava, coffee, chilli, corn, cherry, guava, cucumber, lemon, 
mango, jamun, peach, pepper bell, rice, potato, sugarcane, strawberry, tea, wheat, pomegranate plants disease dataset.  
The link to the dataset used to train the model can be accessed here: https://www.kaggle.com/datasets/alinedobrovsky/plant-disease-classification-merged-dataset/data """
st.write(description)

# st.write("Loading model...")
learn_inf = download_and_load_model()
#st.write("Model loaded successfully!")

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
        if confidence < 0.4:
            st.write("Not within trained class")
        else:
            st.write(f"It is a {label}! ({confidence:.04f})")
