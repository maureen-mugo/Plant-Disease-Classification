from fastcore.all import *
from fastai.vision.all import *
import streamlit as st
from PIL import Image
import io

# Load model
learn_inf = load_learner("export.pkl")

# Classifier
def classify_img(data):
    img = PILImage.create(data)
    # Disable progress bar
    with learn_inf.no_bar():
        pred, pred_idx, probs = learn_inf.predict(img)
    return pred, probs[pred_idx]

# Streamlit
st.title("Plant Disease Detection")

# Description paragraph
description = """
This project is trained on tomato, soybean, grape, apple, cassava, coffee, chilli, corn, cherry, guava, cucumber, lemon, 
mango, jamun, peach, pepper bell, rice, potato, sugarcane, strawberry, tea, wheat, pomegranate plants disease dataset.
"""
st.write(description)

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
        if confidence < 0.8:
            st.write("Not within trained class")
        else:
            st.write(f"It is a {label}! ({confidence:.04f})")
