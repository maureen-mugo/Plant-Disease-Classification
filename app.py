import streamlit as st
from fastai.vision.all import *

# Load your trained model
model_path = '/home/maureen/Code/personal-repos/projects/computer vision/Plant-Disease-Classification/export.pkl'
learn = load_learner(model_path)

st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect the disease")

# File uploader
uploaded_file = st.file_uploader("Choose a file...", type="jpg")

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img.to_thumb(512, 512), caption='Uploaded Image.', use_column_width=True)

    # Make prediction
    pred, pred_idx, probs = learn.predict(img)
    st.write(f"Prediction: {pred}")
    st.write(f"Probability: {probs[pred_idx]:.4f}")
