import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model("C:/Users/balde/OneDrive/Bureau/DSTI/Project/Deep Learning/DEEP-LEARNING-PROJECT/cnn_xception_model.h5")

# Streamlit app title
st.title("Pneumonia Detection Web App")

# Description
st.write("Upload an X-ray image, and I'll tell you if it shows signs of pneumonia.")

# Function to make predictions
def predict_pneumonia(image):
    # Preprocess the image for the model
    img = Image.open(image).convert('RGB')
    img = img.resize((128, 128))  # Ensure the same size as your training data
    img = preprocess_input(np.array(img))
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    prediction = model.predict(img)
    return prediction

# Upload an image
uploaded_image = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded X-ray Image", use_column_width=True)

    # Make predictions
    prediction = predict_pneumonia(uploaded_image)

    # Display the results
    if prediction[0, 0] > 0.5:  # Adjust the threshold as per your model's output
        st.write("Result: Normal")
    else:
        st.write("Result: Pneumonia")

# Add a section to display results from your notebook
st.subheader("Results from Notebook")
# You can display the results from your notebook here, e.g., charts, tables, etc.

# Footer or additional information
st.write("This is a simple Streamlit app for pneumonia detection.")
