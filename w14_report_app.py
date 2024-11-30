import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

#Model Loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5")

model = load_model()


#Streamlit UI
st.title("Handwritten Digit Classification")
st.write("Upload a 28x28 image, I will Classify")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        #Image Open and Resize
        image = Image.open(uploaded_file).convert("L")  # Grayscale
        image = ImageOps.invert(image)  # Color Opposite
        image = image.resize((28, 28))  # Resize
        image_array = np.array(image) / 255.0  #Regulization

        #Data Form Transferring
        image_array = np.expand_dims(image_array, axis=-1)  # (28, 28, 1)
        image_array = np.expand_dims(image_array, axis=0)   # (1, 28, 28, 1)

        #Checking Input Data Size
        st.write(f"Processed input shape: {image_array.shape}")

        #Model Accuracing
        prediction = model.predict(image_array, batch_size=1)
        predicted_label = np.argmax(prediction)

        #Result
        st.subheader("Prediction Conclusion")
        st.write(f"The model predicts this digit as: **{predicted_label}**")

        #Visualize Accurated Model
        st.bar_chart(prediction.flatten())

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

else:
    st.write("Please upload an image to get started.")