import streamlit as st
import tensorflow as tf
from tensorflow import keras
import requests
import numpy as np

# Create a title of web App and Introductory note.
st.title('Welcome to my Image Classification Project!')
st.text('This app classifies images whether they belong to any of the 10 classes on which my ')
st.text('model was trained. The dataset I used for my capstone project was ')
st.text('the CIFAR-10 image dataset and the class names are:')
st.text('class_names = ["airplane", "automobile" , "bird" , "cat" , "deer" , "dog",')
st.text('"frog" ,horse" , "ship" , "truck"].')
st.text("For better prediction, please use URL for images among the 'class_names'...")
st.text("My model is expected to predict with 86% accuracy, but let's see :)")



#load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('models/bestModel.h5')
    return model

with st.spinner("Loading Model...."):
    model=load_model()

#classes for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# image preprocessing
def load_image(image):
    img=tf.image.decode_jpeg(image,channels=3)
    img=tf.cast(img,tf.float32)
    img/=255.0
    img=tf.image.resize(img,(32,32))
    img=tf.expand_dims(img,axis=0)
    return img

#Get image URL from user
image_path=st.text_input("Enter Image URL to classify...","https://i.insider.com/5484d9d1eab8ea3017b17e29?width=600&format=jpeg&auto=webp")

#Get image from URL and predict
if image_path:
    try:
        content=requests.get(image_path).content
        st.write("Predicting Class...")
        with st.spinner("Classifying..."):
            img_tensor=load_image(content)
            pred=model.predict(img_tensor)
            pred_class=class_names[np.argmax(pred)]
            st.write("Predicted Class:",pred_class)
            st.image(content,use_column_width=True)
    except:
        st.write("Invalid URL")
