import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os

# Load the pre-computed features, disease names, and image file paths
features_list = pickle.load(open("image_features_embedding4.pkl", "rb"))
disease_names_list = pickle.load(open("disease_names.pkl", "rb"))
img_files_list = pickle.load(open("img_files4.pkl", "rb"))

# Load the pre-trained ResNet model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

st.title('SKIN DISEASE PREDICTOR')

def save_file(uploaded_file):
    try:
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normalized = flatten_result / norm(flatten_result)

    return result_normalized

def recommend(features, features_list, disease_names_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distance, indices = neighbors.kneighbors([features])

    return indices

def recognize_disease(features, features_list, disease_names_list):
    indices = recommend(features, features_list, disease_names_list)
    recognized_disease = disease_names_list[indices[0][0]]
    return recognized_disease

uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    if save_file(uploaded_file):
        # Display the uploaded image
        show_images = image.load_img(os.path.join("uploader", uploaded_file.name), target_size=(224, 224))
        st.image(show_images, caption="Uploaded Image", use_column_width=True)

        # Extract features of the uploaded image
        features = extract_img_features(os.path.join("uploader", uploaded_file.name), model)

        # Recognize the disease based on the uploaded image
        recognized_disease = recognize_disease(features, features_list, disease_names_list)

        st.subheader("Recognized Skin Disease:")
        st.write(recognized_disease)

        # Find similar images
        img_indices = recommend(features, features_list, disease_names_list)

        st.write("Top 5 similar skin disease images:")
        
        col1, col2, col3, col4, col5 = st.columns(5)

        for i in range(5):
            with col1 if i % 5 == 0 else col2 if i % 5 == 1 else col3 if i % 5 == 2 else col4 if i % 5 == 3 else col5:
                st.subheader(disease_names_list[img_indices[0][i]])
                st.image(image.load_img(img_files_list[img_indices[0][i]], target_size=(224, 224)), caption=disease_names_list[img_indices[0][i]], use_column_width=True)
