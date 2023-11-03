import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os
from PIL import Image

# Load the CSV file with image metadata
csv_file_path = r"C:\Users\rahma\OneDrive\Desktop\Eye Disease Prediction Model\full_df.csv"
df = pd.read_csv(csv_file_path)

features_list = pickle.load(open("image_features_embedding4_new.pkl", "rb"))
img_files_list = pickle.load(open("img_files4_new.pkl", "rb"))

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

st.title('Ocular Disease Prediction')
st.write("This application supports retinal scan data for ocular disease prediction.")
st.write("Upload a retinal scan image to find similar images and their corresponding diseases.")

def save_file(uploaded_file):
    try:
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def extract_img_features(img_path, model):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normalized = flatten_result / norm(flatten_result)
    return result_normalized

def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    dist, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Choose your retinal scan image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if save_file(uploaded_file):
        # Display image
        show_images = Image.open(uploaded_file)
        size = (500, 500)
        resized_im = show_images.resize(size)
        st.image(resized_im, caption='Uploaded Image', use_column_width=True)
        
        # Extract features of the uploaded image
        features = extract_img_features(os.path.join("uploader", uploaded_file.name), model)
        
        # Find similar images
        img_indices = recommendd(features, features_list)
        
        st.subheader("Similar Images and Their Corresponding Diseases:")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        for i in range(5):
            img_index = img_indices[0][i]
            
            # Extract the filename (e.g., "0_left.jpg" or "0_right.jpg")
            filename = os.path.basename(img_files_list[img_index])
            
            # Determine whether it's a left or right image
            is_left = "_left" in filename
            is_right = "_right" in filename
            
            # Extract the diagnostic keywords based on the filename
            if is_left:
                diagnostic_keywords = df[df["Left-Fundus"] == filename]["Left-Diagnostic Keywords"].values
            elif is_right:
                diagnostic_keywords = df[df["Right-Fundus"] == filename]["Right-Diagnostic Keywords"].values
            else:
                diagnostic_keywords = []
            
            col = [col1, col2, col3, col4, col5][i]
            
            # Display the diagnostic keywords (disease names) if available
            col.image(img_files_list[img_index], use_column_width=True, caption='Similar Image')
            if len(diagnostic_keywords) > 0:
                col.write(f"Disease: {', '.join(diagnostic_keywords)}")
            else:
                col.warning(f"Disease information not found in the CSV for {filename}")
    else:
        st.error("Some error occurred while processing the image.")
