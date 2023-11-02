from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([model, GlobalMaxPooling2D()])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normalized = flatten_result / norm(flatten_result)

    return result_normalized

dataset_path = 'C:/Users/rahma/Downloads/Skin disease predictor/disease_images'
img_files = []

# Modify this part to extract disease names from folder names
disease_names = os.listdir(dataset_path)

for disease_name in disease_names:
    disease_path = os.path.join(dataset_path, disease_name)
    for image_file in os.listdir(disease_path):
        image_path = os.path.join(disease_path, image_file)
        img_files.append((image_path, disease_name))

# extracting image features and disease names
image_features = []
disease_names_list = []

for img_file, disease_name in tqdm(img_files):
    features_list = extract_features(img_file, model)
    image_features.append(features_list)
    disease_names_list.append(disease_name)

# Save the features and disease names
pickle.dump(image_features, open("image_features_embedding4.pkl", "wb"))
pickle.dump(disease_names_list, open("disease_names.pkl", "wb"))

model.save("fashion_model.h5")

# Run the below code un commented to get the img_files4.pkl file

# import os
# import pickle

# dataset_path = 'C:/Users/rahma/Downloads/Skin disease predictor/disease_images'
# img_files = []

# # Modify this part to extract image file paths from folder names
# disease_names = os.listdir(dataset_path)

# for disease_name in disease_names:
#     disease_path = os.path.join(dataset_path, disease_name)
#     for image_file in os.listdir(disease_path):
#         image_path = os.path.join(disease_path, image_file)
#         img_files.append(image_path)

# # Save the list of image file paths to img_files4.pkl
# pickle.dump(img_files, open("img_files4.pkl", "wb"))

