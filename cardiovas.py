import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset into a pandas DataFrame
data = pd.read_csv('cardio_dataset.csv', delimiter=';')

# Drop the 'id' column from the dataset
data = data.drop(columns=['id'])

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['cardio'])
y = data['cardio']

# Initialize and train a Logistic Regression model
clf = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# Create a Streamlit app
st.title("Cardiovascular Disease Prediction")

st.write("Enter information for a new person to predict whether they have cardiovascular disease or not:")

# Add interactive options for user input
age = st.slider("Age:", 0, 120, 40)
gender = st.selectbox("Gender", ['Male', 'Female'])  # Changed the options to 'Male' and 'Female'
height = st.number_input("Height (in cm):", min_value=0)
weight = st.number_input("Weight (in kg):", min_value=0.0)
ap_hi = st.number_input("Systolic Blood Pressure (ap_hi):", min_value=0)
ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo):", min_value=0)
cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
gluc = st.selectbox("Glucose Level", [1, 2, 3])
smoke = st.selectbox("Smoking", [0, 1])
alco = st.selectbox("Alcohol Consumption", [0, 1])
active = st.selectbox("Physical Activity", [0, 1])

# Create a DataFrame from the user input
custom_input = pd.DataFrame({
    'age': [age],
    'gender': [1 if gender == 'Female' else 2],  # Encode 'Female' as 0 and 'Male' as 1
    'height': [height],
    'weight': [weight],
    'ap_hi': [ap_hi],
    'ap_lo': [ap_lo],
    'cholesterol': [cholesterol],
    'gluc': [gluc],
    'smoke': [smoke],
    'alco': [alco],
    'active': [active]
})

if st.button("Predict"):
    # Use the trained Logistic Regression model to make a prediction
    prediction = clf.predict(custom_input)

    # Print the prediction
    if prediction == 0:
        st.write('This person probably does not have cardiovascular disease')
    else:
        st.write('This person is probably at risk of having cardiovascular disease')

# # Display model accuracy (no need to calculate it on every user interaction)
# st.write(f"Model Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.2%}")

