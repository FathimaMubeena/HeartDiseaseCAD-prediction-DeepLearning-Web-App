import os
import streamlit as st
import numpy as np
from keras.src.saving import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained Keras model
file_path = os.path.join(os.path.dirname(__file__), 'cad_classifer_model.keras')
model = load_model(file_path)

# CAD labels
cad_labels = {0: 'No CAD', 1: 'CAD'}

# Streamlit title and description
st.title('CAD Classifier')
st.write('Upload a patient record to classify the presence of coronary artery disease (CAD).')

# Upload the patient record through Streamlit, input fields for the features
age = st.number_input('Age', min_value=0, max_value=100, value=50)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

# Collect input data
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Preprocess the input data
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Make predictions using the model
predictions = model.predict(input_data_scaled)
predicted_class = np.argmax(predictions, axis=1)[0]

# Display the prediction results and make it bigger
st.write('---')
st.subheader('Prediction Results')
st.write(f'The model predicts: {cad_labels[predicted_class]}')


if __name__ == '__main__':
    pass