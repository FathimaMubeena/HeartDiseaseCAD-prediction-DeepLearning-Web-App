import os

import pandas as pd
import numpy as np
from keras.src.utils import to_categorical
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import classification_report

import ssl
import urllib.request

# Create an unverified SSL context
ssl_context = ssl._create_unverified_context()

# URL to open
dataset_url = 'https://archive.ics.uci.edu/static/public/45/data.csv'

# Open the URL with the unverified SSL context and read the CSV data
response = urllib.request.urlopen(dataset_url, context=ssl_context)

# Load your dataset here
df = pd.read_csv(response)

# handle missing values in 'ca' and 'thal'
missing_values_columns = ['ca', 'thal']
imputer = SimpleImputer(strategy='mean')  # Impute missing values with the mean
df[missing_values_columns] = imputer.fit_transform(df[missing_values_columns])

# Define features and target
X = df.drop('num', axis=1).values
y = df['num'].values

# One-hot encode the target variable (Option 1)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))  # One-hot encode y

# Alternatively, you can use to_categorical (Option 2)
# y = to_categorical(y, num_classes=5)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale the features (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE for class imbalance - oversample the minority classes
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Define the Keras multi-class classifier model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_resampled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))  # Output layer for multi-class classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_resampled, y_resampled, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on new data (test set)
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Evaluate the predictions using classification metrics
print(classification_report(true_classes, predicted_classes))

# Optionally, print the predicted classes for the first 10 examples
print(predicted_classes[:10])

# Save the model
model.save(os.path.join(os.path.dirname(__file__), '..', 'deploy', 'cad_classifer_model.keras'))

if __name__ == '__main__':
    pass