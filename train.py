import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
image_size = (300, 300)
batch_size = 32
epochs = 10

# Load the dataset
data_folder = "C:\Users\hp\Desktop\pycharm_proj\sign_lang_detec\Data"
labels = os.listdir(data_folder)
X, y = [], []

for label in labels:
    label_path = os.path.join(data_folder, label)
    for image_name in os.listdir(label_path):
        image_path = os.path.join(label_path, image_name)
        img = cv2.imread(image_path)
        img = cv2.resize(img, image_size)
        X.append(img)
        y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode labels
label_encoder = sklearn.preprocessing.LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create a simple convolutional neural network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

# Train the model
model.fit(datagen.flow(X_train, y_train_encoded, batch_size=batch_size), epochs=epochs, validation_data=(X_test, y_test_encoded))

# Save the model
model.save("hand_gesture_model.h5")

# Save the label encoder
np.save("label_encoder.npy", label_encoder.classes_)
