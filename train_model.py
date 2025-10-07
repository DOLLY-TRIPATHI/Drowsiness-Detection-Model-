import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_path = 'D:/Driver_Drowsiness_Project/dataset/train'
test_path = 'D:/Driver_Drowsiness_Project/dataset/test'

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    train_path, target_size=(24,24), color_mode='grayscale', batch_size=64, class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
val_data = test_datagen.flow_from_directory(
    test_path, target_size=(24,24), color_mode='grayscale', batch_size=64, class_mode='binary'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_data, validation_data=val_data, epochs=3)

# Save model
model.save("drowsiness_model.h5")

# Plot
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.legend()
plt.show()
