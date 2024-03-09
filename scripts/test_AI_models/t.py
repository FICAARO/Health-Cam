
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
# Set the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set up the data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
#os.mkdir("train")
#os.mkdir("validation")
train_generator = train_datagen.flow_from_directory('train', target_size=(64, 64), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('validation', target_size=(64, 64), batch_size=32, class_mode='binary')

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10, validation_data=validation_generator, validation_steps=len(validation_generator))

# Save the model
model.save('dog_cat_classifier.h5')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('dog_cat_classifier.h5')

# Define a function to predict the class of an image
def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    if prediction[0][0] >= 0.5:
        return "Dog"
    else:
        return "Cat"

# Example usage
image_path = 'test/cat.jpg'
predicted_class = predict_image_class(image_path)
print("Predicted class:", predicted_class)

####work 