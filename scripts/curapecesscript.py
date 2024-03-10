import numpy as np
import tensorflow as tf
import cv2
import sys
import zipfile
import tarfile
import os
#import pycuda.driver as cuda
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow.python.keras import backend as K

K.clear_session()
"""
directorio="./datos_limpios"
pruebas = 1
alturadelaimagen =150
longituddelaimagen= 150
numerodeimagenesamandar=2
pasos=100#numero de veces que se va aprosesar la informacion
validacon=183
filtroprimeravez= 32
filtrosegundavez= 64
filtroterceravez= 32
filtrocurtavez =128
filtroquintavez =256
filtrouno=(3,3)
filtrodos=(2,2)
filtrotres=(3,3)
filtrocutro=(4,4)
filtroquinto=(5,5)
pulido=(2,2)
numerodenfermedades=14# cambiar mientras encuenbtro imagenes y la sano cuenta como enfermedad
lr = 0.00004

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    directorio,
    target_size=(alturadelaimagen, longituddelaimagen),
    batch_size=numerodeimagenesamandar,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    directorio,
    target_size=(alturadelaimagen, longituddelaimagen),
    batch_size=numerodeimagenesamandar,
    class_mode='categorical')
nn = Sequential()
nn.add(Convolution2D(filtroprimeravez, filtrouno, padding ="same", input_shape=(longituddelaimagen, alturadelaimagen , 3), activation='relu'))
nn.add(MaxPooling2D(pool_size=pulido))

nn.add(Convolution2D(filtrosegundavez, filtrodos, padding ="same"))
nn.add(MaxPooling2D(pool_size=pulido))

nn.add(Flatten())
nn.add(Dense(512, activation='relu'))
nn.add(Dropout(0.5))
nn.add(Dense(numerodenfermedades, activation='softmax'))

#nn.compile(optimizer='sgd', loss='mse')
#nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


#nn.fit_generator(entrenamiento_generador ,validation_data=validacion_generador, steps_per_epoch=pasos,epochs=pruebas)
#nn.fit_generator(entrenamiento_generador,steps_per_epoch=pasos,epochs=pruebas,validation_data=validacion_generador,validation_steps=validacon)

nn.fit(entrenamiento_generador,steps_per_epoch=pasos,epochs=pruebas,validation_data=validacion_generador,validation_steps=validacon)
#model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10, validation_data=validation_generator, validation_steps=len(validation_generator))

nn=curapeces.nn()
target_dir = orendador.archivo_existe.archivo_existe()
nn.save(target_dir+ '/model.h5')
nn.save_weights(target_dir +'/weights.h5')


"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
# Set the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
batch_size=32
target_size_square=256
diases=14 #read folder train and len (folders) 


# Set up the data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
#os.mkdir("train")
#os.mkdir("validation")
train_generator = train_datagen.flow_from_directory('datos_limpios', target_size=(target_size_square, target_size_square), batch_size=batch_size, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('datos_limpios', target_size=(target_size_square, target_size_square), batch_size=batch_size, class_mode='binary')

# Build the CNN model
model = Sequential([
    Conv2D(target_size_square, (3, 3), activation='relu', input_shape=(target_size_square, target_size_square, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(target_size_square*2, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(target_size_square*2, (3, 3), activation='relu'),
    Flatten(),
    Dense(target_size_square*2, activation='relu'),
    Dense(diases, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10, validation_data=validation_generator, validation_steps=len(validation_generator))

# Save the model
model.save('dog_cat_classifier.h5')