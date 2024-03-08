import numpy as np
import tensorflow as tf
import cv2
import sys
import zipfile
import tarfile
import os
#import pycuda.driver as cuda
#import pycuda.autoinit
#from pycuda.compiler import SourceModule
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow.python.keras import backend as K
#from numba import cuda
#import numba
import orendador


class curapeces():
    K.clear_session()
    """solo si es para  la enefemedad"""
    #especimendepractica = './especimendepractica/peces/enfermos/'
    #especimendeexamen= './especimendeexamen/peces/enfermos/'
    """solo si es para  la sie eta sano"""
    #especimendepractica = './especimendepractica/peces/'
    #especimendeexamen= './especimendeexamen/peces/'
    """solo si es para  si es un pez"""
    #especimendepractica = './especimendepractica'
    #especimendeexamen= './especimendeexamen'
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
    #numerodenfermedades=2
    lr = 0.00004


    def image(self):
        self.entrenamiento_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.entrenamiento_generador = self.entrenamiento_datagen.flow_from_directory(
            self.directorio,
            target_size=(self.alturadelaimagen, self.longituddelaimagen),
            batch_size=self.numerodeimagenesamandar,
            class_mode='categorical')

        self.validacion_generador = self.test_datagen.flow_from_directory(
            self.directorio,
            target_size=(self.alturadelaimagen, self.longituddelaimagen),
            batch_size=self.numerodeimagenesamandar,
            class_mode='categorical')
    def nn(self):
    
        nn = Sequential()
        nn.add(Convolution2D(self.filtroprimeravez, self.filtrouno, padding ="same", input_shape=(self.longituddelaimagen, self.alturadelaimagen , 3), activation='relu'))
        nn.add(MaxPooling2D(pool_size=self.pulido))

        nn.add(Convolution2D(self.filtrosegundavez, self.filtrodos, padding ="same"))
        nn.add(MaxPooling2D(pool_size=self.pulido))

        # nn.add(Convolution2D(self.filtroterceravez, self.filtrotres, padding ="same"))
        # nn.add(MaxPooling2D(pool_size=self.pulido))
        #
        # nn.add(Convolution2D(self.filtrocurtavez, self.filtrocutro, padding ="same"))
        # nn.add(MaxPooling2D(pool_size=self.pulido))
        #
        # nn.add(Convolution2D(filtroquintavez, filtroquinto, padding ="same"))
        # nn.add(MaxPooling2D(pool_size=pulido))

        nn.add(Flatten())
        nn.add(Dense(512, activation='relu'))
        nn.add(Dropout(0.5))
        nn.add(Dense(self.numerodenfermedades, activation='softmax'))
        nn.compile(optimizer='sgd', loss='mse')
        #cnn.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])




        nn.fit_generator(self.entrenamiento_generador,steps_per_epoch=self.pasos,epochs=self.pruebas,validation_data=self.validacion_generador,validation_steps=self.validacon)
        #cnn.save('./modelo_lab_experimental/modelo_pezenfermo.h5')
        #cnn.save_weights('./modelo_lab_experimental/pesospezenfermo.h5')
        return nn
    def save_nn(self):
        self.nn=curapeces.nn()
        self.target_dir = orendador.archivo_existe.archivo_existe()
        self.nn.save(self.target_dir+ '/model.h5')
        self.nn.save_weights(self.target_dir +'/weights.h5') 



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)# 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x