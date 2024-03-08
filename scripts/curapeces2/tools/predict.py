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

 class predict():
    pez="./test_fish/"+input("chose a image and put file extision ... png jpg\n")
    imagenpez = cv2.imread(pez, cv2.IMREAD_COLOR)
    numfolders=len(os.listdir("modelos_de_inteligencia_artificial_variedad"))-1
    modelfolder="./modelos_de_inteligencia_artificial_variedad/curapeces"+str(numfolders)+"__models curapeces__2019-11-29"
    model=modelfolder+"/model.h5"
    weights=modelfolder+"/weights.h5"
    longitud, altura = 150, 150
    def display_image(self):
        cv2.imshow ('ventana1',self.imagenpez)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def predict(self):
      self.nn =  tf.keras.models.load_model(self.model)
      self.nn.load_weights(self.weights)
      self.x = load_img(self.pez, target_size=(self.longitud, self.altura))
      self.x = img_to_array(self.x)
      self.x = np.expand_dims(self.x, axis=0)
      self.array = self.nn.predict(self.x)
      self.result = self.array[0]
      self.answer = np.argmax(self.result)
      #x2 = load_img(file, target_size=(longitud, altura))
      #x2 = img_to_array(x2)
      #x2 = np.expand_dims(x2, axis=0)
      #array = cnn2.predict(x2)
      #result = array[0]
      #self.answer = np.argmax(result)
      if self.answer == 0:
        print("prediccion:  atcado o tumor y deformidad")
      #x3 = load_img(file, target_size=(longitud, altura))
      #x3 = img_to_array(x3)
      #x3 = np.expand_dims(x3, axis=0)
      #array = cnn3.predict(x3)
      #result = array[0]
      #self.answer = np.argmax(result)
      elif self.answer ==1 :
        print("prediccion: branquias ")
      elif self.answer ==2 :
        print("prediccion: girodactilo ")
      elif self.answer == 3:
        print("prediccion: gusano lernea ")
      elif self.answer ==4 :
        print("prediccion: hidropecia ")
      elif self.answer == 5:
        print("prediccion: hongos")
      elif self.answer ==6 :
        print("prediccion: huecos en la cabesa")
      elif self.answer == 7 :
        print("prediccion: ich ")
      elif self.answer ==8 :
        print("prediccion: no es un pez")
      elif self.answer == 9:
        print("prediccion: ojo picho ")
      elif self.answer == 10:
        print("parasito en la lengua")
      elif self.answer == 11:
        print("prediccion: podredumbre de aletas ")
      elif self.answer == 12:
        print("prediccion: quemadura de bagre ")
      elif self.answer == 13:
        print("prediccion: es un pez sano")
      #pare="noes.jpg"
      #noespez = cv2.imread(pare, cv2.IMREAD_COLOR)
      #noespezres = cv2.resize(noespez,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
      #seleccion = cv2.add(noespezres,imagenpez)
      #cv2.imshow ("ventana2",seleccion)
      return self.answer
