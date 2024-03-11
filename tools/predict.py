from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
# Load the trained model
class predict:
	def __init__(self):
		self.model = load_model('fishes.h5')
		self.diases_list=os.listdir("train")
		self.target_size_square=512-128
		self.diases=["atacado_tumor_y_deformidad",  "girodactilo","hidropecia_y_vegiga_nadatoria","huecos en la cabesa",   "nopez","parasito_en_la_boca", "quemadura_bagre", "branquias","gusano_lernea"   ,"hongos","ich_punto_planco","ojo_picho","podredumbre aleta","sanos"]

	def predict_image(image_path):
		img = image.load_img(image_path, target_size=(target_size_square, target_size_square))
		img_array = image.img_to_array(img)
		img_array = np.expand_dims(img_array, axis=0)
		img_array /= 255.

		prediction = model.predict(img_array)[0]
		max_val=0
		max_pos=0
		for i in range(len(prediction)):
			if max_val<prediction[i]:
				max_val=prediction[i]
				max_pos=i
		return prediction[max_pos]

