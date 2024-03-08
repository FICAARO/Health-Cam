from rembg import remove 
from PIL import Image  
def remove_bg(input_path,output_path):
	input_image = Image.open(input_path) 
	output = remove(input_image) 
	output.save(output_path) 

#https://www.geeksforgeeks.org/how-to-remove-the-background-from-an-image-using-python/