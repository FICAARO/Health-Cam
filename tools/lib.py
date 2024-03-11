from rembg import remove 
from PIL import Image 
def remove_background(input_pat,output_path):
    input_img = Image.open(input_path) 
    output_img = remove(input_img) 
    output_img.save(output_path) 

#https://www.geeksforgeeks.org/how-to-remove-the-background-from-an-image-using-python/