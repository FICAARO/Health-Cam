import matplotlib.pyplot as plt
from PIL import Image

from fastai.vision.all import *
path = untar_data(URLs.PETS)

files = get_image_files(path/"images")
print(len(files))

print(files[0],files[6])

def label_func(f): return f[0].isupper()

dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

dls.show_batch()

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
print(learn.predict(files[0]))

learn.show_results()

learn.save('my_model')
learn.load('my_model')
#############
print(files[0],learn.predict(files[0]))


learn.show_results()

### funciona