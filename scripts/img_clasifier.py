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
learn.save('model')
print(learn.predict(files[0]))

learn.show_results()

learn.load('my_model')

print(learn.predict(files[0]))


learn.show_results()
files[0].name
pat = r'^(.*)_\d+.jpg'
dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(224))

dls.show_batch()
dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(460),
                                    batch_tfms=aug_transforms(size=224))

dls.show_batch()
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.lr_find()
learn.fine_tune(2, 3e-3)

learn.show_results()
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))
