from tools.lib import *

from tools.predict import *

input_pat=""
p=predict.predict()
remove_background(input_pat,output_path)
predict_image(output_path)