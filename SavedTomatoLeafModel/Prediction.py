import numpy as np
from Library import *

def predict(path, Model):
    image = load_img(
        path,
        target_size=(256,256)
    )

    ConvertArray = img_to_array(image)
    Process = preprocess_input(ConvertArray)    # print(Process.shape)
    dimension = np.expand_dims(
        Process,
        axis=0
    )    # print(dimension.shape)
    Prediction = np.argmax(Model.predict(dimension))
    return Prediction



