from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import os
import numpy as np
import time