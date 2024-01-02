import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from icecream import ic
from keras.regularizers import l1, l2