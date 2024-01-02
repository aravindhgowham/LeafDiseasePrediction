from Library import *


def LoadingSavedModel(val):
    model = load_model('../SavedModel/best_model.h5')
    accuracy = model.evaluate_generator(val)[1]
    return accuracy
