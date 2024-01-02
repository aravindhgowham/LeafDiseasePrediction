from Library import *


def TrainedModel():
    model = load_model('../SavedModel/best_model.h5')
    print("model",model)
    return model

