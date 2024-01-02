from Library import *

def DownloadingNeuronModel():
    base_model = VGG19(
        weights='imagenet',
        input_shape=(256,256,3),
        include_top=False
    )

    for layer in base_model.layers:
        layer.trainable = False

    return base_model
