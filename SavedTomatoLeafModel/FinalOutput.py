from Library import *


def FinalResult(predicted_value):
    train_datagen = ImageDataGenerator(
        zoom_range=0.5,
        shear_range=0.3,
        rescale=1 / 255,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    train = train_datagen.flow_from_directory(
        directory='TestData',
        target_size=(256, 256),
        batch_size=3
    )
    train_Data = train.class_indices
    values = train_Data.values()

    keys = train_Data.keys()
    zipping = dict(zip(values,keys))
    return zipping[predicted_value]
