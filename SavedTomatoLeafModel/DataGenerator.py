from Library import *


def TestDataGenerator():
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test = val_datagen.flow_from_directory(
        directory='TestData',
        target_size=(256, 256),
        batch_size=32
    )

    return test