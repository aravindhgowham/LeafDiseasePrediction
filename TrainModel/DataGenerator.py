from Library import *

def Image_Data_generator():
    train_datagen = ImageDataGenerator(
        zoom_range=0.5,
        shear_range=0.3,
        rescale=1 / 256,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train = train_datagen.flow_from_directory(
        directory='tomato/train',
        target_size=(256, 256),
        batch_size=32
    )

    val = val_datagen.flow_from_directory(
        directory='tomato/val',
        target_size=(256, 256),
        batch_size=32
    )

    test_image, test_label = train.next()

    dictonary = {
        "train":train,
        "Validation":val,
        "test_image":test_image,
        "test_label":test_label
    }

    ic(val.class_indices)
    ic(train.class_indices)
    return dictonary
