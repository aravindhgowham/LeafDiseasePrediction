from Library import *


def early_stopping():
    #stopping trained model if not imporve
    es = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.01,
        patience=3,
        verbose=1
    )

    # model Check Point
    mc = ModelCheckpoint(
        filepath="../SavedModel/best_model.h5",
        monitor='val_accuracy',
        min_delta=0.01,
        patience=3,
        verbose=1,
        save_best_only=True
    )

    # callback
    cb = [es, mc]
    return cb

