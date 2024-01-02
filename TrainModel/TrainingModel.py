import matplotlib.pyplot as plt

from Library import *
def training(model,TrainDataSet, ValDataSet,cb):
    print("\n------------Your Model Has Been Started Training-------------\n")
    model_training = model.fit_generator(
        TrainDataSet,
        steps_per_epoch=16,
        epochs=50,
        verbose=1,
        callbacks=cb,
        validation_data=ValDataSet,
        validation_steps=16,
    )
    print("ModelTrained = ",model_training)

    trained_history = model_training.history
    key = trained_history.keys()
    print(f"Key:{key}")

    '''Plotting Trained Model Accuracy'''
    plt.plot(trained_history['accuracy'])
    plt.plot(trained_history['val_accuracy'], c="red")
    plt.title("Accuracy vs Validation-Accuracy")
    plt.show()

    '''Plotting Training Model Loss'''
    plt.plot(trained_history['loss'])
    plt.plot(trained_history['val_loss'], c="red")
    plt.title("Loss vs Validation-Loss")
    plt.show()

