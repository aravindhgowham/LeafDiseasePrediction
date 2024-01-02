from DataGenerator import Image_Data_generator
from DisplayImage import PlotImage
from InstallingModel import DownloadingNeuronModel
from BuildModel import KerasModel
from CallBacks import early_stopping
from TrainingModel import training
from ModelAccuracy import LoadingSavedModel


def main():
    '''Preproccessing Given Dataset'''
    dictionary = Image_Data_generator()

    train_data = dictionary['train']
    validation_data = dictionary['Validation']
    test_image = dictionary['test_image']
    test_label = dictionary['test_label']

    '''Displaying given data set '''
    PlotImage(test_image, test_label)#Displaying Train Images

    '''Downloading Vgg19 Keras Model'''
    BaseModel = DownloadingNeuronModel()

    '''Create Our Model'''
    model = KerasModel(BaseModel)
    callback = early_stopping()

    '''Training the dataset to the model'''
    training(model,train_data, validation_data,callback)
    Accuracy = LoadingSavedModel(validation_data)
    print(f"Accuracy Of Your Model Is {Accuracy * 100}%")


if __name__ == '__main__':
    main()



