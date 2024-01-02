import keras.losses

from Library import *

def KerasModel(Vgg19model, dropout_rate=0.5):

    FlatternLayer = Flatten()(
                Vgg19model.output)

    DropoutLayer = Dropout(
                rate=dropout_rate)(FlatternLayer)

    DenseLayer = Dense(
                units=10, #Data Set Classes
                activation='relu',
                kernel_regularizer=l2(0.01)
                )(DropoutLayer)#'3000'




    '''Creating Our Model'''
    model = Model(
                Vgg19model.input, DenseLayer) #print(layer_model.summary())

    model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model




