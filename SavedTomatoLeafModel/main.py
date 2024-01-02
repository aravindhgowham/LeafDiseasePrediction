from Library import *
from Prediction import predict
from LoadModel import TrainedModel
from FinalOutput import FinalResult

def TrainedTomatoModel():
    start_time = time.time()
    Model = TrainedModel()
    input_image = 'Healthy.JPG'

    prediction_value = predict(input_image, Model)
    result = FinalResult(prediction_value)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"\nInput Image: {input_image}\n"
          f"Predict: {result}\n"
          f"Excecution Time: {elapsed_time}")

if __name__ == '__main__':
    TrainedTomatoModel()