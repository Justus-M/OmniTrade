from tensorflow.keras.models import load_model
import os
import pandas as pd
import ModelTraining

def Predict(p, Tensor):

    p['TargetTickers'].sort()

    if os.path.exists('models/%s-%d-foresight.h5' %(str(p['TargetTickers']), p['foresight'])):
        print('Loading pre-trained model.')
        Model = load_model('models/%s-%d-foresight.h5' %(str(p['TargetTickers']), p['foresight']))
    else:
        print('No pre-trained model matching requirements. Train model for prediction?')
        answer = None
        while answer not in ('y', 'n'):
            answer = input('y/n: ')
            if answer == 'y':
                Model, history = ModelTraining.Train(p)
            elif answer == 'n':
                return
            else:
                print("Please enter y or n.")

    Predictions = Model.predict(Tensor)
    Predictions = pd.DataFrame(Predictions, index=p['Data']['DFrame'].index)
    headers = [a + b for a, b in zip(p['TargetTickers'], ([' long'] * len(p['TargetTickers'])))]
    headers.append('none')
    Predictions.columns = headers

    return Predictions

