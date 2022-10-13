import kerastuner as kt
import omnitrade
import sys
sys.path.append('/Users/justusmulli/Projects/mltoolkit')
import tuning



tuner = tuning.cv_bayesian_tuning(omnitrade.builder)


def new_model():
    with open('models/Optimal Hyperparameters.txt') as hyperparameters:
        omni_params['hyperparams'] = eval(hyperparameters.read())

    omni_params['layers'] = define_layers(omni_params['hyperparams'])
    return build_model(omni_params)


if not folds:
    model = new_model()
    training_data = data_prep(omni_params)
    history = model.fit(training_data.tf_training_dataset, epochs=omni_params['epochs'],
                        validation_data=training_data.tf_validation_dataset, use_multiprocessing=True,
                        workers=16)
else:
    accuracy = []
    for i in range(folds):
        model = new_model()
        training_data = data_prep(omni_params, fold=[i, folds])
        history = model.fit(training_data.tf_training_dataset, epochs=omni_params['epochs'],
                            validation_data=training_data.tf_validation_dataset, use_multiprocessing=True,
                            workers=16)
        accuracy.append(max(history.history['val_accuracy']))
    print(np.mean(accuracy))