# OmniTrade

![alt text](https://github.com/justinmulli/OmniTrade/blob/master/readme%20images/logo.png)

## High level description ##

###### In a nutshell: ######
* Stock price prediction using machine learning
* Generates buy or sell signals using a neural network
* Automated model building and tuning (hyperparameter search)
* Automated stock price data updates
* Flexible for easy experimentation, ex. for different combinations of stocks, different prediction time periods and return thresholds (ex. predict a buy signal if price goes up 1% 2 hours in the future), etc.

![alt text](https://github.com/justinmulli/OmniTrade/blob/master/readme%20images/Basic%20overview%20flowchart.jpg)

###### More Details: ######

Omnitrade is a project aimed at applying machine learning to stock price prediction. In a nutshell, it uses TensorFlow and Keras to build a Neural Network which generates buy/sell signals when the price is predicted to move up/down beyond a given threshold in a given time period (ex. 1% increase in 2 hours).

Both the hyperparameters for the neural network and the parameters used for feature engineering such as the tickers of the stocks/companies for which to make predictions and the prediction length can be changed easily in the designated files. The code allows for substantial flexibility in this regard, as explained below in the functionality section. OmniTrade requires only minimal effort to use. When input data files and your personal API key are in place, one import and one function call will suffice to launch any of the processes, whether it's one round of training the neural network, automated hyperparameter optimization, or launching a live feed of predictions.

In terms of automation, data updates can be done automatically in the background through the dataupdate script which grabs data from the alphavantage API to enrich the existing data files using the latest data. Furthermore, the models can be automatically optimized using bayesian optimization to conduct an automatic hyperparameter search for a given range of parameters and automatically replace the saved models and hyperparameters when a new top performing model is found. This way the live predictions can be up and running for trading while the model is being optimized in the background, so predictions will improve when a better performing model is found and replaces the existing model immediately.

## Functionality/Getting Started ##

**Quick start**

The absolute minimum requirements to make use of OmniTrade are:

1. Python 3+ (ideally 3.7)
2. Python package requirements as detailed in the requirements.txt file
3. One data file for one stock with at least a few years of at least 5 minute frequency data, formatted in line with this stock price data for Microsoft ranging from 2007 till 2019. This file will suffice to meet the data requirement. https://drive.google.com/open?id=1zO4M-0DAHRDfk2M13fRFkXVdPFVdEhIr. **Save this file under OmniTrade/Data/Minute**.

Once the above requirements are met, simply `import omnitrade` and run any of the following:
1. `omnitrade.live_feed()` to launch the live prediction feed (be sure to keep data up to date and save an API key as described below).
2. `omnitrade.train()` to launch one round of model training
3. `omnitrade.bayes_optimization()` to launch automated hyperparameter optimization.

###### The above functions will run with the default settings unless they are changed in the `params.py` file or the `Optimal Hyperparameters.txt` file, but further instructions should be read before changing settings from the default. ######

**Further details for use**

###### Modifying parameters ######

Most parameters are saved in the `params.py` file, with the exception of most neural network hyperparameters which are saved in the `Optimal Hyperparameters.txt` file as this file is replaced automatically when a new optimal set of hyperparameters is found. Parameters can be changed manually in the respective files. Be sure to follow the requirements detailed in the files.

OmniTrade was designed to make the bulk testing of parameters flexible by importing omni_params from the params.py file (`import omni_params from params as my_params` for example), reassigning the desired parameters in your imported parameter dictionary and calling `omni_trade.train(omni_params = my_params)`. By doing this you could run a loop to train a neural network on historical stock price data for a variety of companies/stocks, prediction lengths, minimum return for a buy signal, hindsight (how many days/weeks etc. the model should look back to make the prediction) and more. This allows the testing of theories or running of experiments in bulk with just a few lines of code.

When a model is trained with new parameters, it is automatically saved with a name corresponding to the ID matching the corresponding specifications and accuracy saved in `Model Index.csv`. This way OmniTrade can automatically check the model index to identify the correct model ID for a given set of parameters, and grab the right model when making live predictions. When a new maximum accuracy is reached the model is replaced by the new optimal model.

###### Keeping data up to date (required for using live predictions) ######

OmniTrade was intended to generate buy and sell signals for NYSE stocks, and is set up to automatically update all saved stock price files once the market closes. In order to enable this functionality, simply do the following:

1. Grab a free API key from alphavantage.co and store it as a text string variable `my_alpha_vantage_key = '123yourkey123'` in a file named `keys.py` in the main OmniTrade folder. 

2. Make sure minimum requirements are met as described at the beginning of the Getting Started section.

3. Run the dataupdate script. The simplest way to do this is to call `python3 dataupdate.py` from the OmniTrade directory in the terminal. The lowest maintenance way of keeping your data upt o date is to leave it running in the background or on a server/cloud instance.
  -If the market is open, it will wait until the market is closed to update, and once complete it will wait a full trading session (i.e. open and close) before the next update.
  -This must be executed at least once every 7 days as the alphavantage minute data only stretches back one week, so if this time is exceeded there will be no overlap and the script will skip the stock to avoid a data gap.
  
###### How it works ######

*Feature engineering overview*

OmniTrade is designed to take csv stock price files for individual stocks, do some initial pre-processing and feature engineering in pandas, then convert the dataframe to a tensorflow dataset and do some final processing before feeding the tensorflow dataset into a neural network. This is done through a class designed to deal with pandas and tensorflow dataset processing, as well as a function specifically created for feature engineering with stock price data.

Data is split up as rolling time windows so that each training example consists of the stock market data in that time window, and a binary classifier indicating whether or not the stock should be bought at that time. The classifier/signal is generated by checking whether the stock increases past a certain threshold. The respective parameters are described below and can be found and changed in the params.py file.

* Time window size is defined by the hindsight and hindsight_interval parameters. Ex the default is 256 hindsight and a hindsight_interval of 5T (5 minutes), equating to approx. 21.3 hours or 3.3 open market days.
* Pediction length is determined by the foresight parameter, and the default is 32, which amounts to approx 2.7 hours given the default interval of 5 minutes.
* Minimum return required to generate a buy signal is buy_threshold. ex. the default is 0.01 meaning a price increase of 1% after the prediction length of approx 2.7 hours will generate a buy signal. In this case the neural network will learn to output a buy signal for a predicted price increase of 1% after 2.7 hours.
* Consecutive buy signals are filtered out of the training set to prevent overfitting. This is because, for example, if a buy signal is generated for a given training example with 5 minute interval data, the next training example and a number of consecutive ones will also generate buy signals as they are only 5 minutes apart. The problem is that they are very similar and the model will hence develop a bias and overfit due to a high number of very similar training examples.
* As most training examples are not cases with buy signals, the training data and validation data is balanced so that there are an equal number of training examples for buying and not buying, otherwise the model could achieve a high accuracy by never predicting buy signals, as most of the unbalanced training examples don't generate buy signals.
* The feature engineering process also includes scaling of the input variables using scikit learn.
 
 *Building the Model and automated optimization*
 
The model is automatically built based on the main input parameters and the hyperparameters indicated in the respective files. The main layers of the model are built based on a matrix which is created in a loop in the `define_layers` function, with the hyperparameters as input, which is then fed into the main training function `train_keras_model`

Some of the parameters and hyperparameters are defined automatically to suit model needs (ex. activation functions for binary vs multiclass classification).

The `bayes_optimization` function uses the pbounds dictionary from the `params.py` file as the search space and uses bayesian statistics to drill down on promising areas in the search space identified during initial random tests, treating the optimization process as a multi-armed bandit problem.

After training, if the accuracy beats the previous best for the same model specifications (hindsight, foresight, stock tickers etc.), or a model with the same specifications has not yet been saved, the model is saved and model details are added to `Model_Index.csv`

For more information contact justus.mulli@icloud.com
 
 


