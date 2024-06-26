from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def validate_arima(dir, product_name):
    # load and prepare datasets
    model_path = f'{dir}/{product_name}/model/arima.pkl'
    bias_path = f'{dir}/{product_name}/model/bias.npy'
    train_data = f'{dir}/{product_name}/data/cleaned_data.csv'
    validate_data = f'{dir}/{product_name}/data/cleaned_data_valid.csv'
	
    dataset = read_csv(train_data, header=0, index_col=0, parse_dates=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    months_in_year = 12
    validation = read_csv(validate_data, header=0, index_col=0, parse_dates=True)
    y = validation.values.astype('float32')
    # load model
    model_fit = ARIMAResults.load(model_path)
    bias = numpy.load(bias_path)
	
    # make first prediction
    predictions = list()
    yhat = float(model_fit.forecast()[0])
    yhat = bias + inverse_difference(history, yhat, months_in_year)
    predictions.append(yhat)
    history.append(y[0])
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
    # rolling forecasts
    for i in range(1, len(y)):
        # difference data
        months_in_year = 12
        diff = difference(history, months_in_year)
        # predict
        model = ARIMA(diff, order=(0,0,1))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        yhat = bias + inverse_difference(history, yhat, months_in_year)
        if(yhat[0] < 0):
            yhat[0] = 0
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    # report performance
    rmse = sqrt(mean_squared_error(y, predictions))
    print('RMSE: %.3f' % rmse)
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    pyplot.savefig(f'{dir}/{product_name}/visual/arima_validation.png')