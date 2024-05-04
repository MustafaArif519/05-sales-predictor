# load finalized model and make a prediction
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMAResults
import numpy

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def predict(dir, product_name):
    model_path = f'{dir}/{product_name}/model/arima.pkl'
    bias_path = f'{dir}/{product_name}/model/bias.npy'
    csv_path = f'{dir}/{product_name}/data/cleaned_data.csv'
    series = read_csv(csv_path, header=0, index_col=0, parse_dates=True)
    months_in_year = 12
    model_fit = ARIMAResults.load(model_path)
    bias = numpy.load(bias_path)
    yhat = float(model_fit.forecast()[0])
    yhat = bias + inverse_difference(series.values, yhat, months_in_year)
    print('Predicted: %.3f' % yhat)