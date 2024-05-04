# load finalized model and make a prediction
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMAResults
import numpy
from datetime import datetime
import pandas as pd


# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def clean_data(date):
    date = date.replace('/', '-')
    
    month, day, year = date.split('-')
    
    # Add leading zeros to day and month if necessary
    day = int(day)
    if day <= 9:
        day = 1
    elif day <= 19:
        day = 10
    elif day <= 29:
        day = 20
    else:
        day = 30
    day = str(day)
    day = day.zfill(2)
    month = month.zfill(2)

    # Reconstruct the date string
    date_str = '-'.join([year, month, day])

    # Parse the date string
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return date.strftime('%Y-%m-%d')

def predict(dir, product_name, date):
    model_path = f'{dir}/{product_name}/model/arima.pkl'
    bias_path = f'{dir}/{product_name}/model/bias.npy'
    csv_path = f'{dir}/{product_name}/data/stationary_data.csv'
    date = clean_data(date)

    series = read_csv(csv_path, header=0, index_col=0, parse_dates=True)
    months_in_year = 12
    model_fit = ARIMAResults.load(model_path)
    bias = numpy.load(bias_path)

    # Assuming `series` is your time series data
    last_date_in_data = series.index[-1]

    # Assuming `target_date` is the date you want to forecast for
    target_date = pd.to_datetime(date)  # convert `date` to a pandas Timestamp, if it's not already

    # Calculate the number of steps to forecast
    forecast_steps = (target_date - last_date_in_data).days

    # Make the forecast
    forecast = model_fit.forecast(steps=forecast_steps)

    # Get the last forecasted value (i.e., the forecast for `target_date`)
    yhat = float(forecast[-1])
    yhat = bias + inverse_difference(series.values, yhat, months_in_year)
    if(yhat[0] < 0):
            yhat[0] = 0
    print('Predicted: %.3f' % yhat)