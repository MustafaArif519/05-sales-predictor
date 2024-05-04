import warnings
from pandas import read_csv, DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from math import sqrt
import numpy
import os


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')

	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]

	# make predictions
	predictions = list()
	for t in range(len(test)):
		# difference data
		months_in_year = 12
		diff = difference(history, months_in_year)
		model = ARIMA(diff, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		yhat = inverse_difference(history, yhat, months_in_year)
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse, model_fit, test, predictions


# calculate residuals of current best model and return bias
def residual_analysis(dir, product_name, test, predictions):
	residuals = [test[i]-predictions[i] for i in range(len(test))]
	residuals = DataFrame(residuals)
	pyplot.figure()
	pyplot.subplot(211)
	residuals.hist(ax=pyplot.gca())
	pyplot.subplot(212)
	residuals.plot(kind='kde', ax=pyplot.gca())
	pyplot.savefig(f'{dir}/{product_name}/visual/residuals.png')
	return residuals.mean().iloc[0]


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values, dir, product_name):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse, model_fit, test, predictions = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						# Create a new folder for the product in case it doesn't exist
						os.makedirs(f"{dir}/{product_name}/model", exist_ok=True)
						best_score, best_cfg = rmse, order
						
						file_path = f"{dir}/{product_name}/model/arima.pkl"
						model_fit.save(file_path)
						residual_mean = residual_analysis(dir, product_name, test, predictions)
						with open(f"{dir}/{product_name}/model/hyperparameters.txt", 'w') as f:
							f.write(f"P Value: {p}\n")
							f.write(f"D Value: {d}\n")
							f.write(f"Q Value: {q}\n")
							f.write(f"RMSE: {rmse}\n")
							f.write(f"Residual Mean Value: {residual_mean}\n")

					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					print("errored")
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

def grid_search_arima(dir, product_name, pU = 7, dU = 3, qU = 7):
    # load dataset
    series = read_csv(f"{dir}/{product_name}/data/stationary_data.csv", \
					  header=None, index_col=0, parse_dates=True)
    # evaluate parameters
    p_values = range(0, pU)
    d_values = range(0, dU)
    q_values = range(0, qU)
    warnings.filterwarnings("ignore")
    # print(series.values.shape, "series shape")
    evaluate_models(series.values, p_values, d_values, q_values, dir, product_name)
    

def persistence(dir, product_name):
      # load data
	series = read_csv(f"{dir}/{product_name}/data/cleaned_data.csv", \
                   header=0, index_col=0, parse_dates=True)
	# prepare data
	X = series.values
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	# walk-forward validation
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		# predict using persistence technique
			# we predict simply by using the last observation
		yhat = history[-1]
		predictions.append(yhat)
		# observation
		obs = test[i]
		history.append(obs)
		# print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
  
	# report performance
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	print('RMSE: %.3f' % rmse)