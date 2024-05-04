## [Article Link](https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/)
## Goals of Tutorial
- How to confirm Python environment and clearly define a time series forecasting problem to solve
- How to create a test harness for evaluating models, develop a baseline forecast, and better understand the problem
- How to develop an autoregressive integrated moving average model, save it to file, and later load it to make predictions for new time steps
## Virtual Environment
- Requirements:
	- `SciPy`
	- `NumPy`
	- `Matplotlib`
	- `Pandas`
	- `scikit-learn`
	- `statsmodels`
- The following script below will ensure if the dependencies are met:
```Python
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
```
## Problem Description
- The problem being solved is a sales predictor given a particular product
## Test Harness
- Must have a way to investigate the data and evaluate candidate models
- This means splitting the data into a validation and testing dataset
## Validation Dataset
- Observe the dataset is not current, which means we cannot easily collect data to validate the model
- The tutorial is going to treat the last year of data as the test data and everything prior as the validation
- Following code is from tutorial for segmenting data into two csv files, one for training and one for validation
```Python
# separate out a validation dataset
from pandas import read_csv

series = read_csv('./data/champagne.csv', header=0, index_col=0, parse_dates=True)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('./data/dataset.csv', header=False)
validation.to_csv('./data/validation.csv', header=False)
```
- Note: `squeeze` parameter for reading the csv is not recognized, so had to remove it
## Model Evaluation
- Model evaluation will only be performed on data that was partitioned to `dataset.csv`
- Two components to model evaluation
	- Performance measure
	- Testing strategy
## Performance Measure
- The performance of the model will be evaluated using the root mean squared error (RMSE). This metric gives more weight to predictions that are severely wrong 
- The RMSE can be calculated using the function `mean_squared_error()` from the `sklearn.metrics` library
	- This function calculates the mean squared error between the expected values (the test set) and the predicted values (what the model generated)
```Python
from sklearn.metrics import mean_squared_error
from math import sqrt
...
test = ...
predictions = ...
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
```
## Test Strategy
- Candidate models will be evaluated using *walk forward validation*
	- *Walk Forward Validation* is training a model on a subset of data, making predictions on the next data point, then incorporating that data point into the training set before making the next prediction
- For the context of this problem, this is how we will use walk forward validation:
	- The first 50% of the dataset will be held back to train the model
	- The remaining 50% of the dataset will be iterated and test the model
	- For each step in the test dataset:
	    - A model will be trained
	    - A one-step prediction made and the prediction stored for later evaluation
	    - The actual observation from the test dataset will be added to the training dataset for the next iteration
	- The predictions made during the iteration of the test dataset will be evaluated and an RMSE score reported
- First we need to split the data using `NumPy` and `Python` code by splitting the `dataset` into `train` and `test` sets directly 
```Python
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```
- The following code demonstrates how we are going to use the split data to perform walk forward validation
```Python
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	yhat = ...
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
```
## Persistence
- First step before getting into data analysis is establishing a baseline of performance 
- The purpose of this would be for evaluating models during training and a performance baseline at which more elaborate predictive models can be compared
- The baseline prediction for time series forecasting is called naive forecast or *persistence*
	- *Persistence* is where the observation from the previous time step is used as the prediction for the observation at the next time step
- We can directly integrate this into the code we have written so far
```Python
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
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
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
```
- The output of running the above code on the `dataset.csv` should follow as below
```Python
>Predicted=5428.000, Expected=8314
>Predicted=8314.000, Expected=10651
>Predicted=10651.000, Expected=3633
>Predicted=3633.000, Expected=4292
>Predicted=4292.000, Expected=4154
>Predicted=4154.000, Expected=4121
>Predicted=4121.000, Expected=4647
>Predicted=4647.000, Expected=4753
>Predicted=4753.000, Expected=3965
>Predicted=3965.000, Expected=1723
>Predicted=1723.000, Expected=5048
>Predicted=5048.000, Expected=6922
>Predicted=6922.000, Expected=9858
>Predicted=9858.000, Expected=11331
>Predicted=11331.000, Expected=4016
>Predicted=4016.000, Expected=3957
>Predicted=3957.000, Expected=4510
>Predicted=4510.000, Expected=4276
>Predicted=4276.000, Expected=4968
>Predicted=4968.000, Expected=4677
>Predicted=4677.000, Expected=3523
>Predicted=3523.000, Expected=1821
>Predicted=1821.000, Expected=5222
>Predicted=5222.000, Expected=6872
>Predicted=6872.000, Expected=10803
>Predicted=10803.000, Expected=13916
>Predicted=13916.000, Expected=2639
>Predicted=2639.000, Expected=2899
>Predicted=2899.000, Expected=3370
>Predicted=3370.000, Expected=3740
>Predicted=3740.000, Expected=2927
>Predicted=2927.000, Expected=3986
>Predicted=3986.000, Expected=4217
>Predicted=4217.000, Expected=1738
>Predicted=1738.000, Expected=5221
>Predicted=5221.000, Expected=6424
>Predicted=6424.000, Expected=9842
>Predicted=9842.000, Expected=13076
>Predicted=13076.000, Expected=3934
>Predicted=3934.000, Expected=3162
>Predicted=3162.000, Expected=4286
>Predicted=4286.000, Expected=4676
>Predicted=4676.000, Expected=5010
>Predicted=5010.000, Expected=4874
>Predicted=4874.000, Expected=4633
>Predicted=4633.000, Expected=1659
>Predicted=1659.000, Expected=5951
RMSE: 3186.501
```
## Data Analysis
- We can use diagrams and line plots to visualize the effectiveness of our predictions
- The five kinds of summary statistics we will be investigating are
	- Summary statistics
	- Line plot
	- Seasonal line plots
	- Density plots
	- Box and whisker plot
## Summary Statistics
- Standard summary statistics regarding the data and achieved using the code below:
```Python
from pandas import read_csv
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
print(series.describe())
```

## Line Plot
- The line plot can provide invaluable insight as to the trends of sales for a particular product over time
- Below is the code to generate the line plot:
```Python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```
- Some relationships that can be observed are
	- Overall trend (comparing peaks or averages at certain points)
	- Seasonality of peaks 
	- The trend in the seasonal peaks (growing or decreasing over time)
	- Outliers if evident
	- Seasonality vs non-stationary nature of the product
## Seasonal Line Plots
- Can confirm the seasonality nature of line plots by providing the season interval
- Below is the code that generates such seasonal plots
```Python
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
groups = series['1964':'1970'].groupby(Grouper(freq='A'))
years = DataFrame()
pyplot.figure()
i = 1
n_groups = len(groups)
for name, group in groups:
	pyplot.subplot((n_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()
```
- The plots (in this example) show a dip right around August then almost an immediate increase until December
	- This observation helps with explicit season modeling later
## Density Plot
- Reviewing the plots of the density of observations can provide more insight on the structure of the data
- The following code creates a histogram and density plot of the observations by stripping away temporal structure
```Python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure(1)
pyplot.subplot(211)
series.hist()
pyplot.subplot(212)
series.plot(kind='kde')
pyplot.show()
```
- Some observations that can be made include
	- Gaussian vs non-Gaussian distribution 
	- Skewed direction (long tails suggest an exponential distribution)
- Lends support to power transforms of the data prior to modeling
## Box and Whisker Plots
- Can group monthly observations by year to get sense of trends regarding the sales of the product over a long course of time as well as how much variation there is in sales within a year
- From the prior data analysis we can already make conclusions on how the median/mean is changing over the years but the box plot allows us to see how the rest of the distribution (quartile wise) also changes from year to year
- Following code generates said box-whisker plot for analysis
```Python
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
groups = series['1964':'1970'].groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.boxplot()
pyplot.show()
```
- Some observations that can be made regarding the plot are
	- The change in the spread of data (box length)
	- Outliers trends
## ARIMA Models
- We will now design an Autoregressive Integrated Moving Average (ARIMA) model to solve this problem
- Design of this model includes both manual and automated components
- There are 3 steps in designing said model
	- Manually configuring the ARIMA
	- Automatically configuring the ARIMA
	- Reviewing the residual errors
## Manually Configuring ARIMA
- The ARIMA(p, q, d) model is traditionally configured manually
- Time series data makes the assumption we are working with a stationary time series (no seasonal trends) which is almost never the case, so we need to make the adjustments
- To convert the data from non-stationary to stationary, we need a de-seasonalized version of the data. This is achieved by placing points that occur at the same time in their respective seasons together (for example in our case we eliminate the year variable and group same months together)
- The code below generates the de-seasonalized version of the data based on the determined yearly nature of the season
```Python
from pandas import read_csv
from pandas import Series
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

series = read_csv('./data/dataset.csv', header=None, index_col=0, parse_dates=True)
X = series.values
X = X.astype('float32')
# difference data
months_in_year = 12
stationary = difference(X, months_in_year)
stationary.index = series.index[months_in_year:]
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
# Assuming that 'stationary' is a DataFrame with a single column
stationary = stationary.apply(lambda x: x[0])

# save
stationary.to_csv('./data/stationary.csv', header=False)
stationary = stationary.apply(pd.to_numeric, errors='coerce')
# plot
stationary.plot()
pyplot.savefig('./visualization/stationary_plot.png')

```
- Expected output of the above code is as follows:
```Python
ADF Statistic: -7.134898
p-value: 0.000000
Critical Values:
	5%: -2.898
	1%: -3.515
	10%: -2.586
```
- Now we need to select the value for Autoregression (AR) and Moving Average (MA), `p` and `q` respectively
- The following code selects `p` and `q` based on the stationary data
```Python
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
series = read_csv('stationary.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure()
pyplot.subplot(211)
plot_acf(series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(series, ax=pyplot.gca())
pyplot.show()
```
- Optimal `p`, `q` and `d` can also be selected through automatically testing variables within a range although this does take some time to compute
```Python
# grid search ARIMA parameters for time series
import warnings
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy

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
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# load dataset
series = read_csv('./data/dataset.csv', header=None, index_col=0, parse_dates=True)
# evaluate parameters
p_values = range(0, 7)
d_values = range(0, 3)
q_values = range(0, 7)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```
## Review Residual Errors
- It is good practice to review residual forecast errrors
- A good model will have errors normally distributed with mean at 0
- To view the current distribution of the residuals, run the following code:
```Python
# summarize ARIMA forecast residuals
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

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

# load data
series = read_csv('./data/dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.savefig('./visualization/residuals.png')

```
- To correct the distribution of residuals, you can adjust the distribution by the means like this:
```Python
# plots of residual errors of bias corrected forecasts
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt

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

# load data
series = read_csv('./data/dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
bias = -204.270035
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.savefig('./visualization/bias_crorrected_residuals.png')
```
- Lastly you can check the time series for any autocorrelation of the residuals which if present would suggest the model has more opportunity to model the temporal structure of the data leading to more accurate predictions
- Code below tests for this:
```Python
# ACF and PACF plots of residual errors of bias corrected forecasts
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

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

# load data
series = read_csv('./data/dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.savefig('./visualization/acf_pacf_residuals.png')
```
## Finalizing Model
- Fitting the final model involves fitting on the entire dataset, not just the training data we originally split from the data
- Following code saves the model with the hyperparameters decided as well as the bias detected
```Python
# save finalized model
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# load data
series = read_csv('./data/dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32')
# difference data
months_in_year = 12
diff = difference(X, months_in_year)
# fit model
model = ARIMA(diff, order=(0,0,1))
model_fit = model.fit()
# bias constant, could be calculated from in-sample mean residual
bias = -240
# save model
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])
```
## Using Model to Make Prediction
- The following code demonstrates how to apply the model to make a prediction on a value in the validation dataset:
```Python
# load finalized model and make a prediction
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMAResults
import numpy

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

series = read_csv('./data/dataset.csv', header=None, index_col=0, parse_dates=True)
months_in_year = 12
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(series.values, yhat, months_in_year)
print('Predicted: %.3f' % yhat)
```
## Validating the Model
- We have reserved the final year of our data to validate the final model we ended up creating
- We will now use the model we generated to make predictions for all the days in the last year of data 
- There are two approaches to how we might test the data
	- Load the model and test it for each of the days in the last year. The performance will significantly degrade after the first 2 months
	- Load the model and use it in a loading forecast manner, updating the model each step. This almost always results in a better performance than the previous idea
- Below is the code implementing the rolling forecast prediction idea:
```Python
# load and evaluate the finalized model on the validation dataset
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

# load and prepare datasets
dataset = read_csv('./data/dataset.csv', header=None, index_col=0, parse_dates=True)
X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 12
validation = read_csv('./data/validation.csv', header=None, index_col=0, parse_dates=True)
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
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
pyplot.savefig('validation_plot.png')
```
- Looking at `validation_plot.png`, we should see that the model predictions vs reality looks great!