from pandas import read_csv
from pandas import Series
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
import globals
import os

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



# read the dataset
series = pd.read_csv(globals.CLEANED_DATASET, header=0)



# filter the dataset based on the product only
filtered_series = series[(series['product'] == globals.PRODUCT)]

# extract only the sales values
filtered_series = filtered_series['sales']

X = filtered_series.values
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

# Assuming that 'stationary' is a DataFrame with a single column
stationary = stationary.apply(lambda x: x[0])

# save
dir_name, base = os.path.split(globals.CLEANED_DATASET)
base, ext = os.path.splitext(base)
globals.DESEASONALIZED_DATASET = f"{dir_name}/{base}/{globals.PRODUCT}/deseasonalized_data{ext}"
stationary.to_csv(globals.DESEASONALIZED_DATASET, header=False)
stationary = stationary.apply(pd.to_numeric, errors='coerce')

# plot
stationary.plot()
pyplot.savefig(f"{dir_name}/{base}/{globals.PRODUCT}/stationary_plot.png")