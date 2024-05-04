import pandas as pd
from datetime import datetime
import globals
from statsmodels.tsa.stattools import adfuller
import os
import fnmatch
from matplotlib import pyplot


def extract_and_filter(input_csv, product_name):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Filter the dataset based on the product only
    filtered_df = df[(df['product'] == product_name)]

    # Select the desired columns
    filtered_df = filtered_df[['date', 'sales']]

    # Standardize the dates in the 'Order Date' column
    filtered_df['date'] = filtered_df['date'].apply(standardize_date)


     # Sort the dataframe by date from oldest to most recent
    sorted_df = filtered_df.sort_values(by='date')


    return sorted_df


def standardize_date(date_str):
    try:
        # Split the date string into day, month, and year
         # Replace "/" with "-" in the date string
        date_str = date_str.replace('/', '-')

        month, day, year = date_str.split('-')

        # Add leading zeros to day and month if necessary
        day = day.zfill(2)
        month = month.zfill(2)

        # Reconstruct the date string
        date_str = '-'.join([year, month, day])

        # Parse the date string
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return "Invalid date format"
    return date.strftime('%Y-%m-%d')

def total_sales(df):
    # Group the dataset by date and calculate the sum of sales for each month
    df = df.groupby('date').sum()
    return df

def clean_data(dir, product_name):

    # Find the CSV file in the directory
    input_csv = dir + '/original_dataset.csv'

    # Obtained the filtered and cleaned dataset
    product_df = extract_and_filter(input_csv, product_name)

    product_df = total_sales(product_df)
    
    split_point = len(product_df) - (len(product_df) * 0.1)
    split_point = int(split_point)
    dataset, validation = product_df[0:split_point], product_df[split_point:]

    # Create a new folder for the product in case it doesn't exist
    os.makedirs(f"{dir}/{product_name}/data", exist_ok=True)

    # Write the new CSV file
    dataset.to_csv(f"{dir}/{product_name}/data/cleaned_data.csv", index=True)
    validation.to_csv(f"{dir}/{product_name}/data/cleaned_data_valid.csv", index=True)

    print(f"{product_name} Data Successfully Cleaned")
    
def difference(dataset, interval=1):
    diff = list()

    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def deseasonalize_data(base_dir, product_name):

    # read the dataset
    series = pd.read_csv(f"{base_dir}/{product_name}/data/cleaned_data.csv", \
                         header=0, index_col=0, parse_dates=True)

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


    # Assuming that 'stationary' is a DataFrame with a single column
    stationary = stationary.apply(lambda x: x[0])
    # save
    # stationary = stationary.apply(pd.to_numeric, errors='coerce')
    stationary.to_csv(f"{base_dir}/{product_name}/data/stationary_data.csv",\
                       header=False)
    # stationary = stationary.apply(pd.to_numeric, errors='coerce')
    # plot
    stationary.plot()

    # Create a new folder for the product in case it doesn't exist
    os.makedirs(f"{base_dir}/{product_name}/visual", exist_ok=True)
    pyplot.savefig(f"{base_dir}/{product_name}/visual/stationary_plot.png")

    print(f"{product_name} Cleaned Data Successfully Deseasonalized")