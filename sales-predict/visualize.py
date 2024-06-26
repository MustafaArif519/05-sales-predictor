from pandas import read_csv
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import Grouper
import pandas as pd
from matplotlib import pyplot
from pandas import DataFrame, Grouper, Series
import os


def display(dir, product_name, display_type):
    input_csv = f"{dir}/{product_name}/data/cleaned_data.csv"
    series = read_csv(input_csv, header=0, index_col=0, parse_dates=True)

    # Create a new folder for the product in case it doesn't exist
    os.makedirs(f"{dir}/{product_name}/visual", exist_ok=True)

    if display_type == 'summary statistics':
        with open(f"{dir}/{product_name}/visual/summary_stats_clean.txt", 'w') as f:
            f.write(str(series.describe()))
        print("Summary Statistics Saved Successfully")


    elif display_type == 'line plot':
        series.plot()
        pyplot.savefig(f"{dir}/{product_name}/visual/line_plot_clean.png")
        print("Line Plot Saved Successfully")

    elif display_type == 'seasonal line plot':
        # Load the CSV file
        df = pd.read_csv(input_csv)
        # Convert the date column to datetime format
        df['date'] = pd.to_datetime(df['date'])
        # Set the date column as the index
        df.set_index('date', inplace=True)
        # Group the data by year
        groups = df['sales'].groupby(Grouper(freq='YE'))
        # Create a new DataFrame to hold the grouped data
        years = pd.DataFrame()
        # Iterate over the grouped data
        for name, group in groups:

            years[name.year] = group.values
        # Plot the new DataFrame
        years.plot(subplots=True, legend=False)
        plt.show()
        pyplot.savefig(f"{dir}/{product_name}/visual/seasonal_line_plot_clean.png")
        print("Seasonal Line Plot Saved Successfully")

    elif display_type == 'density plot':
        series.plot(kind='kde')
        pyplot.savefig(f"{dir}/{product_name}/visual/density_plot_clean.png")
        print("Density Plot Saved Successfully")

    elif display_type == 'box plot':
        groups = series['2000':'2024'].groupby(Grouper(freq='YE'))
        years = DataFrame()
        for name, group in groups:
            group_series = Series(group.values.flatten(), index=group.index)
            years[name.year] = group_series
        years.boxplot()
        pyplot.savefig(f"{dir}/{product_name}/visual/box_plot_clean.png")
        print("Box Plot Saved Successfully")

    else:
        print("Invalid display type")