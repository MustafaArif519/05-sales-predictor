import pandas as pd
from datetime import datetime,  timedelta
import numpy as np


def calculate_velocity(dir, product_name, velocity_interval, date):
    csv_path = f'{dir}/{product_name}/data/cleaned_data.csv'
    df = pd.read_csv(csv_path, header=0, index_col=0, parse_dates=True)
    # print(date)
    start_date = date - timedelta(days=velocity_interval)
    end_date = date
    # print(start_date)
    # print(end_date)
    sales_data = []

    # Iterate over the DataFrame rows
    for idx, row in df.iterrows():
        # Check if the index is between start_date and end_date
        current_date = str(idx)
        current_date = datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S")
        if start_date <= current_date <= end_date:
            # If it is, append the row to the sales_data list
            sales_data.append(row['sales'])

    
    # print(sales_data)
    if len(sales_data) == 0:
        return None
    sales_velocity = (sum(sales_data) / len(sales_data))
    return sales_velocity

def calculate_average_velocity(dir, product_name, velocity_intervals, date):
    average_velocity = 0
    date = datetime.strptime(date, "%m/%d/%Y")
    for interval in velocity_intervals:
        interval = int(interval)
        velocity = calculate_velocity(dir, product_name, interval, date)
        if velocity is not None:
            average_velocity += velocity
    if(average_velocity == 0):
        print("No sales data found for the specified interval(s) and date")
        return average_velocity
    aggregated_velocity = average_velocity / len(velocity_intervals)
    print(f"Average velocity: {aggregated_velocity}")

    velocity_path = f'{dir}/{product_name}/model/velocity.npy'
    np.save(velocity_path, [aggregated_velocity])

    return aggregated_velocity

def calculate_day_inventory_left(dir, product_name, date, buffer=True):
    csv_path = f'{dir}/{product_name}/data/cleaned_data.csv'
    df = pd.read_csv(csv_path, header=0, index_col=0, parse_dates=True)
    date = datetime.strptime(date, "%m/%d/%Y")

    starting_inventory = 80000
    inventory = starting_inventory
     # Iterate over the DataFrame rows
    for idx, row in df.iterrows():
        # Check if the index is between start_date and end_date
        current_date = str(idx)
        current_date = datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S")
        if current_date <= date:
            # If it is, append the row to the sales_data list
            # print(row)
            inventory -= row['sales']
    aggregated_velocity = np.load(f'{dir}/{product_name}/model/velocity.npy')


    days_left = (inventory) / int(aggregated_velocity)
    stock_out_date = date + timedelta(days=days_left)
    buffered_days_left = days_left - 30
    buffered_date = date + timedelta(days=buffered_days_left)

    print(f"Days of inventory left: {days_left}")
    print(f"Stock out date: {stock_out_date}")
    if buffer:
        print(f"Days until only buffer remaining: {buffered_days_left}")
        print(f"Stock out date with buffer: {buffered_date}")
    return (inventory) 

