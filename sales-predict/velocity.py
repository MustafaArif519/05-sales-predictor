import pandas as pd
from datetime import datetime,  timedelta

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
    sales_velocity = (sum(sales_data) / len(sales_data)) * 0.5
    return sales_velocity

def calculate_average_velocity(dir, product_name, velocity_intervals, date):
    average_velocity = 0
    date = datetime.strptime(date, "%m/%d/%Y")
    for interval in velocity_intervals:
        interval = int(interval)
        velocity = calculate_velocity(dir, product_name, interval, date)
        if velocity is not None:
            average_velocity += velocity
    print(f"Average velocity: {average_velocity / len(velocity_intervals)}")
    return average_velocity / len(velocity_intervals)