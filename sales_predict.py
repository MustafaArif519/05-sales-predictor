import argparse
import process_data  # Assuming process_data.py contains the necessary functions
import forecast  # Assuming forecast.py contains the necessary functions

def main():
    parser = argparse.ArgumentParser(description='Sales Forecaster')
    
    parser.add_argument('--clean', action='store_true', \
                        help='Generate a cleaned dataset for a given product')
    parser.add_argument('--product', type=str, help='Specify the product')
    parser.add_argument('--dir', type=str, help='Specify location of directory you\
                        want to perform the operation on')
    parser.add_argument('--model', type=str, help='Select a prediction model')
    parser.add_argument('--deseasonalize', action='store_true', help='Deseasonalize data')
    parser.add_argument('--visualize', action='store_true', \
                        help='Generate visuals for the model\'s performance')
    
    args = parser.parse_args()
    
    if args.clean and args.product and args.dir:
        process_data.clean_data(args.dir, args.product)  # Assuming clean_data is a function in process_data.py
    
    if args.deseasonalize and args.dir and args.product:
        process_data.deseasonalize_data(args.dir, args.product)
    
    if args.model == "arima" and args.dir and args.product:
        forecast.grid_search_arima(args.dir, args.product)

    if args.visualize and args.product:
        process_data.visualize_model(args.product)  # Assuming visualize_model is a function in process_data.py

if __name__ == "__main__":
    main()