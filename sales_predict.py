import argparse
import process_data  # Assuming process_data.py contains the necessary functions
import forecast  # Assuming forecast.py contains the necessary functions
import visualize  # Assuming visualize.py contains the necessary functions
import predict
import validate


def main():
    parser = argparse.ArgumentParser(description='Sales Forecaster')

    parser.add_argument('-c', '--clean', action='store_true', \
                        help='Generate a cleaned dataset for a given product')
    parser.add_argument('-p', '--product', type=str, \
                        help='Specify the product')
    parser.add_argument('-d', '--dir', type=str, \
                        help='Specify location of directory you want to \
                            perform the operation on')
    parser.add_argument('-m', '--model', type=str, \
                        help='Select a prediction model')
    parser.add_argument('--predict', type=str, \
                        help='Make predictions based on model argument selected')
    parser.add_argument('--validate', type=str, \
                        help='Validate model based on validation data')
    parser.add_argument('-P', '--p_val', type=int, \
                        help='Specify the p value for the ARIMA model')
    parser.add_argument('-Q', '--q_val', type=int, \
                        help='Specify the q value for the ARIMA model')
    parser.add_argument('-D', '--d_val', type=int, \
                        help='Specify the d value for the ARIMA model')
    parser.add_argument('-s', '--deseasonalize', action='store_true', \
                        help='Deseasonalize data')
    parser.add_argument('-v', '--visualize', type=str, \
                        help='Generate visuals for the model\'s performance')

    args = parser.parse_args()
    
    if args.predict and args.dir and args.product:
        predict.predict(args.dir, args.product, args.predict)

    if args.validate and args.dir and args.product:
        validate.validate_arima(args.dir, args.product)
    
    if args.clean and args.product and args.dir:
        process_data.clean_data(args.dir, args.product)  

    if args.deseasonalize and args.dir and args.product:
        process_data.deseasonalize_data(args.dir, args.product)
    
    if args.visualize and args.dir and args.product:
        visualize.display(args.dir, args.product, args.visualize)

    
    if args.model == "arima" and args.dir and args.product:
        p_val = args.p_val if args.p_val is not None else 7
        q_val = args.q_val if args.q_val is not None else 3
        d_val = args.d_val if args.d_val is not None else 7

        forecast.grid_search_arima(args.dir, args.product, p_val, q_val, d_val)

    if args.model == "persistence" and args.dir and args.product:
        forecast.persistence(args.dir, args.product)


if __name__ == "__main__":
    main()