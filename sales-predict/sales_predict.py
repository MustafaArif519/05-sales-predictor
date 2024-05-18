import argparse
import process_data
import forecast
import predict
import validate
import visualize
import velocity


def main():
    parser = argparse.ArgumentParser(description='Sales Forecaster')

    parser.add_argument('-c', '--clean', action='store_true', \
                        help='Generate a cleaned dataset for a given product')
    parser.add_argument('-p', '--product', type=str, \
                        help='Specify the product you want to perform the operation on')
    parser.add_argument('-d', '--dir', type=str, \
                        help='Specify location of directory you want to \
                            perform the operation on')
    parser.add_argument('-t', '--train', type=str, \
                        help='Select a prediction model tor train "arima" or "persistence"')
    parser.add_argument('--predict', type=str, \
                        help='Make predictions based on model argument selected')
    parser.add_argument('--predict_date', type=str, \
                        help='Specify the date you want to make a prediction for')
    parser.add_argument('--validate', type=str, \
                        help='Validate model based on validation data')
    parser.add_argument('-P', '--p_val', type=int, \
                        help='Specify the p value for the ARIMA model')
    parser.add_argument('-Q', '--q_val', type=int, \
                        help='Specify the q value for the ARIMA model')
    parser.add_argument('-D', '--d_val', type=int, \
                        help='Specify the d value for the ARIMA model')
    parser.add_argument('-s', '--deseasonalize',action='store_true',  \
                        help='Deseasonalize data basesd on anual seasonality')
    parser.add_argument('-v', '--visualize', type=str, \
                        help='Generate visuals for the cleaned dataset. These include \
                            "line plot", "summary statistics", and "density plot"')
    parser.add_argument('--velocity', nargs=2, \
                    help='Calculate the velocity of a product over a given interval\
                        and date. The first argument is the interval and the second is the date"')
    parser.add_argument('--inventory', type=str, \
                    help='Calculate the stock out and buffered date of a product\
                        based on the velocity calculated previously after a given date"')
    

    args = parser.parse_args()
    products = args.product.split(',')

    if args.clean:
        if not args.dir or not args.product:
            print("Please specify the directory and product using \
                  the -d and -p flags respectively")
            return
        for product in products:
            process_data.clean_data(args.dir, product)
            print("Data successfully cleaned for product: ", product)
            print("\n")

    elif args.inventory:
        if not args.dir or not args.product:
            print("Please specify the directory and product using \
                  the -d and -p flags respectively")
            return
        if not args.inventory:
            print("Please specify the date to calculate inventory")
            return
        for product in products:
            velocity.calculate_day_inventory_left(args.dir, product, args.inventory)
            print("Inventory successfully calculated for product: ", product)
            print("\n")

    elif args.velocity:
        if not args.dir or not args.product:
            print("Please specify the directory and product using \
                  the -d and -p flags respectively")
            return
        if not args.velocity[0] or not args.velocity[1]:
            print("Please specify the interval and date to calculate velocity")
            return
        args = parser.parse_args()
        interval = args.velocity[0].split(',')
        date = args.velocity[1] 
        for product in products:
            velocity.calculate_average_velocity(args.dir, product, interval, date)
            print("Velocity successfully calculated for product: ", product)
            print("\n")
        
    elif args.train:
        if not args.dir or not args.product:
            print("Please specify the directory and product")
            return
        if args.train == "arima":
            if args.p_val and args.q_val and args.d_val:
                forecast.train_arima(args.dir, args.product, args.p_val, args.q_val, args.d_val)
            else:
                forecast.grid_search_arima(args.dir, args.product)
        elif args.train == "persistence":
            forecast.persistence(args.dir, args.product)
        else:
            print("Please specify a valid model to train")

    
    elif args.predict:
        if not args.dir or not args.product:
            print("Please specify the directory and product using \
                  the -d and -p flags respectively")
            return
        if not args.predict_date:
            print("Please specify a date to make a prediction for")
            return
        if args.predict == "arima":
            predict.predict_arima(args.dir, args.product, args.predict_date)
        elif args.predict == "persistence":
            predict.predict_persistence(args.dir, args.product, args.predict_date)
        else:
            print("Please specify a valid model to use for prediction")

    elif args.validate:
        if not args.dir or not args.product:
            print("Please specify the directory and product using \
                  the -d and -p flags respectively")
            return
        if args.validate == "arima":
            validate.validate_arima(args.dir, args.product)
        else:
            print("Validation only available for ARIMA model")


    elif args.deseasonalize:
        if not args.dir or not args.product:
            print("Please specify the directory and product using \
                  the -d and -p flags respectively")
            return
        process_data.deseasonalize_data(args.dir, args.product)
    
    elif args.visualize:
        if not args.dir or not args.product:
            print("Please specify the directory and product using \
                  the -d and -p flags respectively")
            return
        visualize.display(args.dir, args.product, args.visualize)
    else:
        print("Please specify an operation to perform, type -h for help")
        return



if __name__ == "__main__":
    main()