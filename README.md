## Project Abstract
- The project is a sales predictor that utilizes time forecasting to predict the sales of a product given a particular date
- The dataset currently consists of grocery sales for regions in Tamil Nadu, India, Adidas Sales in North America, and alcohol sales from an unnamed company
	- Perform ARIMA modeling on beer product in the alcohol dataset to see a good example of annual seasonal sales trends and how de-seasonalization allows for significantly better predicting
- For this reason the project structure has been segmented off into folders based on the data on which the predictions/data analysis is being done to ensure organized structure and segmentation based on which data is being processed
## Data Format
- Create a folder containing the csv containing the product whose sales you would like to analyze
- Rename the dataset to `original_dataset.csv` and rename the folder to something meaningful, ideally describing the theme or the vendor of the dataset
- The csv can have variable column and description length. However the dataset must have the product column labeled as `product`, date column labeled as `date` and following the format `%md/%dd/%YYYY` , and sales column named as `sales` for extraction and augmentation of the dataset to work
- The sales data is assumed to be in normal decimal format (ex 10 equals 10 items sold) and later converted to sales in the 100s for easier visualization and interpretation 
## Command Line Interface
- The following is the expected format and use cases for the command line interface
```
Sales Forecaster

options:
  -h, --help            show this help message and exit
  -c, --clean           Generate a cleaned dataset for a given product
  -p PRODUCT, --product PRODUCT
                        Specify the product you want to perform the operation on
  -d DIR, --dir DIR     Specify location of directory you want to perform the operation on
  -t TRAIN, --train TRAIN
                        Select a prediction model tor train "arima" or "persistence"
  --predict PREDICT     Make predictions based on model argument selected
  --predict_date PREDICT_DATE
                        Specify the date you want to make a prediction for
  --validate VALIDATE   Validate model based on validation data
  -P P_VAL, --p_val P_VAL
                        Specify the p value for the ARIMA model
  -Q Q_VAL, --q_val Q_VAL
                        Specify the q value for the ARIMA model
  -D D_VAL, --d_val D_VAL
                        Specify the d value for the ARIMA model
  -s, --deseasonalize   Deseasonalize data basesd on anual seasonality
  -v VISUALIZE, --visualize VISUALIZE
                        Generate visuals for the cleaned dataset. These include "line plot", "summary statistics", and
                        "density plot"
```
## Forecasting and Prediction Method
- The program uses time forecasting method by leveraging the Autoregressive Integrated Moving Average (ARIMA) model alongside depersonalization of trends of the original data to predict the sales of a product on a given date
- This model works best when it has a few years worth of data to predict seasonal trends as well. In the event seasonal data is not available, the program will simply use a moving average model to predict the sales 
## Visualization Tools
- A couple visualization tools are available for the user to make assessments about the data
- Line plot for seeing how sales changes over time
- Summary statistics for general statistical summary of sales data such as 
- Density plot for viewing the density of sales numbers and what numbers are the most frequent in the dataset
- All visuals are presented with sales in the 100s 