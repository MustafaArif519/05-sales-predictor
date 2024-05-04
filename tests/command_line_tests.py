import unittest
from unittest.mock import patch
import sales_predict  # Assuming sales_predict.py is in the same directory
import argparse  # Add this line to import the argparse module

class TestCommandLineArguments(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args',
           return_value=argparse.Namespace(clean=True, dir='alcohol', product='beer'))
    def test_clean(self, mock_args):
        with self.assertRaises(SystemExit):
            sales_predict.main()

    @patch('argparse.ArgumentParser.parse_args',
           return_value=argparse.Namespace(train='arima', dir='alcohol', product='beer', p_val=1, q_val=1, d_val=1))
    def test_train_arima(self, mock_args):
        with self.assertRaises(SystemExit):
            sales_predict.main()

    @patch('argparse.ArgumentParser.parse_args',
           return_value=argparse.Namespace(predict='arima', dir='alcohol', product='beer'))
    def test_predict_arima(self, mock_args):
        with self.assertRaises(SystemExit):
            sales_predict.main()

    @patch('argparse.ArgumentParser.parse_args',
           return_value=argparse.Namespace(validate='arima', dir='alcohol', product='beer'))
    def test_validate_arima(self, mock_args):
        with self.assertRaises(SystemExit):
            sales_predict.main()

    @patch('argparse.ArgumentParser.parse_args',
           return_value=argparse.Namespace(deseasonalize=True, dir='alcohol', product='beer'))
    def test_deseasonalize(self, mock_args):
        with self.assertRaises(SystemExit):
            sales_predict.main()

    @patch('argparse.ArgumentParser.parse_args',
           return_value=argparse.Namespace(visualize=True, dir='alcohol', product='beer'))
    def test_visualize(self, mock_args):
        with self.assertRaises(SystemExit):
            sales_predict.main()

if __name__ == '__main__':
    unittest.main()