import pandas as pd
import numpy as np
from redd_processing import REDDMLData
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from argparse import ArgumentParser
import pickle
import json
import os
import math

def generate_data(args):
    train_data = REDDMLData(args.train_data, args.window_segment_size)
    train_data_arr, train_data_out = train_data.generate_window_data()
    print('Shape of Training data and output labels is {} and {}'.format(train_data_arr.shape, 
                                                                         train_data_out.shape))

    test_data = REDDMLData(args.test_data, args.window_segment_size)
    test_data_arr, test_data_out = test_data.generate_window_data()
    print('Shape of Test data and output labels is {} and {}'.format(test_data_arr.shape, 
                                                                         test_data_out.shape))
    
    return train_data_arr, train_data_out, test_data_arr, test_data_out

def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics['rmse'] = math.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['sae'] = np.abs(np.sum(y_pred) - np.sum(y_true))/np.sum(y_true)
    return metrics

def fit_rf_model(args, train_data_arr, train_data_out, test_data_arr, test_data_out):
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features='log2')
    model.fit(train_data_arr, train_data_out)
    test_pred = model.predict(test_data_arr)
    metrics = compute_metrics(test_data_out, test_pred)
    print(metrics)
    model_name = 'rf_' + args.appliance + '.pkl'
    model_save_path = os.path.join(args.output_dir, model_name)
    pickle.dump(model, open(model_save_path, 'wb'))
    json.dump(metrics, open(os.path.join(args.output_dir, 'metrics_rf.json'), 'w'))

def fit_lr_model(args, train_data_arr, train_data_out, test_data_arr, test_data_out):
    model = LinearRegression()
    model.fit(train_data_arr, train_data_out)
    test_pred = model.predict(test_data_arr)
    metrics = compute_metrics(test_data_out, test_pred)
    print(metrics)
    model_name = 'lr_' + args.appliance + '.pkl'
    model_save_path = os.path.join(args.output_dir, model_name)
    pickle.dump(model, open(model_save_path, 'wb'))
    json.dump(metrics, open(os.path.join(args.output_dir, 'metrics_lr.json'), 'w'))
    
def generate_arguments():
    parser = ArgumentParser()
    parser.add_argument('--train_data', type=str, default=None, required=True,
                       help='Path for the training data')
    parser.add_argument('--test_data', type=str, default=None, required=True,
                       help='Path for the training data')
    parser.add_argument('--window_segment_size', type=int, default=None, required=True,
                       help='Window size')
    parser.add_argument('--output_dir', type=str, default=None, required=True,
                        help='Path where the output model should be stored')
    parser.add_argument('--appliance', type=str, default=None, required=True,
                        help='Name of appliance')
    parser.add_argument('--model_type', type=str, default='rf', required=True,
                       help='model type Linear Regression(lr)/ Random Forest(rf)')
    return parser

if __name__ == '__main__':
    parser = generate_arguments()
    args = parser.parse_args()
    train_data_arr, train_data_out, test_data_arr, test_data_out = generate_data(args)
    print('Training the model')
    if args.model_type == 'rf':
        fit_rf_model(args, train_data_arr, train_data_out, test_data_arr, test_data_out)
    elif args.model_type == 'lr':
        fit_lr_model(args, train_data_arr, train_data_out, test_data_arr, test_data_out)