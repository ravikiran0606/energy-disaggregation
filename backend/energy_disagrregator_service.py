import json
import os
import sys
import pandas as pd
from flask import Flask
from flask import request
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from argparse import Namespace
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime, timedelta

dirname = os.path.dirname(os.path.abspath(__file__))
modeling_path = os.path.join(dirname, "../")
sys.path.append(modeling_path)

from modeling.datasets import REDDDataset
from modeling.models import LSTMAttn, CNN
from load_models import load_models
from forecast_model import ForecastModel

app = Flask(__name__)
CORS(app)

config = json.load(open('config.json'))

dishwasher_lstm_model, dishwasher_cnn_model, \
    refrigerator_lstm_model, refrigerator_cnn_model = load_models()

dishwaser_scaler = pickle.load(open(config['dishwasher_normalization_factor'], 'rb'))
refrigerator_scaler = pickle.load(open(config['refrigerator_normalization_factor'], 'rb'))

@app.route('/')
def main():
    return "Energy Disaggregation Service"

@app.route('/disaggregate', methods=['POST'])
def disaggregate():
    appliance = ["dishwasher", "refrigerator"]
    cols = ['mains_1', 'mains_2']
    appliance_predicted = defaultdict(list)
    json_output = dict()
    file_path = request.files['file']
    prediction_model = request.args.get('model', 'lstm')
    df = pd.read_csv(file_path)
    dishwaser_df = dishwaser_scaler.transform(df[cols])
    refrigerator_df = refrigerator_scaler.transform(df[cols])

    for app in appliance:
        s_idx = config['{}_{}_window'.format(app, prediction_model)]
        e_idx = len(df) - 1 - config['{}_{}_window'.format(app, prediction_model)]
        new_df = df.iloc[s_idx:e_idx + 1]
        print("Predicting values for {} using {} model".format(app, prediction_model))
        key = app + '_batch_size'
        batch_size = config[key]
        
        if prediction_model == 'lstm':
            if app == 'dishwasher':
                args = Namespace(window_segment_size=config['dishwasher_lstm_window'])
                dataset = REDDDataset(args, type_path='infer', df=dishwaser_df)
                infer_dataloader = DataLoader(dataset, batch_size=batch_size, 
                                              collate_fn=collate_with_padding)
                y_pred = predict(dishwasher_lstm_model, infer_dataloader, prediction_model)
            elif app == 'refrigerator':
                args = Namespace(window_segment_size=config['refrigerator_lstm_window'])
                dataset = REDDDataset(args, type_path='infer', df=refrigerator_df)
                infer_dataloader = DataLoader(dataset, batch_size=batch_size, 
                                              collate_fn=collate_with_padding)
                y_pred = predict(refrigerator_lstm_model, infer_dataloader, prediction_model)
            appliance_predicted[app] = y_pred
            
        elif prediction_model == 'cnn':
            if app == 'dishwasher':
                args = Namespace(window_segment_size=config['dishwasher_cnn_window'])
                dataset = REDDDataset(args, type_path='infer', df=dishwaser_df)
                infer_dataloader = DataLoader(dataset, batch_size=batch_size,
                                            collate_fn=collate_with_padding)
                y_pred = predict(dishwasher_cnn_model, infer_dataloader, prediction_model)
            elif app == 'refrigerator':
                args = Namespace(window_segment_size=config['refrigerator_cnn_window'])
                dataset = REDDDataset(args, type_path='infer', df=refrigerator_df)
                infer_dataloader = DataLoader(dataset, batch_size=batch_size,
                                            collate_fn=collate_with_padding)
                y_pred = predict(refrigerator_cnn_model, infer_dataloader, prediction_model)
            appliance_predicted[app] = y_pred

        predicted_df = pd.DataFrame(columns=['timestamp'])
        predicted_df['timestamp'] = new_df['timestamp']
        predicted_df['mains_1'] = new_df['mains_1']
        predicted_df['mains_2'] = new_df['mains_2']
        predicted_df['predictions'] = y_pred

        json_output[app] = predicted_df.to_dict('records')

    # for k, pred in appliance_predicted.items():
    #     output_column = k + '_predicted'
    #     new_df[output_column] = pred
    # remaining_watt_list = []
    # for i, row in new_df.iterrows():
    #     remaining_watt = ((row['mains_1'] + row['mains_2']) - (row['dishwasher_predicted'] + row['dishwasher_predicted']))
    #     remaining_watt_list.append(remaining_watt)
    # new_df['remaining'] = remaining_watt_list
    # new_df.to_csv('predicted_values.csv', index=False)

    return json_output

@app.route('/forecast', methods=['POST'])
def forecast():
    '''
    Gives hourly forecast
    A flag list is maintained. A 0 in the flag list means that 
    the corresponding output value is historical data and a 1 in the
    flag list means that the corresponding output value is the forecasted 
    value by the model. 
    '''
    forecast_mod = ForecastModel()
    file_path = request.files['file']
    time_period = int(request.args.get('time', '12'))
    df = pd.read_csv(file_path)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    timestamp_list = list(df['time_stamp'])
    history_data = list(df['output'])
    flag_list = [0 for _ in range(len(history_data))]
    for cur_time in range(time_period):
        cur_out = forecast_mod.trainARIMA(history_data)
        cur_time_obj = timestamp_list[-1] + timedelta(hours=1)
        history_data.append(cur_out)
        timestamp_list.append(cur_time_obj)
        flag_list.append(1)

    timestamp_list = list(map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'), timestamp_list))
    out_df = pd.DataFrame({'timestamp': timestamp_list, 'output': history_data, 'flag': flag_list})
    out_df.index = out_df.index.astype('str')
    out_df.to_csv('forecast_values.csv')
    out_dict = out_df.to_dict('records')
    return { 'data': out_dict }


def collate_with_padding(batch):
    sorted_batch = batch
    inputs_list = [cur_row["inputs"] for cur_row in sorted_batch if cur_row is not None]
    inputs_lengths = torch.tensor([len(cur_input) for cur_input in inputs_list])
    #targets_list = torch.FloatTensor([cur_row["targets"] for cur_row in sorted_batch if cur_row is not None])
    inputs_padded_list = nn.utils.rnn.pad_sequence(inputs_list, batch_first=True)

    result_batch = {
        "inputs": inputs_padded_list,
        "inputs_lengths": inputs_lengths,
    }
    return result_batch

def predict(model, data_loader, prediction_model):
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    model.eval()
    y_pred = []
    for batch_idx, cur_batch in enumerate(tqdm(data_loader)):
        inputs = cur_batch["inputs"].to(device)
        if prediction_model == 'lstm':
            inputs_lengths = cur_batch["inputs_lengths"]
            with torch.no_grad():
                predictions, attn_weight = model(inputs, inputs_lengths, batch_size=inputs.shape[0])
                pred = torch.squeeze(predictions)
                y_pred += list(pred.data.cpu().tolist())
        elif prediction_model == 'cnn':
            with torch.no_grad():
                predictions = model(inputs)
                pred = torch.squeeze(predictions)
                y_pred += list(pred.data.cpu().tolist())

    return y_pred


if __name__ == '__main__':
    app.run(threaded=True, host=config['host'], port=config['port'])