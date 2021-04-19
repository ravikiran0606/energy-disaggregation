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

dirname = os.path.dirname(os.path.abspath(__file__))
modeling_path = os.path.join(dirname, "../")
sys.path.append(modeling_path)

from modeling.datasets import REDDDataset
from modeling.models import LSTMAttn, CNN
from load_models import load_models

app = Flask(__name__)

config = json.load(open('config.json'))

dishwasher_lstm_model, refrigerator_lstm_model = load_models()

@app.route('/')
def main():
    return "Energy Disaggregation Service"

@app.route('/disaggregate', methods=['POST'])
def disaggregate():
    window_size = [3, 3]
    appliance = ["dishwasher", "refrigerator"]
    appliance_predicted = defaultdict(list)
    file_path = request.files['file']
    df = pd.read_csv(file_path)
    for app in appliance:
        print("Predicting values for {}".format(app))
        key = app + '_batch_size'
        batch_size = config[key]
        dataset = REDDDataset({}, df=df)
        infer_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_with_padding)
        if app == 'dishwasher':
            y_pred = predict(dishwasher_lstm_model, infer_dataloader)
        elif app == 'refrigerator':
            y_pred = predict(refrigerator_lstm_model, infer_dataloader)
        appliance_predicted[app] = y_pred
    
    for k, pred in appliance_predicted.items():
        output_column = k + '_predicted'
        df[output_column] = pred
    
    df.to_csv('predicted_values.csv', index=False)

    return appliance_predicted


def collate_with_padding(batch):
    sorted_batch = batch
    inputs_list = [cur_row["inputs"] for cur_row in sorted_batch]
    inputs_lengths = torch.tensor([len(cur_input) for cur_input in inputs_list])
    targets_list = torch.FloatTensor([cur_row["targets"] for cur_row in sorted_batch])
    inputs_padded_list = nn.utils.rnn.pad_sequence(inputs_list, batch_first=True)

    result_batch = {
        "inputs": inputs_padded_list,
        "inputs_lengths": inputs_lengths,
        "targets": targets_list,
    }
    return result_batch

def predict(model, data_loader):
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    model.eval()
    y_pred = []
    for batch_idx, cur_batch in enumerate(tqdm(data_loader)):
        inputs = cur_batch["inputs"].to(device)
        inputs_lengths = cur_batch["inputs_lengths"]
        with torch.no_grad():
            predictions, attn_weight = model(inputs, inputs_lengths, batch_size=inputs.shape[0])
            pred = torch.squeeze(predictions)
            y_pred += list(pred.data.cpu().tolist())
    
    return y_pred


if __name__ == '__main__':
    app.run(threaded=True, host=config['host'], port=config['port'])