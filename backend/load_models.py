import json
import os
import sys
import torch

dirname = os.path.dirname(os.path.abspath(__file__))
modeling_path = os.path.join(dirname, "../")
sys.path.append(modeling_path)

from modeling.datasets import REDDDataset
from modeling.models import LSTMAttn, CNN

def load_models():
    config = json.load(open('config.json'))

    # Load LSTM model for 
    dishwasher_lstm_model = LSTMAttn(config["lstm_feature_size"], config["lstm_hidden_size"], config["lstm_output_size"])
    ref_lstm_model = LSTMAttn(config["lstm_feature_size"], config["lstm_hidden_size"], config["lstm_output_size"])
    dishwasher_checkpoint = torch.load(config["dishwasher_lstm_model"], map_location=torch.device("cpu"))
    dishwasher_lstm_model.load_state_dict(dishwasher_checkpoint["model"])

    return dishwasher_lstm_model