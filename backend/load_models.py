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

    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    # Load LSTM model for dishwasher
    dishwasher_lstm_model = LSTMAttn(config["lstm_feature_size"], config["lstm_hidden_size"], config["lstm_output_size"])
    dishwasher_checkpoint = torch.load(config["dishwasher_lstm_model"], map_location=device)
    dishwasher_lstm_model.load_state_dict(dishwasher_checkpoint["model"])

    # Load LSTM model for Refrigerator
    refrigerator_lstm_model = LSTMAttn(config["lstm_feature_size"], config["lstm_hidden_size"], config["lstm_output_size"])
    refrigerator_checkpoint = torch.load(config["refrigerator_lstm_model"], map_location=device)
    refrigerator_lstm_model.load_state_dict(refrigerator_checkpoint["model"])
    
    return dishwasher_lstm_model, refrigerator_lstm_model