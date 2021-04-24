import argparse
from datasets import REDDDataset, REDDForecastDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import LSTMAttn, CNN, LSTMForecast
import os, json, math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def computeMetrics(y_true, y_pred):
    metrics_dict = {}
    print(y_true[0])
    print(y_pred[0])
    metrics_dict["rmse"] = math.sqrt(mean_squared_error(y_true[:][0], y_pred[:][0]))
    metrics_dict["rmse_2"] = math.sqrt(mean_squared_error(y_true[:][1], y_pred[:][1]))
    metrics_dict["rmse_3"] = math.sqrt(mean_squared_error(y_true[:][2], y_pred[:][2]))
    return metrics_dict

def collate_with_padding(batch):
    sorted_batch = batch
    inputs_list = [cur_row["inputs"] for cur_row in sorted_batch]
    inputs_lengths = torch.tensor([len(cur_input) for cur_input in inputs_list])
    targets_list = torch.stack([cur_row["targets"] for cur_row in sorted_batch])
    inputs_padded_list = nn.utils.rnn.pad_sequence(inputs_list, batch_first=True)

    result_batch = {
        "inputs": inputs_padded_list,
        "inputs_lengths": inputs_lengths,
        "targets": targets_list,
    }

    return result_batch


def predict_k_steps(model, inputs, inputs_lengths, batch_size, num_ts=3):
    predictions_k_ts = []
    for ts in range(num_ts):
        with torch.no_grad():
            cur_pred = model(inputs, inputs_lengths, batch_size=batch_size)
            predictions_k_ts.append(cur_pred.squeeze())

        inputs = torch.cat([inputs.squeeze(), cur_pred], dim=1)
        inputs = inputs[:, 1:].unsqueeze(dim=2)

    predictions_k_ts = torch.stack(predictions_k_ts).T
    return predictions_k_ts


def train_lstm(args):
    # Create the dataset and data loaders
    train_dataset = REDDForecastDataset(args, type_path="train")
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_with_padding)

    val_dataset = REDDForecastDataset(args, type_path="test")
    val_data_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_with_padding)

    # Create the model
    model = LSTMForecast(feature_size=1, hidden_size=args.lstm_hidden_size, output_size=1, num_layers=args.lstm_num_layers)

    training_logs = []
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    best_model_state_dict = None
    best_opt_state_dict = None
    best_epoch = None
    best_val_loss = None
    best_metrics = None
    best_predictions = None

    model.to(device)
    loss_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    for epoch in range(args.num_epochs):

        # Train
        model.train()
        avg_training_loss = 0.0
        num_train_steps = 1
        for batch_idx, cur_batch in enumerate(tqdm(train_data_loader, desc="Train Epoch {}".format(epoch))):
            inputs = cur_batch["inputs"].to(device)
            inputs_lengths = cur_batch["inputs_lengths"]
            targets = cur_batch["targets"].to(device)

            predictions = model(inputs, inputs_lengths, batch_size=inputs.shape[0])
            targets = targets[:, 0]

            loss = loss_criterion(predictions, targets)
            avg_training_loss += loss.item()
            num_train_steps += 1

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(model, 1e-1)
            optimizer.step()

        avg_training_loss /= num_train_steps

        # Evaluate
        if args.do_eval:
            model.eval()
            y_true = []
            y_pred = []
            avg_val_loss = 0.0
            num_val_steps = 1
            for batch_idx, cur_batch in enumerate(tqdm(val_data_loader, desc="Val Epoch {}".format(epoch))):
                inputs = cur_batch["inputs"].to(device)
                inputs_lengths = cur_batch["inputs_lengths"]
                targets = cur_batch["targets"].to(device)

                predictions = predict_k_steps(model, inputs, inputs_lengths, inputs.shape[0], num_ts=args.num_ts_predict)
                loss = loss_criterion(predictions[:,0], targets[:,0])

                y_pred += list(predictions.data.cpu().tolist())
                y_true += list(targets.cpu().tolist())

                avg_val_loss += loss.item()
                num_val_steps += 1

            avg_val_loss /= num_val_steps
            metrics = computeMetrics(y_true, y_pred)
            print("Metrics :", metrics)

            # model_save_path = os.path.join(args.output_dir, "model_epoch={}.tar".format(epoch))
            # print("Saving model " + model_save_path)
            # torch.save({
            #     'iteration': epoch,
            #     'model': model.state_dict(),
            #     'opt': optimizer.state_dict(),
            # }, model_save_path)

            if best_val_loss is None:
                best_val_loss = metrics["rmse"]
                best_epoch = epoch
                best_metrics = metrics
                best_predictions = y_pred
                best_model_state_dict = model.state_dict().copy()
                best_opt_state_dict = optimizer.state_dict().copy()
            elif metrics["rmse"] < best_val_loss:
                best_val_loss = metrics["rmse"]
                best_epoch = epoch
                best_metrics = metrics
                best_predictions = y_pred
                best_model_state_dict = model.state_dict().copy()
                best_opt_state_dict = optimizer.state_dict().copy()
        else:
            avg_val_loss = None
            metrics = None

        # Log the training process
        training_logs.append({
            "epoch": epoch,
            "avg_train_loss": avg_training_loss,
            "avg_val_loss": avg_val_loss,
            "metrics": metrics
        })

    # Store the best model
    if args.do_eval:
        # Save the final best model:
        model_save_path = os.path.join(args.output_dir, 'best_model_epoch={}.tar'.format(best_epoch))
        print("Saving model " + model_save_path)
        torch.save({
            'iteration': best_epoch,
            'model': best_model_state_dict,
            'opt': best_opt_state_dict,
        }, model_save_path)

    # Store the training logs
    log_save_path = os.path.join(args.output_dir, "training_logs.json")
    with open(log_save_path, "w") as f:
        json.dump(training_logs, f)

    # Get best results
    if args.do_eval:
        print("Best epoch = ", best_epoch)
        print("Best metrics = ", best_metrics)
        best_metrics["stop_epoch"] = best_epoch + 1

    return best_metrics, best_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters:
    parser.add_argument("--appliance", type=str, default="dishwaser",
                        help="Appliance. Options = [refrigerator, dishwaser]")
    parser.add_argument("--house_idx", type=int, default=1,
                        help="Index of the house to consider.")
    parser.add_argument("--window_segment_size", type=int, default=5,
                        help="Window Segment Size.")
    parser.add_argument("--num_ts_predict", type=int, default=3,
                        help="Number of time steps to predict in future.")
    parser.add_argument("--mode", type=str, default="train",
                        help="Whether to do training or testing.")

    parser.add_argument("--data_dir", type=str, default="../data/redd_forecast_processed/",
                        help="Data directory which contains training and testing data")
    parser.add_argument("--output_dir", type=str, default="../outputs_forecast/",
                        help="Output directory to store the model checkpoints")

    # Optional parameters:
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Train Batch Size")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Eval Batch Size")
    parser.add_argument("--lstm_hidden_size", type=int, default=100,
                        help="Hidden Size of LSTM")
    parser.add_argument("--lstm_num_layers", type=int, default=1,
                        help="Number of layers of LSTM")

    args = parser.parse_known_args()[0]

    args.do_eval = True
    args.output_dir = os.path.join(args.output_dir, "window_{}".format(args.window_segment_size), args.appliance)
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode == "train":
        best_metrics, best_pred = train_lstm(args)

        # Store the best predictions
        df_pred = pd.DataFrame(data=best_pred, columns=["pred_1", "pred_2", "pred_3"])
        pred_results_path = os.path.join(args.output_dir, "best_predictions.csv")
        df_pred.to_csv(pred_results_path, index=False)

        # Store the best metrics
        best_metrics_path = os.path.join(args.output_dir, "best_metrics.json")
        with open(best_metrics_path, "w") as f:
            json.dump(best_metrics, f)


