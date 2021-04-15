import argparse
from datasets import REDDDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import LSTMAttn
import os, json, math
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def computeMetrics(y_true, y_pred, loss_type):
    metrics_dict = {}
    if loss_type == "regression":
        metrics_dict["rmse"] = math.sqrt(mean_squared_error(y_true, y_pred))
    elif loss_type == "classification":
        metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)
        metrics_dict["precision"] = precision_score(y_true, y_pred)
        metrics_dict["recall"] = recall_score(y_true, y_pred)
        metrics_dict["f1"] = f1_score(y_true, y_pred)
    else:
        metrics_dict = None

    return metrics_dict

def collate_with_padding(batch):
    sorted_batch = batch
    inputs_list = [cur_row["inputs"] for cur_row in sorted_batch]
    inputs_lengths = torch.FloatTensor([len(cur_input) for cur_input in inputs_list])
    targets_list = torch.FloatTensor([cur_row["targets"] for cur_row in sorted_batch])
    inputs_padded_list = nn.utils.rnn.pad_sequence(inputs_list, batch_first=True)

    result_batch = {
        "inputs": inputs_padded_list,
        "inputs_lengths": inputs_lengths,
        "targets": targets_list,
    }
    return result_batch

def train(args):
    # Create the dataset and data loaders
    train_dataset = REDDDataset(args, type_path="train")
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_with_padding)

    val_dataset = REDDDataset(args, type_path="test")
    val_data_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_with_padding)

    # Create the model
    model = LSTMAttn(feature_size=2, hidden_size=args.lstm_hidden_size, output_size=1, bidirectional=True)

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
    if args.loss_type == "regression":
        loss_criterion = nn.MSELoss()
    elif args.loss_type == "classification":
        loss_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Loss type argument is invalid")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    for epoch in range(args.num_epochs):

        # Train
        model.train()
        avg_training_loss = 0.0
        num_train_steps = 1
        for batch_idx, cur_batch in enumerate(tqdm(train_data_loader, desc="Train Epoch {}".format(epoch))):
            inputs = cur_batch["inputs"].to(device)
            inputs_lengths = cur_batch["inputs_lengths"].to(device)
            targets = cur_batch["targets"].to(device)

            if args.loss_type == "regression":
                targets = targets.view((-1, 1))
            elif args.loss_type == "classification":
                targets = targets.type(torch.LongTensor).to(device)

            predictions, attn_weight = model(inputs, inputs_lengths, batch_size=inputs.shape[0])

            loss = loss_criterion(predictions, targets)
            avg_training_loss += loss.item()
            num_train_steps += 1

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(model, 1e-1)
            optimizer.step()
            break

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
                inputs_lengths = cur_batch["inputs_lengths"].to(device)
                targets = cur_batch["targets"].to(device)

                with torch.no_grad():
                    predictions, attn_weight = model(inputs, inputs_lengths, batch_size=inputs.shape[0])

                if args.loss_type == "regression":
                    targets = targets.view((-1, 1))
                    loss = loss_criterion(predictions, targets)
                    y_pred += list(predictions.view(targets.size()).data.cpu().tolist())
                    y_true += list(targets.cpu().tolist())
                elif args.loss_type == "classification":
                    targets = targets.type(torch.LongTensor).to(device)
                    loss = loss_criterion(predictions, targets)
                    y_pred += list(torch.max(predictions, 1)[1].view(targets.size()).data.cpu().tolist())
                    y_true += list(targets.cpu().tolist())
                else:
                    raise ValueError("Loss type argument is invalid")

                avg_val_loss += loss.item()
                num_val_steps += 1
                break

            avg_val_loss /= num_val_steps
            metrics = computeMetrics(y_true, y_pred, loss_type=args.loss_type)
            print("Metrics :", metrics)

            if args.loss_type == "regression":
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
                if best_val_loss is None:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_metrics = metrics
                    best_predictions = y_pred
                    best_model_state_dict = model.state_dict().copy()
                    best_opt_state_dict = optimizer.state_dict().copy()
                elif avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
    pred_result = []
    if args.do_eval:
        print("Best epoch = ", best_epoch)
        print("Best metrics = ", best_metrics)
        best_metrics["stop_epoch"] = best_epoch + 1

        for cur_pred in best_predictions:
            if args.loss_type == "regression":
                pred_result.append(cur_pred[0])
            else:
                pred_result.append(cur_pred)

    return best_metrics, pred_result

def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters:
    parser.add_argument("--model_type", type=str, default="LSTM",
                        help="Model type. Options = [LSTM, CNN, WAVENET]")
    parser.add_argument("--loss_type", type=str, default="regression",
                        help="Loss type. Options = [regression, classification]")
    parser.add_argument("--appliance", type=str, default="refrigerator",
                        help="Appliance. Options = [refrigerator, dishwaser]")
    parser.add_argument("--window_segment_size", type=int, default=3,
                        help="Window Segment Size.")
    parser.add_argument("--mode", type=str, default="train",
                        help="Whether to do training or testing.")

    parser.add_argument("--data_dir", type=str, default="../data/redd_processed/",
                        help="Data directory which contains training and testing data")
    parser.add_argument("--output_dir", type=str, default="../outputs/",
                        help="Output directory to store the model checkpoints")

    # Optional parameters:
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Train Batch Size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Eval Batch Size")
    parser.add_argument("--lstm_hidden_size", type=int, default=50,
                        help="Hidden Size of LSTM")

    args = parser.parse_known_args()[0]
    args.do_eval = True
    print(args)

    if args.mode == "train":
        best_metrics, best_pred = train(args)

        # Store the best predictions
        df_pred = pd.DataFrame(data=best_pred, columns=["predicted_output"])
        pred_results_path = os.path.join(args.output_dir, "best_predictions.csv")
        df_pred.to_csv(pred_results_path, index=False)

        # Store the best metrics
        best_metrics_path = os.path.join(args.output_dir, "best_metrics.json")
        with open(best_metrics_path, "w") as f:
            json.dump(best_metrics, f)
    else:
        test(args)


