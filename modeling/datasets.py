from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob


class REDDDataset(Dataset):
    def __init__(self, args, type_path="train"):
        # Getting the device info
        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.args = args
        self.type_path = type_path
        self.window_segment_size = self.args.window_segment_size

        self.file_name_map = {}
        self.data_map = {}
        self.index_map = {}
        self.total_num_samples = 0
        self._build()

    def __len__(self):
        return self.total_num_samples

    def __getitem__(self, index):
        for f_idx, val in self.index_map.items():
            if index < val["num_samples"]:
                cur_idx = val["start_idx"] + index
                cur_start = cur_idx - self.window_segment_size
                cur_end = cur_idx + self.window_segment_size + 1
                cur_input = self.data_map[f_idx][cur_start:cur_end, [1, 2]].astype(np.float32)
                cur_output = self.data_map[f_idx][cur_idx, [3]].astype(np.float32)

                return {
                    "inputs": torch.tensor(cur_input),
                    "targets": torch.tensor(cur_output)
                }
            else:
                index -= val["num_samples"]

        return None

    def _build(self):
        self.data_folder = os.path.join(self.args.data_dir,
                                        self.args.appliance,
                                        self.type_path)
        print("Reading data from", self.data_folder)
        data_files_list = glob.glob(self.data_folder + "/*.csv")
        data_files_list = list(sorted(data_files_list, key=lambda x: x.split("/")[-1].split(".")[0], reverse=False))

        # Load the data
        cols = ["time_stamp", "mains_1", "mains_2", "output"]
        for idx, cur_file in enumerate(data_files_list):
            cur_df = pd.read_csv(cur_file, index_col=0)
            cur_df["time_stamp"] = cur_df.index
            cur_df = cur_df[cols]
            self.data_map[idx] = cur_df.to_numpy()
            self.file_name_map[idx] = cur_file

        # Build the index map
        for idx in self.data_map.keys():
            num_records = len(self.data_map[idx])
            s_idx = self.window_segment_size
            e_idx = num_records - 1 - self.window_segment_size
            num_samples = e_idx - s_idx + 1

            if num_samples <= 0:
                continue

            self.total_num_samples += num_samples
            self.index_map[idx] = {
                "start_idx": s_idx,
                "end_idx": e_idx,
                "num_samples": num_samples
            }

        print("Total number of samples =", self.total_num_samples)


class REDDForecastDataset(Dataset):
    def __init__(self, args, type_path="train"):
        # Getting the device info
        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.args = args
        self.type_path = type_path
        self.window_segment_size = args.window_segment_size
        self.num_ts_predict = args.num_ts_predict

        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "inputs": torch.tensor(self.inputs[index]).reshape(-1, 1),
            "targets": torch.tensor(self.targets[index])
        }

    def _build(self):
        self.file_path = os.path.join(self.args.data_dir, self.args.appliance, "h" + str(self.args.house_idx) + "_" + self.type_path + ".csv")
        print("Reading data from", self.file_path)

        # Load the data
        df = pd.read_csv(self.file_path, index_col=0)
        output_list = list(df["output"])

        i = 0
        for idx in tqdm(range(len(output_list))):
            start = i - self.window_segment_size
            end = i

            if start < 0 or end + self.num_ts_predict > len(output_list):
                i += 1
                continue

            x_vals = output_list[start: end]
            y_vals = output_list[end: end+self.num_ts_predict]

            self.inputs.append(x_vals)
            self.targets.append(y_vals)
            i += 1

        print("Total number of samples =", len(self.inputs))