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