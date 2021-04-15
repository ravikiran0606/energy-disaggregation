from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
from tqdm import tqdm

class REDDDataset(Dataset):
    def __init__(self, args, type_path="train"):
        # Getting the device info
        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.inputs = []
        self.targets = []
        self.args = args
        self.type_path = type_path
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "inputs": self.inputs[index],
            "targets": self.targets[index]
        }

    def _build(self):
        self.file_path = os.path.join(self.args.data_dir, "window_{}".format(self.args.window_segment_size),
                                      self.args.appliance, "{}.csv".format(self.type_path))
        print("Reading data from ", self.file_path)
        cur_df = pd.read_csv(self.file_path)
        cur_df_cols = cur_df.columns
        mains1_cols = []
        mains2_cols = []
        for cur_col in cur_df_cols:
            if "mains_1" in cur_col:
                mains1_cols.append(cur_col)
            elif "mains_2" in cur_col:
                mains2_cols.append(cur_col)

        mains1 = cur_df[mains1_cols].values.tolist()
        mains2 = cur_df[mains2_cols].values.tolist()
        outputs = list(cur_df["output"])

        num_records = len(mains1)
        for idx in tqdm(range(num_records)):
            m1 = mains1[idx]
            m2 = mains2[idx]
            cur_inp = []
            for m1_val, m2_val in zip(m1, m2):
                cur_inp.append([m1_val, m2_val])
            self.inputs.append(torch.tensor(cur_inp))

            out = outputs[idx]
            self.targets.append(torch.tensor(out))









