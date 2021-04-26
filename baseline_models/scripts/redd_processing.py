import pandas as pd
import glob
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import pickle

class REDDMLData():
    def __init__(self, data_dir, window_segment_size):
        self.data_dir = data_dir
        self.window_segment_size = window_segment_size
        self.data_map = {}
        self.file_name_map = {}
        self.index_map = {}
        self.total_num_samples = 0
        self.data_files_list = glob.glob(data_dir + '/*.csv')
        self.data_files_list = list(sorted(self.data_files_list, 
                                          key=lambda x: x.split("/")[-1].split(".")[0], reverse=False))
        self._gen_data_map()
        
    def _gen_data_map(self):
        #columns in the data
        cols = ["time_stamp", "mains_1", "mains_2", "output"]
        for idx, cur_file in enumerate(self.data_files_list):
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
            
    def generate_window_data(self, past_only=False):
        window_map = defaultdict(list)
        window_target = defaultdict(list)
        for idx, val in self.index_map.items():
            for index in range(val['num_samples']):
                if not(past_only):
                    cur_idx = val['start_idx'] + index
                    cur_start = cur_idx - self.window_segment_size
                    cur_end = cur_idx + self.window_segment_size + 1
                    cur_input = self.data_map[idx][cur_start:cur_end, [1, 2]].astype(np.float32)
                    cur_output = self.data_map[idx][cur_idx, [3]].astype(np.float32)
                    window_map[idx].append(np.reshape(cur_input, (cur_input.shape[0]*2)))
                    window_target[idx].append(cur_output)
                elif past_only:
                    cur_idx = val['start_idx'] + index
                    cur_start = cur_idx - self.window_segment_size
                    cur_end = cur_idx +  1
                    cur_input = self.data_map[idx][cur_start:cur_end, [1, 2]].astype(np.float32)
                    cur_output = self.data_map[idx][cur_idx, [3]].astype(np.float32)
                    window_map[idx].append(np.reshape(cur_input, (cur_input.shape[0]*2)))
                    window_target[idx].append(cur_output)
                
        for idx, val in window_map.items():
            window_map[idx] = np.array(val)
            window_target[idx] = np.array(window_target[idx])
        
        train_arr = np.concatenate(tuple([window_map[idx] for idx in window_map.keys()]))
        train_out = np.concatenate(tuple([window_target[idx] for idx in window_target.keys()]))
        print("Shape of training data and output array generated is {} and {}".format(train_arr.shape, train_out.shape))
        return train_arr, train_out
    
    def compute_normalization_factor(self, appliance):
        cols = ['mains_1', 'mains_2']
        new_df_list = []
        for file in glob.glob(self.data_dir + '*.csv'):
            df = pd.read_csv(file)[cols]
            new_df_list.append(df)
        new_df = pd.concat(new_df_list)
        #print(new_df.shape)
        scaler = StandardScaler()
        scaler.fit_transform(new_df)
        pickle.dump(scaler, open(appliance + '_norm_factor.pkl', 'wb'))
        scaler_load = pickle.load(open(appliance + '_norm_factor.pkl', 'rb'))
        print(scaler_load.mean_)

if __name__ == '__main__':
    data = REDDMLData('../../../data/redd_processed/original/raw/refrigerator/train/', 7)
    #t_arr, target_arr = data.generate_window_data()
    data.compute_normalization_factor('refrigerator')
    #print(t_arr.shape, target_arr.shape)