import numpy as np
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import math
import typing

class ForecastModel:
    def __init__(self):
        pass
    
    def trainARIMA(self, history_data: list):
        cur_model = ARIMA(history_data, order=(10,1,0))
        cur_model_fit = cur_model.fit()
        cur_out = cur_model_fit.forecast()[0]
        return cur_out
