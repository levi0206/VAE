import os
import glob
import pandas as pd
import random
import yfinance as yf
from fbm import fbm, MBM
import tqdm

import torch
import numpy as np
from lib.utils import sample_indices, load_obj, save_obj

def train_test_split(
        x: torch.Tensor,
        train_test_ratio: float,
        device: str
):
    size = x.shape[0]
    train_set_size = int(size * train_test_ratio)

    indices_train = sample_indices(size, train_set_size, device)
    indices_test = torch.LongTensor([i for i in range(size) if i not in indices_train])

    x_train = x[indices_train]
    x_test = x[indices_test]
    return x_train, x_test

def download_stock_price(
        ticker : str,
        start : str = '2012-01-01',
        end : str = '2024-12-31',
        interval: str = '1d',
):  
    '''
    If you want to download other stock data, please do it before execute your code.
    '''
    dataframe = yf.download(ticker, start=start, end=end, interval=interval)
    file_name = ticker+"_"+interval+".csv"
    csv_file_file = os.path.join("datasets", "stock", file_name) 
    if not os.path.exists(csv_file_file):
        dataframe.to_csv(file_name)
    return dataframe

def rolling_window(x: torch.Tensor, window_size: int):
    '''
    See https://se.mathworks.com/help/econ/rolling-window-estimation-of-state-space-models.html
    '''
    print("Tensor shape before rolling:",x.shape)
    windowed_data = []
    for t in range(x.shape[0] - window_size + 1):
        window = x[t:t + window_size, :]
        windowed_data.append(window)
    print("Tensor shape after rolling:",torch.stack(windowed_data, dim=0).shape)
    return torch.stack(windowed_data, dim=0)

def get_stock_price(data_config):
    """
    Get stock price
    Returns
    -------
    dataset: torch.Tensor
        torch.tensor of shape (#data, window_size, 1)
    """
    csv_file_name = data_config['ticker']+"_"+data_config['interval']+".csv"
    pt_file_name = data_config['ticker']+"_"+data_config['interval']+"_rolled.pt"
    csv_file_path = os.path.join(data_config['dir'], data_config['subdir'], csv_file_name) 
    pt_file_path = os.path.join(data_config['dir'], data_config['subdir'], pt_file_name)

    if not os.path.exists(csv_file_path):
        _ = download_stock_price(ticker=data_config['ticker'],interval=data_config['interval'])

    if os.path.exists(pt_file_path):
        dataset = load_obj(pt_file_path)
        print(f'Rolled data for training, shape {dataset.shape}')
        
    else:
        df = pd.read_csv(csv_file_path)
        print(f'Original data: {os.path.basename(csv_file_name)}, shape {df.shape}')
        dataset = df[df.columns[data_config['column']]].to_numpy(dtype='float')
        dataset = torch.FloatTensor(dataset).unsqueeze(dim=1)
        # print(dataset[:5])
        dataset = rolling_window(dataset, data_config['window_size'])
        print(f'Rolled data before transfer_percentage, shape {dataset.shape}')

        # We don't apply the transfer percentage here.
        # dataset = transfer_percentage(dataset)
        # print(f'Rolled data after transfer_percentage, shape {dataset.shape}')
        
        save_obj(dataset, pt_file_path)
    return dataset