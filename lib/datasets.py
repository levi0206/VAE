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

def rolling_window_1D(x: torch.Tensor, window_size: int):
    '''
    Creates rolling windows from a 1D tensor.
    Input shape: (length,)
    Output shape: (length - window_size + 1, window_size)
    
    See https://se.mathworks.com/help/econ/rolling-window-estimation-of-state-space-models.html
    '''
    # Ensure input is 1D
    if len(x.shape) != 1:
        raise ValueError("Input tensor must be 1D with shape (length,)")
    
    print("Tensor shape before rolling:", x.shape)
    length = x.shape[0]
    
    # Check if window_size is valid
    if window_size > length:
        raise ValueError("Window size must be smaller than or equal to tensor length")
    
    windowed_data = []
    for t in range(length - window_size + 1):
        window = x[t:t + window_size]
        windowed_data.append(window)
    
    result = torch.stack(windowed_data, dim=0)
    print("Tensor shape after rolling:", result.shape)

    return result

def get_stock_price(data_config):
    """
    Get stock price.
    Shape: (#data, window_size, 1)
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
def get_OU(theta=0.8,mu=0,sigma=0.7,X0=1.0,T=15.0,dt=0.01):
    """
    Get an numpy array OU process of length (T / dt).
    Shape: (T / dt,)
    """
    N = int(T / dt)
    X = np.zeros(N)
    X[0] = X0

    # Generate the OU process
    for t in range(1, N):
        dW = np.sqrt(dt) * np.random.normal(0, 1)
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW

    return X