import pandas as pd
import numpy as np
from pathlib import Path

def label(path):
    df = pd.read_csv(path, index_col="Time_UTC_Start")
    df.drop(["Timestamp", "Timestamp End", "N/A.5"], axis=1, inplace=True)
    df.rename(columns = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                        'N/A': 'volume', 'N/A.1': 'quote_asset_volume', 'N/A.2': 'number_of_trades',
                        'N/A.3': 'base_asset_volume', 'N/A.4': 'quote_asset_volume'}, inplace = True)
    df = df.iloc[26:1999947, :]
    df = df.drop_duplicates(keep = 'last')
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('H').fillna(method = 'ffill')
    length = df.shape[0]
    op = df['open'].values.reshape(length,1)
    cl = df['close'].values.reshape(length,1 )
    label = np.ones((length, 1)) * (op < cl)
    df['label'] = label
    
    return df
