import numpy as np
import pandas as pd

def label(path, t=15):
    df = pd.read_csv(path)
    df = df[df.shape[0] % t:]
    df2 = df.groupby(np.arange(len(df))//t, axis=0).agg({'Timestamp': 'min', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                                'N/A': 'sum', 'Timestamp End': 'max', 'N/A.1': 'sum', 'N/A.2': 'sum', 'N/A.3': 'sum',
                                'N/A.4': 'sum', 'N/A.5': 'sum', 'Time_UTC_Start': 'first'})
    df2.rename(columns = {'N/A': 'Volumn', 'N/A.1': 'Quote_asset_volume', 'N/A.2': 'Number_of_trades',
                         'N/A.3': 'Base_asset_volume', 'N/A.4': 'Quote_asset_volume'}, inplace = True)
    length = df2.shape[0]
    op = df2['Open'].values.reshape(length,1)
    cl = df2['Close'].values.reshape(length,1 )
    v = np.ones((length, 1)) * (op < cl)
    df2['label'] = v
    df2.index = df2["Timestamp"]
    df2.drop(columns=["Timestamp", "N/A.5"], inplace=True)
    return df2

