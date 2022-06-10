import numpy as np

def normalize_trend_and_seasonal(ts_dict: dict, lookback: int=20):
    '''
    This function normalize trend + seasonal by subtract the curernt value by the mean
    and then divide by the std of the 'lookback' previous values
    Parameters:
        lookback: the number of previous data value to calculate
        ts_dict: the dictionary containing the trend + seasonal of each column
    '''
    
    ts_dict_copy = ts_dict.copy()
    for col, value in ts_dict_copy.items():
        value_rolling_mean = value.rolling(lookback).mean()
        value_rolling_std = value.rolling(lookback).std()
        value = (value - value_rolling_mean) / value_rolling_std
        ts_dict_copy[col] = value
        
    return ts_dict_copy
        
    