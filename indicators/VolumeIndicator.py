import numpy as np
import pandas as pd


from pandas import DataFrame
from pandas._typing import Axes, Dtype
from MomentumIndicator import MomentumIndicator


class VolumeIndicator(MomentumIndicator):
    '''
    This class provides indicators including volume in their formula
    '''
    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)
        

    def volume_SMA(
        self, 
        days: int, 
        inplace: bool=False, 
        volume_col_name: str="volume",
        dropna: bool=False
    ) -> DataFrame:
        '''
        Smoothed Moving Average of volume in n days from the current day
        '''
        cp_data = self.copy()
        cp_data["volume_SMA"+str(days)] = cp_data[volume_col_name].rolling(days).mean()
        
        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="volume_SMA"+str(days))
    
    
    def volume_EMA(
        self, 
        days: int, 
        inplace: bool=False, 
        volume_col_name: str="volume",
        dropna: bool=False
    ) -> DataFrame:
        '''
        Exponential Moving Average of volume in n days from the current day
        '''
        cp_data = self.copy()

        tmp_volume_SMA_data = cp_data[volume_col_name].rolling(days).mean()

        np_tmp_volume_SMA_data = tmp_volume_SMA_data.to_numpy()
        
        lst = [pd.NA for _ in range(0, len(cp_data))]
        lst[days-1] = np_tmp_volume_SMA_data[days-1]
        K = 2/(days+1)
        
        for i in range(days, len(cp_data)):
            lst[i] = cp_data[volume_col_name].to_numpy()[i]*K + lst[i-1]*(1-K)
            
        cp_data["volume_EMA"+str(days)] = np.array(lst)
        
        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="volume_EMA"+str(days))
    
    
    def volume_WMA(
        self, 
        days: int, 
        inplace: bool=False, 
        volume_col_name: str="volume",
        dropna: bool=False
    ) -> DataFrame:
        '''
        Weighted Moving Average of volume in n days from the current day
        '''
        cp_data = self.copy()
        multipliers = np.array([day+1 for day in range(days)])
        lst = [pd.NA for _ in range(len(cp_data))]
        
        for i in range(days,len(cp_data)+1):
            lst[i-1] = sum(multipliers*cp_data[volume_col_name][i-days:i])/(days*(days+1)/2)
            
        cp_data["volume_WMA"+str(days)] = np.array(lst)
        
        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="volume_WMA"+str(days))



