import numpy as np
import pandas as pd


from pandas import DataFrame
from pandas._typing import Axes, Dtype
from MomentumIndicator import MomentumIndicator


class VolumnIndicator(MomentumIndicator):

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)
        
    # Moving Average
    def volumn_SMA(
        self, 
        days: int, 
        inplace: bool=False, 
        volumn_col_name: str="volumn",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        cpy_data["volumn_SMA"+str(days)] = cpy_data[volumn_col_name].rolling(days).mean()
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="volumn_SMA"+str(days))
    
    
    def volumn_EMA(
        self, 
        days: int, 
        inplace: bool=False, 
        volumn_col_name: str="volumn",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()

        tmp_volumn_SMA_data = cpy_data[volumn_col_name].rolling(days).mean()

        np_tmp_volumn_SMA_data = tmp_volumn_SMA_data.to_numpy()
        
        lst = [pd.NA for _ in range(0, len(cpy_data))]
        lst[days-1] = np_tmp_volumn_SMA_data[days-1]
        K = 2/(days+1)
        
        for i in range(days, len(cpy_data)):
            lst[i] = cpy_data[volumn_col_name].to_numpy()[i]*K + lst[i-1]*(1-K)
            
        cpy_data["volumn_EMA"+str(days)] = np.array(lst)
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="volumn_EMA"+str(days))
    
    
    def volumn_WMA(
        self, 
        days: int, 
        inplace: bool=False, 
        volumn_col_name: str="volumn",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        multipliers = np.array([day+1 for day in range(days)])
        lst = [pd.NA for _ in range(len(cpy_data))]
        
        for i in range(days,len(cpy_data)):
            lst[i-1] = sum(multipliers*cpy_data[volumn_col_name][i-days:i])/(days*(days+1)/2)
            
        cpy_data["volumn_WMA"+str(days)] = np.array(lst)
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="volumn_WMA"+str(days))



