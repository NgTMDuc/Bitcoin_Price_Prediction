import numpy as np
import pandas as pd


from pandas import DataFrame
from pandas._typing import Axes, Dtype
from AdditionalIndicator import AdditionalIndicator


class MAIndicator(AdditionalIndicator):       

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
    def price_SMA(
        self, 
        days: int, 
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        cpy_data["price_SMA"+str(days)] = cpy_data[closing_price_col_name].rolling(days, closed="left").mean()
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_SMA"+str(days))
    
    
    def price_EMA(
        self, 
        days: int, 
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        tmp_price_SMA_data = cpy_data[closing_price_col_name].rolling(days, closed="left").mean()
        tmp_price_SMA_data.dropna(inplace=True)
        np_tmp_price_SMA_data = tmp_price_SMA_data.to_numpy()
        
        lst = [pd.NA for i in range(0, len(cpy_data))]
        lst[days] = np_tmp_price_SMA_data[0]
        lst[days+1] = np_tmp_price_SMA_data[1]
        K = 2/(days+1)
        
        for i in range(days+2, len(cpy_data)):
            lst[i] = lst[i-1]*K + lst[i-2]*(1-K)
            
        cpy_data["price_EMA"+str(days)] = np.array(lst)
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_EMA"+str(days))
    
    
    def price_WMA(
        self, 
        days: int, 
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        multipliers = np.array([day+1 for day in range(days)])
        lst = [pd.NA for i in range(0, len(cpy_data))]
        
        for i in range(days,len(cpy_data)):
            lst[i] = sum(multipliers*cpy_data[closing_price_col_name][i-days:i].to_numpy())/(days*(days+1)/2)
            
        cpy_data["price_WMA"+str(days)] = np.array(lst)
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_WMA"+str(days))