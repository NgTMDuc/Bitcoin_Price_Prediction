import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas._typing import Axes, Dtype
from PriceTransform import PriceTransform


class AdditionalIndicator(PriceTransform):

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)
    

    # Additional indicators
    def price_up(
        self, 
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        
        c = pd.DataFrame({
            "Diff": self[closing_price_col_name] - self[closing_price_col_name].shift(1),
            "Zero": [0 for i in np.arange(len(self))]
        })
        
        cpy_data["price_up"] = c.max(axis=1)
    
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_up")
    
    
    def price_up_SMA(
        self, 
        days: int, 
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        cpy_price_up_data = AdditionalIndicator(cpy_data).price_up(
            inplace=False, closing_price_col_name=closing_price_col_name)["price_up"]
        cpy_data["price_up_SMA"+str(days)] = cpy_price_up_data.rolling(days, closed="left").mean()
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_up_SMA"+str(days))
    
    
    def price_down(
        self, 
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame: 
        
        cpy_data = self.copy()
        
        c = pd.DataFrame({
            "Diff": self[closing_price_col_name] - self[closing_price_col_name].shift(1),
            "Zero": [0 for i in np.arange(len(self))]
        })
        
        cpy_data["price_down"] = c.min(axis=1)
    
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_down")
    
    
    def price_down_SMA(
        self, 
        days: int, 
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        cpy_price_down_data = AdditionalIndicator(cpy_data).price_down(
            inplace=False, closing_price_col_name=closing_price_col_name)["price_down"]
        cpy_data["price_down_SMA"+str(days)] = cpy_price_down_data.rolling(days, closed="left").mean()

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_down_SMA"+str(days))
    
    
    def price_diff(
        self, 
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame: 
        
        cpy_data = self.copy()
        
        cpy_data["price_diff"] = AdditionalIndicator(cpy_data).price_down(inplace=inplace, closing_price_col_name=closing_price_col_name)["price_down"] \
                                 + AdditionalIndicator(cpy_data).price_up(inplace=inplace, closing_price_col_name=closing_price_col_name)["price_up"]
    
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_diff")


    def std(
        self,
        days: int,
        inplace: bool = False,
        closing_price_col_name: str = "Close",
        dropna: bool = False
    ) -> DataFrame:

        cpy_data = self.copy()

        cpy_data["std"+str(days)] = cpy_data[closing_price_col_name].rolling(days, closed="left").std()

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="std"+str(days))