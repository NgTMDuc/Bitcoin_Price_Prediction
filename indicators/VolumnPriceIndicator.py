import numpy as np
import pandas as pd


from pandas import DataFrame
from pandas._typing import Axes, Dtype
from PriceTransform import PriceTransform
from VolumnIndicator import VolumnIndicator


class VolumnPriceIndicator(VolumnIndicator):

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)

    
    def raw_money_flow(
        self, 
        inplace: bool=False, 
        volumn_col_name: str="Volumn",
        dropna: bool=False
    ) -> DataFrame:
        cpy_data = self.copy()
        typical_price_data= self.typical_price(inplace=False)["typical_price"]
        cpy_data["raw_money_flow"] = typical_price_data * cpy_data[volumn_col_name]

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="raw_money_flow")

    
    def MFI(
        self, 
        days: int=14,
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:

        cpy_data = self.copy()
        raw_money_flow = self.raw_money_flow(inplace=False)["raw_money_flow"].to_numpy()
        lst = [pd.NA for i in range(len(self))]
        closing_price = self[closing_price_col_name].to_numpy()

        for i in range(days*2+1, len(self)):
            j = i - 1
            count_up = 0
            count_down = 0
            up = 0
            down = 0
            while j >= 0 and count_up <= 14 and count_down <= 14: 
                if closing_price[j] > closing_price[j-1]:
                    up += raw_money_flow[j]
                    count_up += 1
                else:
                    down += raw_money_flow[j]
                    count_down += 1
                j -= 1
            
            if j < 0:
                lst[i] = pd.NA
            else:
                lst[i] = 100 - 100/(1+(up/down))

        cpy_data["MFI"+str(days)] = np.array(lst)

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_diff")