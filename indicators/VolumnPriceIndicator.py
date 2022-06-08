import numpy as np
import pandas as pd


from pandas import DataFrame
from pandas._typing import Axes, Dtype
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

    # Not optimized because of using for
    def MFI(
        self, 
        days: int=14,
        inplace: bool=False, 
        dropna: bool=False
    ) -> DataFrame:

        cpy_data = self.copy()
        cpy_data = VolumnPriceIndicator(cpy_data).raw_money_flow(inplace=True)
        cpy_data = VolumnPriceIndicator(cpy_data).typical_price(inplace=True)
        shift_cpy_data = cpy_data.copy().shift(1)
        lst = [pd.NA for _ in range(len(self))]

        for i in range(days+1, len(self)):
            up = shift_cpy_data.iloc[i-days:i,:][cpy_data.iloc[i-days:i,:]["typical_price"] > shift_cpy_data.iloc[i-days:i,:]["typical_price"]]["raw_money_flow"].sum()
            down = shift_cpy_data.iloc[i-days:i,:][cpy_data.iloc[i-days:i,:]["typical_price"] <=
                                  shift_cpy_data.iloc[i-days:i,:]["typical_price"]]["raw_money_flow"].sum()
            lst[i] = 100 - 100/(1+(up/down))

        cpy_data["MFI"+str(days)] = np.array(lst)

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="price_diff")