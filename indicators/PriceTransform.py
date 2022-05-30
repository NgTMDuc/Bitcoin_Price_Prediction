import numpy as np
import pandas as pd


from pandas import DataFrame
from pandas._typing import Axes, Dtype


class PriceTransform(DataFrame):

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)

    # Return function
    def return_df(self, cpy_data: DataFrame, dropna: bool, inplace: bool, drop_col_name: str) -> DataFrame:
        if dropna:
            cpy_data._update_inplace(cpy_data.dropna(subset=[drop_col_name]))
        if inplace:
            self._update_inplace(cpy_data)
        return cpy_data

    # Calculate typical price
    def typical_price(
        self,
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        high_price_col_name: str="High",
        low_price_col_name: str="Low",
        dropna: bool=False
    ) -> DataFrame:
        cpy_data = self.copy()

        cpy_data["typical_price"] = (cpy_data[closing_price_col_name] + cpy_data[high_price_col_name] + cpy_data[low_price_col_name])/3

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="typical_price")