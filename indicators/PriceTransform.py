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
        closing_price_col_name: str="close",
        high_price_col_name: str="high",
        low_price_col_name: str="low",
        dropna: bool=False
    ) -> DataFrame:
        cpy_data = self.copy()

        cpy_data["typical_price"] = (cpy_data[closing_price_col_name] + cpy_data[high_price_col_name] + cpy_data[low_price_col_name])/3

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="typical_price")



    def plus_DM(
        self,
        inplace: bool=False, 
        high_price_col_name: str="high",
        dropna: bool=False
    ) -> DataFrame:
        cpy_data = self.copy()
        cpy_data["zeros"] = [0 for _ in range(len(cpy_data))]
        cpy_data["plus_DM"] = cpy_data[high_price_col_name] - cpy_data.shift(1)[high_price_col_name]
        cpy_data["plus_DM"] = cpy_data[["plus_DM", "zeros"]].max(axis=1)
        cpy_data.drop(columns=["zeros"], inplace=True)

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="plus_DM")


    def smoothed_plus_DM(
        self,
        days: int=14,
        inplace: bool=False, 
        high_price_col_name: str="high",
        dropna: bool=False
    ) -> DataFrame:
        cpy_data = self.copy()

        cpy_data["smoothed_plus_DM"+str(days)] = PriceTransform(cpy_data).plus_DM(
            inplace=False, high_price_col_name=high_price_col_name)["plus_DM"].rolling(days).sum() - PriceTransform(cpy_data).plus_DM(
            inplace=False, high_price_col_name=high_price_col_name)["plus_DM"].rolling(days).mean() + PriceTransform(cpy_data).plus_DM(
            inplace=False, high_price_col_name=high_price_col_name)["plus_DM"]

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="smoothed_plus_DM"+str(days))


    def minus_DM(
        self,
        inplace: bool=False, 
        low_price_col_name: str="low",
        dropna: bool=False
    ) -> DataFrame:
        cpy_data = self.copy()
        cpy_data["zeros"] = [0 for _ in range(len(cpy_data))]
        cpy_data["minus_DM"] = cpy_data.shift(1)[low_price_col_name] - cpy_data[low_price_col_name]
        cpy_data["minus_DM"] = cpy_data[["minus_DM", "zeros"]].max(axis=1)
        cpy_data.drop(columns=["zeros"], inplace=True)

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="minus_DM")


    def smoothed_minus_DM(
        self,
        days: int=14,
        inplace: bool=False, 
        low_price_col_name: str="low",
        dropna: bool=False
    ) -> DataFrame:
        cpy_data = self.copy()

        cpy_data["smoothed_minus_DM"+str(days)] = PriceTransform(cpy_data).minus_DM(
            inplace=False, low_price_col_name=low_price_col_name)["minus_DM"].rolling(days).sum() - PriceTransform(cpy_data).minus_DM(
            inplace=False, low_price_col_name=low_price_col_name)["minus_DM"].rolling(days).mean() + PriceTransform(cpy_data).minus_DM(
            inplace=False, low_price_col_name=low_price_col_name)["minus_DM"]

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="smoothed_minus_DM"+str(days))


    def TR(
        self,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        high_price_col_name: str = "high",
        low_price_col_name: str = "low",
        dropna: bool = False
    ) -> DataFrame:
        cpy_data = self.copy()

        cpy_data["high_low"] = cpy_data[high_price_col_name] - cpy_data[low_price_col_name]
        cpy_data["high_close"] = abs(cpy_data[high_price_col_name] - cpy_data[closing_price_col_name])
        cpy_data["low_close"] = abs(cpy_data[low_price_col_name] - cpy_data[closing_price_col_name])

        cpy_data["TR"] = cpy_data[["high_close",
                                   "high_low", "low_close"]].max(axis=1)

        cpy_data.drop(columns=["high_low", "high_close", "low_close"], inplace=True)

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="TR")


    def ATR(
        self,
        days: int=14,
        inplace: bool = False,
        dropna: bool = False
    ) -> DataFrame:
        cpy_data = self.copy()

        cpy_data["ATR"+str(days)] = PriceTransform(cpy_data).TR(inplace=True)["TR"].rolling(days).mean()

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="ATR"+str(days))
