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
    def return_df(self, cp_data: DataFrame, dropna: bool, inplace: bool, drop_col_name: str) -> DataFrame:
        if dropna:
            cp_data._update_inplace(cp_data.dropna(subset=[drop_col_name]))
        if inplace:
            self._update_inplace(cp_data)
        return cp_data


    def typical_price(
        self,
        inplace: bool=False, 
        closing_price_col_name: str="close",
        high_price_col_name: str="high",
        low_price_col_name: str="low",
        dropna: bool=False
    ) -> DataFrame:
        '''
        Calculate typical price - the average value of close price, high price and low price
        '''
        cp_data = self.copy()

        cp_data["typical_price"] = (cp_data[closing_price_col_name] + cp_data[high_price_col_name] + cp_data[low_price_col_name])/3

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="typical_price")



    def plus_DM(
        self,
        inplace: bool=False, 
        high_price_col_name: str="high",
        dropna: bool=False
    ) -> DataFrame:
        '''
        +DM = current high - previous high
        '''

        cp_data = self.copy()
        cp_data["zeros"] = [0 for _ in range(len(cp_data))]
        cp_data["plus_DM"] = cp_data[high_price_col_name] - cp_data.shift(1)[high_price_col_name]
        cp_data["plus_DM"] = cp_data[["plus_DM", "zeros"]].max(axis=1)
        cp_data.drop(columns=["zeros"], inplace=True)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="plus_DM")


    def smoothed_plus_DM(
        self,
        days: int=14,
        inplace: bool=False, 
        high_price_col_name: str="high",
        dropna: bool=False
    ) -> DataFrame:
        '''
        Smoothed Moving Average of plus_DM
        '''
        cp_data = self.copy()

        cp_data["smoothed_plus_DM"+str(days)] = PriceTransform(cp_data).plus_DM(
            inplace=False, high_price_col_name=high_price_col_name)["plus_DM"].rolling(days).sum() - PriceTransform(cp_data).plus_DM(
            inplace=False, high_price_col_name=high_price_col_name)["plus_DM"].rolling(days).mean() + PriceTransform(cp_data).plus_DM(
            inplace=False, high_price_col_name=high_price_col_name)["plus_DM"]

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="smoothed_plus_DM"+str(days))


    def minus_DM(
        self,
        inplace: bool=False, 
        low_price_col_name: str="low",
        dropna: bool=False
    ) -> DataFrame:
        '''
        -DM = previous low - current low
        '''
        cp_data = self.copy()
        cp_data["zeros"] = [0 for _ in range(len(cp_data))]
        cp_data["minus_DM"] = cp_data.shift(1)[low_price_col_name] - cp_data[low_price_col_name]
        cp_data["minus_DM"] = cp_data[["minus_DM", "zeros"]].max(axis=1)
        cp_data.drop(columns=["zeros"], inplace=True)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="minus_DM")


    def smoothed_minus_DM(
        self,
        days: int=14,
        inplace: bool=False, 
        low_price_col_name: str="low",
        dropna: bool=False
    ) -> DataFrame:
        '''
        Smoothed Moving Average of minus_DM
        '''
        cp_data = self.copy()

        cp_data["smoothed_minus_DM"+str(days)] = PriceTransform(cp_data).minus_DM(
            inplace=False, low_price_col_name=low_price_col_name)["minus_DM"].rolling(days).sum() - PriceTransform(cp_data).minus_DM(
            inplace=False, low_price_col_name=low_price_col_name)["minus_DM"].rolling(days).mean() + PriceTransform(cp_data).minus_DM(
            inplace=False, low_price_col_name=low_price_col_name)["minus_DM"]

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="smoothed_minus_DM"+str(days))


    def TR(
        self,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        high_price_col_name: str = "high",
        low_price_col_name: str = "low",
        dropna: bool = False
    ) -> DataFrame:
        '''
        The True Range for today is the greatest of the following:
        + Today's high minus today's low
        + The absolute value of today's high minus yesterday's close
        + The absolute value of today's low minus yesterday's close
        (Source: Fidelity)
        '''

        cp_data = self.copy()

        cp_data["high_low"] = cp_data[high_price_col_name] - cp_data[low_price_col_name]
        cp_data["high_close"] = abs(cp_data[high_price_col_name] - cp_data[closing_price_col_name])
        cp_data["low_close"] = abs(cp_data[low_price_col_name] - cp_data[closing_price_col_name])

        cp_data["TR"] = cp_data[["high_close",
                                   "high_low", "low_close"]].max(axis=1)

        cp_data.drop(columns=["high_low", "high_close", "low_close"], inplace=True)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="TR")


    def ATR(
        self,
        days: int=14,
        inplace: bool = False,
        dropna: bool = False
    ) -> DataFrame:
        '''
        The mean of TR values in n days from the current day
        '''
        cp_data = self.copy()

        cp_data["ATR"+str(days)] = PriceTransform(cp_data).TR(inplace=True)["TR"].rolling(days).mean()

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="ATR"+str(days))
