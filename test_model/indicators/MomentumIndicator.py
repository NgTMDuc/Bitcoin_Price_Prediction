from calendar import c
from matplotlib.pyplot import close
from pandas import DataFrame
from MAIndicator import MAIndicator
from pandas._typing import Axes, Dtype


import numpy as np
import pandas as pd


class MomentumIndicator(MAIndicator):

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)

    # Relative strength indicators
    def RS(
        self,
        days: int,
        inplace: bool = False,
        closing_price_col_name: str = "Close",
        dropna: bool = False
    ) -> DataFrame:

        cpy_data = self.copy()

        price_up_data = \
            MomentumIndicator(cpy_data).price_up_SMA(days=days, inplace=False,
                                                     closing_price_col_name=closing_price_col_name)["price_up_SMA"+str(days)]
        price_down_data = \
            MomentumIndicator(cpy_data).price_down_SMA(days=days, inplace=False,
                                                       closing_price_col_name=closing_price_col_name)["price_down_SMA"+str(days)]

        # 1e-6 is used for avoiding zero division
        cpy_data["RS"+str(days)] = price_up_data / (1e-6 - price_down_data)

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="RS"+str(days))

    def RSI(
        self,
        days: int,
        inplace: bool = False,
        closing_price_col_name: str = "Close",
        dropna: bool = False
    ) -> DataFrame:

        cpy_data = self.copy()
        cpy_data_RS = MomentumIndicator(cpy_data).RS(
            days=days, inplace=False, closing_price_col_name=closing_price_col_name)["RS"+str(days)]
        cpy_data["RSI"+str(days)] = 100 - 100/(1 + cpy_data_RS.to_numpy())

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="RSI"+str(days))

    # Indicators deduced from MA

    def PPO(
        self,
        first_period_EMA: int = 12,
        second_period_EMA: int = 26,
        inplace: bool = False,
        closing_price_col_name: str = "Close",
        dropna: bool = False
    ) -> DataFrame:

        cpy_data = self.copy()
        cpy_data_first_EMA = MomentumIndicator(cpy_data).price_EMA(
            days=first_period_EMA, inplace=False, closing_price_col_name=closing_price_col_name)["price_EMA"+str(first_period_EMA)]
        cpy_data_second_EMA = MomentumIndicator(cpy_data).price_EMA(
            days=second_period_EMA, inplace=False, closing_price_col_name=closing_price_col_name)["price_EMA"+str(second_period_EMA)]
        cpy_data["PPO%s-%s" % (str(first_period_EMA), str(second_period_EMA))] = \
            100 * (cpy_data_first_EMA - cpy_data_second_EMA) / \
            cpy_data_second_EMA

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="PPO%s-%s" % (str(first_period_EMA), str(second_period_EMA)))

    def DX(
        self,
        days: int = 14,
        inplace: bool = False,
        high_price_col_name: str = "High",
        low_price_col_name: str = "Low",
        dropna: bool = False
    ) -> DataFrame:

        cpy_data = self.copy()

        cpy_data["ATR"+str(days)] = MomentumIndicator(
            cpy_data).ATR(days=days, inplace=True)["ATR"+str(days)]
        cpy_data["plus_DI"+str(days)] = 100 * MomentumIndicator(cpy_data).smoothed_plus_DM(days=days, inplace=True,
                                                                                           high_price_col_name=high_price_col_name)["smoothed_plus_DM"+str(days)] / cpy_data["ATR"+str(days)]
        cpy_data["minus_DI"+str(days)] = 100 * MomentumIndicator(cpy_data).smoothed_minus_DM(days=days, inplace=True,
                                                                                             low_price_col_name=low_price_col_name)["smoothed_minus_DM"+str(days)] / cpy_data["ATR"+str(days)]
        cpy_data["DX"+str(days)] = 100 * abs((cpy_data["plus_DI"+str(days)] - cpy_data["minus_DI" +
                                                                                       str(days)])/(cpy_data["plus_DI"+str(days)] + cpy_data["minus_DI"+str(days)]))

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="DX"+str(days))

    def ADX(
        self,
        days: int = 14,
        inplace: bool = False,
        high_price_col_name: str = "High",
        low_price_col_name: str = "Low",
        dropna: bool = False
    ) -> DataFrame:

        cpy_data = self.copy()

        cpy_data["DX"+str(days)] = MomentumIndicator(cpy_data).DX(days=days, inplace=True,
                                                                  high_price_col_name=high_price_col_name, low_price_col_name=low_price_col_name)["DX"+str(days)]

        lst = [pd.NA for _ in range(len(cpy_data))]
        lst[days*2] = cpy_data["DX"+str(days)].to_numpy()[days:days*2].mean()
        lst[days*2+1] = cpy_data["DX" +
                                 str(days)].to_numpy()[days+1:days*2+1].mean()

        for i in range(days*2+2, len(cpy_data)):
            lst[i] = (cpy_data["DX"+str(days)].to_numpy()[i-2] *
                      (days-1) + cpy_data["DX"+str(days)].to_numpy()[i-1])/days

        cpy_data["ADX"+str(days)] = np.array(lst)

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="ADX"+str(days))

    def BOP(
        self,
        days: int = 14,
        inplace: bool = False,
        closing_price_col_name: str = "Close",
        opening_price_col_name: str = "Open",
        high_price_col_name: str = "High",
        low_price_col_name: str = "Low",
        dropna: bool = False
    ) -> DataFrame:

        cpy_data = self.copy()
        cpy_data["close_open"] = cpy_data[closing_price_col_name] - \
            cpy_data[opening_price_col_name]
        cpy_data["high_low"] = cpy_data[high_price_col_name] - \
            cpy_data[low_price_col_name]

        cpy_data["BOP"+str(days)] = (cpy_data["close_open"] /
                                     cpy_data["high_low"]).rolling(days).mean()
        cpy_data.drop(columns=["close_open", "high_low"], inplace=True)

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="BOP"+str(days))

    def CCI(
        self,
        days: int = 14,
        inplace: bool = False,
        closing_price_col_name: str = "Close",
        high_price_col_name: str = "High",
        low_price_col_name: str = "Low",
        dropna: bool = False
    ) -> DataFrame:

        cpy_data = self.copy()
        cpy_data = MomentumIndicator(cpy_data).typical_price(closing_price_col_name=closing_price_col_name,
                                                             high_price_col_name=high_price_col_name, low_price_col_name=low_price_col_name, inplace=True)

        cpy_data["CCI"+str(days)] = (cpy_data["typical_price"] - cpy_data["typical_price"].rolling(days).mean()) / (
            0.015 * MomentumIndicator(cpy_data).mad(days=days, closing_price_col_name=closing_price_col_name)["mad"+str(days)])
        cpy_data.drop(columns=["typical_price"])

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="CCI"+str(days))

    def MACD(
        self,
        first_period_EMA: int = 12,
        second_period_EMA: int = 26,
        inplace: bool = False,
        closing_price_col_name: str = "Close",
        dropna: bool = False
    ) -> DataFrame:

        cpy_data = self.copy()

        cpy_data = MomentumIndicator(cpy_data).price_EMA(
            days=12, closing_price_col_name=closing_price_col_name, inplace=True)

        cpy_data["MACD"+str(first_period_EMA)+'-'+str(second_period_EMA)] = MomentumIndicator(cpy_data).price_EMA(days=first_period_EMA, closing_price_col_name=closing_price_col_name, inplace=True)[
            "price_EMA"+str(first_period_EMA)] - MomentumIndicator(cpy_data).price_EMA(days=second_period_EMA, closing_price_col_name=closing_price_col_name, inplace=True)["price_EMA"+str(second_period_EMA)]

        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="MACD"+str(first_period_EMA)+'-'+str(second_period_EMA))
