import numpy as np
import pandas as pd


from pandas import DataFrame
from pandas._typing import Axes, Dtype
from AdditionalIndicator import AdditionalIndicator


class MAIndicator(AdditionalIndicator):
    """
    This class provides some moving average indicators for price.
    """

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)

    def price_SMA(
        self,
        days: int,
        inplace: bool = False,
        price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the Smoothed Moving Average of 
        the price in n days from the current day

        Parameters
        ----------
        days : int
            the number of days included in SMA calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        price_col_name : str, default 'close'
            the colume name of the price feature to be calculated.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including '[price_col_name]_price_SMA[days]' colume
        """

        cp_data = self.copy()
        cp_data[price_col_name+"_price_SMA" +
                 str(days)] = cp_data[price_col_name].rolling(days).mean()

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name=price_col_name+"_price_SMA"+str(days))


    def price_EMA(
        self,
        days: int,
        inplace: bool = False,
        price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the Exponential Moving Average of 
        the price in n days from the current day

        Parameters
        ----------
        days : int
            the number of days included in price's EMA calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        price_col_name : str, default 'close'
            the colume name of the price feature to be calculated.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including '[price_col_name]_price_EMA[days]' colume
        """

        cp_data = self.copy()

        tmp_price_SMA_data = cp_data[price_col_name].rolling(
            days).mean()
        
        np_tmp_price_SMA_data = tmp_price_SMA_data.to_numpy()

        lst = [pd.NA for _ in range(0, len(cp_data))]
        lst[days-1] = np_tmp_price_SMA_data[days-1]
        K = 2/(days+1)

        for i in range(days, len(cp_data)):
            lst[i] = cp_data[price_col_name].to_numpy()[i]*K + lst[i-1]*(1-K)

        cp_data[price_col_name+"_price_EMA"+str(days)] = np.array(lst)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name=price_col_name+"_price_EMA"+str(days))


    def price_WMA(
        self,
        days: int,
        inplace: bool = False,
        price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the Weighted Moving Average
        of the price in n days from the current day

        Parameters
        ----------
        days : int
            the number of days included in price's WMA calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        price_col_name : str, default 'close'
            the colume name of the price feature to be calculated.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including '[price_col_name]_price_WMA[days]' colume
        """

        cp_data = self.copy()
        multipliers = np.array([day+1 for day in range(days)])
        lst = [pd.NA for _ in range(len(cp_data))]

        for i in range(days, len(cp_data)+1):
            lst[i-1] = (multipliers*cp_data[price_col_name][i-days:i]).sum()/(days*(days+1)/2)

        cp_data[price_col_name+"_price_WMA"+str(days)] = np.array(lst)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name=price_col_name+"_price_WMA"+str(days))
