import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas._typing import Axes, Dtype
from PriceTransform import PriceTransform


class AdditionalIndicator(PriceTransform):
    """
    This class provides some useful indicators for others main indicators
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


    def price_up(
        self,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the difference between the current price 
        and the previous price if the current price is greater than the previous price.

        Parameters
        ----------
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'price_up' colume
        """

        cp_data = self.copy()

        c = pd.DataFrame({
            "Diff": self[closing_price_col_name] - self[closing_price_col_name].shift(1),
            "Zero": [0 for i in np.arange(len(self))]
        })

        cp_data["price_up"] = c.max(axis=1)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="price_up")


    def price_up_SMA(
        self,
        days: int,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the SMA of price_up.

        Parameters
        ----------
        days : int
            the number of days included in price_up's mean calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'price_up_SMA + str(days)' colume
        """

        cp_data = self.copy()
        cp_price_up_data = AdditionalIndicator(cp_data).price_up(
            inplace=False, closing_price_col_name=closing_price_col_name)["price_up"]
        cp_data["price_up_SMA" +
                 str(days)] = cp_price_up_data.rolling(days).mean()

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="price_up_SMA"+str(days))


    def price_down(
        self,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the difference between the 
        current price and the previous price if the current price is less than the previous price.

        Parameters
        ----------
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'price_down' colume
        """

        cp_data = self.copy()

        c = pd.DataFrame({
            "Diff": self[closing_price_col_name] - self[closing_price_col_name].shift(1),
            "Zero": [0 for i in np.arange(len(self))]
        })

        cp_data["price_down"] = c.min(axis=1)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="price_down")


    def price_down_SMA(
        self,
        days: int,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the SMA of price_down.

        Parameters
        ----------
        days : int
            the number of days included in price_down's mean calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'price_down_SMA + str(days)' colume
        """

        cp_data = self.copy()
        cp_price_down_data = AdditionalIndicator(cp_data).price_down(
            inplace=False, closing_price_col_name=closing_price_col_name)["price_down"]
        cp_data["price_down_SMA" +
                 str(days)] = cp_price_down_data.rolling(days).mean()

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="price_down_SMA"+str(days))


    def price_diff(
        self,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the price difference of 
        the current row and the previous row

        Parameters
        ----------
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'price_diff' colume
        """

        cp_data = self.copy()

        cp_data["price_diff"] = AdditionalIndicator(cp_data).price_down(inplace=inplace, closing_price_col_name=closing_price_col_name)["price_down"] \
            + AdditionalIndicator(cp_data).price_up(inplace=inplace,
                                                     closing_price_col_name=closing_price_col_name)["price_up"]

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="price_diff")


    def std(
        self,
        days: int,
        ddof: int=0,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the standard deviation 
        of the price in n days from the current day.

        Parameters
        ----------
        days : int
            the number of days included in standard deviation calculation.
        ddof : int, default 0
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements. Do not change this value if
            you don't know how to use it, read pandas readme for more precise.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'std + str(days)' colume
        """

        cp_data = self.copy()

        cp_data["std" +
                 str(days)] = cp_data[closing_price_col_name].rolling(days).std(ddof=ddof)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="std"+str(days))


    def mad(
        self,
        days: int,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        high_price_col_name: str = "high",
        low_price_col_name: str = "low",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the mean absolute deviation 
        of the price in n days from the current day.

        Parameters
        ----------
        days : int
            the number of days included in mad calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        high_price_col_name : str, default 'high'
            the colume name of the high price feature.
        low_price_col_name : str, default 'low'
            the colume name of the low price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'mad + str(days)' colume
        """

        cp_data = self.copy()

        cp_data["typical_price"] = AdditionalIndicator(cp_data).typical_price(
            inplace=True, closing_price_col_name=closing_price_col_name, high_price_col_name=high_price_col_name, low_price_col_name=low_price_col_name)["typical_price"]
        cp_data["mad" +
                 str(days)] = cp_data["typical_price"].rolling(days).apply(DataFrame.mad)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="mad"+str(days))


    def median(
        self,
        days: int,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the median of 
        the price in n days from the current day.

        Parameters
        ----------
        days : int
            the number of days included in median calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'median + str(days)' colume
        """

        cp_data = self.copy()

        cp_data["median" +
                 str(days)] = cp_data[closing_price_col_name].rolling(days).median()

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="median"+str(days))