import numpy as np
import pandas as pd


from pandas import DataFrame
from pandas._typing import Axes, Dtype
from VolumeIndicator import VolumeIndicator


class VolumePriceIndicator(VolumeIndicator):
    """
    This class provides indicators which combine volume and price in their formula
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

    
    def raw_money_flow(
        self, 
        inplace: bool=False, 
        closing_price_col_name: str="close",
        high_price_col_name: str="high",
        low_price_col_name: str="low",
        volume_col_name: str="volume",
        dropna: bool=False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is raw money flow - the multiplication 
        of typical price and volume.

        Parameters
        ----------
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        high_price_col_name : str, default 'high'
            the colume name of the high price feature.
        low_price_col_name : str, default 'low'
            the colume name of the low price feature.
        volume_col_name : str, default 'volume'
            the name of the volume colume.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'raw_money_flow' colume
        """

        cp_data = self.copy()
        typical_price_data= self.typical_price(inplace=False, closing_price_col_name=closing_price_col_name, high_price_col_name=high_price_col_name, low_price_col_name=low_price_col_name)["typical_price"]
        cp_data["raw_money_flow"] = typical_price_data * cp_data[volume_col_name]

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="raw_money_flow")


    def MFI(
        self, 
        days: int=14,
        inplace: bool=False, 
        closing_price_col_name: str="close",
        high_price_col_name: str="high",
        low_price_col_name: str="low",
        volume_col_name: str="volume",
        dropna: bool=False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is MFI feature.

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
        volume_col_name : str, default 'volume'
            the name of the volume colume.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'MFI + str(days)' colume

        Note
        -------
        The Money Flow Index (MFI) is a technical oscillator that uses price and volume data for 
        identifying overbought or oversold signals in an asset. It can also be used to spot 
        divergences which warn of a trend change in price. The oscillator moves between 0 and 100.
        (Source: Investopedia)
        """

        cp_data = self.copy()
        cp_data = VolumePriceIndicator(cp_data).raw_money_flow(inplace=True, closing_price_col_name=closing_price_col_name, high_price_col_name=high_price_col_name, low_price_col_name=low_price_col_name, volume_col_name=volume_col_name)
        cp_data = VolumePriceIndicator(cp_data).typical_price(inplace=True, closing_price_col_name=closing_price_col_name, high_price_col_name=high_price_col_name, low_price_col_name=low_price_col_name)
        shift_cp_data = cp_data.copy().shift(1)
        lst = [pd.NA for _ in range(len(self))]

        for i in range(days-1, len(self)):
            tmp_shift_cp_data = shift_cp_data.iloc[i-days+1:i+1,:]
            tmp_cp_data = cp_data.iloc[i-days+1:i+1,:]
            
            up = tmp_shift_cp_data[tmp_cp_data["typical_price"] > tmp_shift_cp_data["typical_price"]]["raw_money_flow"].sum()
            down = tmp_shift_cp_data[tmp_cp_data["typical_price"] <=
                                  tmp_shift_cp_data["typical_price"]]["raw_money_flow"].sum()
            lst[i] = 100 - 100/(1+(up/(1e-10+down)))

        cp_data["MFI"+str(days)] = np.array(lst)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="MFI"+str(days))
