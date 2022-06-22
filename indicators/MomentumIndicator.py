from pandas import DataFrame
from MAIndicator import MAIndicator
from pandas._typing import Axes, Dtype


import numpy as np
import pandas as pd


class MomentumIndicator(MAIndicator):
    """
    This class provide momentrum indicators, which are technical analysis tools 
    used to determine the strength or weakness of a stock's price
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


    def RS(
        self,
        days: int,
        inplace: bool = False,
        price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the Relative Strength feature.

        Parameters
        ----------
        days : int
            the number of days included in RS calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        price_col_name : str, default 'close'
            the colume name of the price feature to be calculated.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame1
            a dataframe including '[price_col_name]_RS[days]' colume
        
        Note
        -------
        RS (relative strength) is an indicator used for RSI (relative strenght index). The RS is a measure 
        of the price trend of a stock compared to its benchmark index or sectoral index whereas Relative 
        Strength Index (RSI) is a momentum oscillator. This indicator oscillates between 0 and 100. RSI 
        is considered to be overbought when it is above 70 and oversold when it below 30.
        """

        cp_data = self.copy()

        price_up_data = \
            MomentumIndicator(cp_data).price_up_SMA(days=days, inplace=False,
                                                     price_col_name=price_col_name)[price_col_name+"_price_up_SMA"+str(days)]
        price_down_data = \
            MomentumIndicator(cp_data).price_down_SMA(days=days, inplace=False,
                                                       price_col_name=price_col_name)[price_col_name+"_price_down_SMA"+str(days)]

        # 1e-10 is used for avoiding zero division
        cp_data[price_col_name+"_RS"+str(days)] = price_up_data / (1e-10 - price_down_data)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name=price_col_name+"_RS"+str(days))


    def RSI(
        self,
        days: int,
        inplace: bool = False,
        price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the Relative Strength Index feature.

        Parameters
        ----------
        days : int
            the number of days included in RS calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        price_col_name : str, default 'close'
            the colume name of the price feature to be calculated.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including '[price_col_name]_RSI[days]' colume
        
        Note
        -------
        The relative strength index (RSI) is a momentum indicator used in technical analysis that measures 
        the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of 
        a stock or other asset. The RSI is displayed as an oscillator (a line graph that moves between two 
        extremes) and can have a reading from 0 to 100. (Source: Investopedia)
        """

        cp_data = self.copy()
        cp_data_RS = MomentumIndicator(cp_data).RS(
            days=days, inplace=False, price_col_name=price_col_name)[price_col_name+"_RS"+str(days)]
        cp_data[price_col_name+"_RSI"+str(days)] = 100 - 100/(1 + cp_data_RS.to_numpy())

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name=price_col_name+"_RSI"+str(days))


    def PPO(
        self,
        first_period_EMA: int = 12,
        second_period_EMA: int = 26,
        inplace: bool = False,
        price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the percentage price oscillator feature.

        Parameters
        ----------
        first_period_EMA : int, default 12,
            the first period for calculating EMA.
        first_period_EMA : int, default 26,
            the second period for calculating EMA.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        price_col_name : str, default 'close'
            the colume name of the price feature to be calculated.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including '[price_col_name]_PPO[first_period_EMA]-[second_period_EMA]' colume
        
        Note
        -------
        The percentage price oscillator (PPO) is a technical momentum indicator that shows the relationship 
        between two moving averages in percentage terms. The moving averages are a 26-period and 12-period 
        exponential moving average (EMA). But in this function, users can change the periods of 2 EMAs. 
        (Source: Investopedia)
        """

        cp_data = self.copy()
        cp_data_first_EMA = MomentumIndicator(cp_data).price_EMA(
            days=first_period_EMA, inplace=False, price_col_name=price_col_name)[price_col_name+"_price_EMA"+str(first_period_EMA)]
        cp_data_second_EMA = MomentumIndicator(cp_data).price_EMA(
            days=second_period_EMA, inplace=False, price_col_name=price_col_name)[price_col_name+"_price_EMA"+str(second_period_EMA)]
        cp_data[price_col_name+"_PPO%s-%s" % (str(first_period_EMA), str(second_period_EMA))] = \
            100 * (cp_data_first_EMA - cp_data_second_EMA) / \
            cp_data_second_EMA

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name=price_col_name+"_PPO%s-%s" % (str(first_period_EMA), str(second_period_EMA)))


    def DX(
        self,
        days: int = 14,
        inplace: bool = False,
        high_price_col_name: str = "high",
        low_price_col_name: str = "low",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the Directional Movement Index feature.

        Parameters
        ----------
        days : int
            the number of days included in DX calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        high_price_col_name : str, default 'high'
            the colume name of the high price feature.
        low_price_col_name : str, default 'low'
            the colume name of the low price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'DX+str(days)' colume
        
        Note
        -------
        The Directional Movement Index (DX) is an intermediate result in calculating of the Average Directional 
        Index (ADX) that was developed by J. Welles Wilder to evaluate the strength of a trend and to define a 
        period of sideway trading. The Directional Movement index is based on the positive and negative Directional 
        indicators and is used to spot crossovers of positive and negative directional indicators - when 
        bullish/bearish change of power takes place. (Source: Market Volume)
        """

        cp_data = self.copy()

        cp_data["ATR"+str(days)] = MomentumIndicator(
            cp_data).ATR(days=days, inplace=True)["ATR"+str(days)]
            
        cp_data["plus_DI"+str(days)] = 100 * MomentumIndicator(cp_data).smoothed_plus_DM(days=days, inplace=True,
                                                                                           high_price_col_name=high_price_col_name)["smoothed_plus_DM"+str(days)] / cp_data["ATR"+str(days)]
        cp_data["minus_DI"+str(days)] = 100 * MomentumIndicator(cp_data).smoothed_minus_DM(days=days, inplace=True,
                                                                                             low_price_col_name=low_price_col_name)["smoothed_minus_DM"+str(days)] / cp_data["ATR"+str(days)]
        cp_data["DX"+str(days)] = 100 * abs((cp_data["plus_DI"+str(days)] - cp_data["minus_DI" +
                                                                                       str(days)])/(1e-10 + cp_data["plus_DI"+str(days)] + cp_data["minus_DI"+str(days)]))
        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="DX"+str(days))


    def ADX(
        self,
        days: int = 14,
        inplace: bool = False,
        high_price_col_name: str = "high",
        low_price_col_name: str = "low",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the average directional index feature.

        Parameters
        ----------
        days : int
            the number of days included in DX calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        high_price_col_name : str, default 'high'
            the colume name of the high price feature.
        low_price_col_name : str, default 'low'
            the colume name of the low price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'ADX+str(days)' colume
        
        Note
        -------
        The average directional index (ADX) is a technical analysis indicator used by some traders to 
        determine the strength of a trend.  The trend can be either up or down, and this is shown by 
        two accompanying indicators, the negative directional indicator (-DI) and the positive directional 
        indicator (+DI). Therefore, the ADX commonly includes three separate lines. These are used to help 
        assess whether a trade should be taken long or short, or if a trade should be taken at all. 

        Calculating the Average Directional Movement Index (ADX)
        1. Calculate +DM, -DM, and the true range (TR) for each period. Fourteen periods are typically used.
        2. +DM = current high - previous high.
        3. -DM = previous low - current low.
        4. Use +DM when current high - previous high > previous low - current low. Use -DM when previous low - current low > current high - previous high.
        5. TR is the greater of the current high - current low, current high - previous close, or current low - previous close.
        6. Smooth the 14-period averages of +DM, -DM, and TR—the TR formula is below. Insert the -DM and +DM values to calculate the smoothed averages of those.
        7. First 14TR = sum of first 14 TR readings.
        8. Next 14TR value = first 14TR - (prior 14TR/14) + current TR.
        9. Next, divide the smoothed +DM value by the smoothed TR value to get +DI. Multiply by 100.
        10. Divide the smoothed -DM value by the smoothed TR value to get -DI. Multiply by 100.
        11. The directional movement index (DMI) is +DI minus -DI, divided by the sum of +DI and -DI (all absolute values). Multiply by 100.
        12. To get the ADX, continue to calculate DX values for at least 14 periods. Then, smooth the results to get ADX.
        13. First ADX = sum 14 periods of DX / 14.
        14. After that, ADX = ((prior ADX * 13) + current DX) / 14.
        (Source: Investopedia)
        """

        cp_data = self.copy()

        cp_data["DX"+str(days)] = MomentumIndicator(cp_data).DX(days=days, inplace=True,
                                                                  high_price_col_name=high_price_col_name, low_price_col_name=low_price_col_name)["DX"+str(days)]

        lst = [pd.NA for _ in range(len(cp_data))]
        lst[days*2-2] = cp_data["DX"+str(days)].to_numpy()[days-1:days*2-1].mean()


        for i in range(days*2-1, len(cp_data)):
            lst[i] = (lst[i-1]*(days-1) + cp_data["DX"+str(days)].to_numpy()[i])/days

        cp_data["ADX"+str(days)] = np.array(lst)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="ADX"+str(days))


    def BOP(
        self,
        days: int = 14,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        opening_price_col_name: str = "open",
        high_price_col_name: str = "high",
        low_price_col_name: str = "low",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the Balance of Power feature.

        Parameters
        ----------
        days : int
            the number of days included in BOP calculation.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        closing_price_col_name : str, default 'close'
            the colume name of the closing price feature.
        opening_price_col_name : str, default 'open'
            the colume name of the opening price feature.
        high_price_col_name : str, default 'high'
            the colume name of the high price feature.
        low_price_col_name : str, default 'low'
            the colume name of the low price feature.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including 'BOP+str(days)' colume
        
        Note
        -------
        The Balance of Power (BOP) indicator uses price to measure buying and selling pressure. 
        It determines the strength of the buyers and sellers by looking at how strongly the price 
        has changed, rather than using volume. Zero-line crossovers of the BOP indicator can be 
        used as a signal for trend reversals. (Source: StockCharts)
        """

        cp_data = self.copy()
        cp_data["close_open"] = cp_data[closing_price_col_name] - \
            cp_data[opening_price_col_name]
        cp_data["high_low"] = cp_data[high_price_col_name] - \
            cp_data[low_price_col_name]

        cp_data["BOP"+str(days)] = (cp_data["close_open"] /
                                     cp_data["high_low"]).rolling(days).mean()
        cp_data.drop(columns=["close_open", "high_low"], inplace=True)

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="BOP"+str(days))


    def CCI(
        self,
        days: int = 14,
        inplace: bool = False,
        closing_price_col_name: str = "close",
        high_price_col_name: str = "high",
        low_price_col_name: str = "low",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the Commodity Channel Index​ feature.

        Parameters
        ----------
        days : int
            the number of days included in CCI calculation.
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
            a dataframe including 'CCI+str(days)' colume
        
        Note
        -------
        The Commodity Channel Index​ (CCI) is a momentum-based oscillator used to help determine 
        when an investment vehicle is reaching a condition of being overbought or oversold. (Source: Investopedia)
        """

        cp_data = self.copy()
        cp_data = MomentumIndicator(cp_data).typical_price(closing_price_col_name=closing_price_col_name,
                                                             high_price_col_name=high_price_col_name, low_price_col_name=low_price_col_name, inplace=True)

        cp_data["CCI"+str(days)] = (cp_data["typical_price"] - cp_data["typical_price"].rolling(days).mean()) / (1e-10 +
            0.015 * MomentumIndicator(cp_data).mad(days=days, closing_price_col_name=closing_price_col_name)["mad"+str(days)])
        cp_data.drop(columns=["typical_price"])

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name="CCI"+str(days))


    def MACD(
        self,
        first_period_EMA: int = 12,
        second_period_EMA: int = 26,
        inplace: bool = False,
        price_col_name: str = "close",
        dropna: bool = False
    ) -> DataFrame:
        """
        Return the original dataframe with a new colume which is the  moving average convergence divergence feature.

        Parameters
        ----------
        first_period_EMA : int, default 12,
            the first period for calculating EMA.
        first_period_EMA : int, default 26,
            the second period for calculating EMA.
        inplace : bool, default False
            whether to modify the DataFrame rather than creating a new one.
        price_col_name : str, default 'close'
            the colume name of the price feature to be calculated.
        dropna : bool, default False
            whether to drop the nan value in the DataFrame or not.

        Returns
        -------
        DataFrame
            a dataframe including '[price_col_name]_MACD[first_period_EMA]-[second_period_EMA]' colume
        
        Note
        -------
        Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows 
        the relationship between two moving averages of a security’s price. The MACD is calculated by 
        subtracting the 26-period exponential moving average (EMA) from the 12-period EMA. (Source: Investopedia)
        """

        cp_data = self.copy()

        cp_data = MomentumIndicator(cp_data).price_EMA(
            days=12, price_col_name=price_col_name, inplace=True)

        cp_data[price_col_name+"_MACD"+str(first_period_EMA)+'-'+str(second_period_EMA)] = MomentumIndicator(cp_data).price_EMA(days=first_period_EMA, price_col_name=price_col_name, inplace=True)[
            price_col_name+"_price_EMA"+str(first_period_EMA)] - MomentumIndicator(cp_data).price_EMA(days=second_period_EMA, price_col_name=price_col_name, inplace=True)[price_col_name+"_price_EMA"+str(second_period_EMA)]

        return self.return_df(cp_data=cp_data, dropna=dropna, inplace=inplace, drop_col_name=price_col_name+"_MACD"+str(first_period_EMA)+'-'+str(second_period_EMA))