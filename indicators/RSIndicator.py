from pandas import DataFrame
from MAIndicator import MAIndicator
from pandas._typing import Axes, Dtype


class RSIndicator(MAIndicator):    
    
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
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
            
        price_up_data = \
            RSIndicator(cpy_data).price_up_SMA(days=days, inplace=False, closing_price_col_name=closing_price_col_name)["price_up_SMA"+str(days)]
        price_down_data = \
            RSIndicator(cpy_data).price_down_SMA(days=days, inplace=False, closing_price_col_name=closing_price_col_name)["price_down_SMA"+str(days)]
        
        cpy_data["RS"+str(days)] = price_up_data / (0.001 - price_down_data) # 0.001 is used for avoiding zero division
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="RS"+str(days))
    
        
    def RSI(
        self, 
        days: int, 
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        cpy_data_RS = RSIndicator(cpy_data).RS(days=days, inplace=False, closing_price_col_name=closing_price_col_name)["RS"+str(days)]
        cpy_data["RSI"+str(days)] = 100 - 100/(1 + cpy_data_RS.to_numpy())
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="RSI"+str(days))
    
    
    # Indicators deduced from MA
    def PPO(
        self, 
        first_period_EMA: int=12, 
        second_period_EMA: int=26,
        inplace: bool=False, 
        closing_price_col_name: str="Close",
        dropna: bool=False
    ) -> DataFrame:
        
        cpy_data = self.copy()
        cpy_data_first_EMA = RSIndicator(cpy_data).EMA(days=first_period_EMA, inplace=False, closing_price_col_name=closing_price_col_name)["EMA"+str(first_period_EMA)]
        cpy_data_second_EMA = RSIndicator(cpy_data).EMA(days=second_period_EMA, inplace=False, closing_price_col_name=closing_price_col_name)["EMA"+str(second_period_EMA)]
        cpy_data["PPO%s-%s"%(str(first_period_EMA), str(second_period_EMA))] = \
            100 * (cpy_data_first_EMA - cpy_data_second_EMA) / cpy_data_second_EMA
        
        return self.return_df(cpy_data=cpy_data, dropna=dropna, inplace=inplace, drop_col_name="PPO%s-%s" % (str(first_period_EMA), str(second_period_EMA)))
