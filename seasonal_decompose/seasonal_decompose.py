from pandas import DataFrame
from numpy.typing import ArrayLike

from statsmodels.tsa.seasonal import seasonal_decompose as sm

def seasonal_decompose (x: DataFrame, model: str='additive', filt:ArrayLike=None, period:int=None, extrapolate_trend:int=0):
    '''
    Return a dictionary of statsmodels.tsa.seasonal.DecomposeResult for each column
    + Key = column name of the dataframe
    + Value = statsmodels.tsa.seasonal.DecomposeResult corresponding to the column name in key
    '''
    res = {}
    for col in x.columns:
        res[col] =  sm(x[col], model, filt, period, False, extrapolate_trend)
    return res


def seasonal_decompose_get_resid (x: DataFrame, model: str='additive', filt:ArrayLike=None, period:int=None, extrapolate_trend:int=0):
    '''
    Return a dictionary of residual result in pandas.core.series.Series after decomposing x for each column
    + Key = column name of the dataframe
    + Value = residual result corresponding to the column name in key
    '''
    res = {}
    for col in x.columns:
        res[col] = sm(x[col], model, filt, period, False, extrapolate_trend).resid
    return res


def seasonal_decompose_get_seasonal (x: DataFrame, model: str='additive', filt:ArrayLike=None, period:int=None, extrapolate_trend:int=0):
    '''
    Return a dictionary of seasonal result in pandas.core.series.Series after decomposing x for each column
    + Key = column name of the dataframe
    + Value = seasonal result corresponding to the column name in key
    '''
    res = {}
    for col in x.columns:
        res[col] = sm(x[col], model, filt, period, False, extrapolate_trend).seasonal
    return res


def seasonal_decompose_get_trend (x: DataFrame, model: str='additive', filt:ArrayLike=None, period:int=None, extrapolate_trend:int=0):
    '''
    Return a dictionary of trend result in pandas.core.series.Series after decomposing x for each column
    + Key = column name of the dataframe
    + Value = trend result corresponding to the column name in key
    '''
    res = {}
    for col in x.columns:
        res[col] = sm(x[col], model, filt, period, False, extrapolate_trend).trend
    return res


def seasonal_decompose_get_seasonal_and_trend (x: DataFrame, model: str='additive', filt:ArrayLike=None, period:int=None, extrapolate_trend:int=0):
    '''
    Return a dictionary of seasonal and trend combined result in pandas.core.series.Series after decomposing x for each column
    + Key = column name of the dataframe
    + Value = seasonal and trend combined result corresponding to the column name in key
    '''
    res = {}
    for col in x.columns:
        tmp = sm(x[col], model, filt, period, False, extrapolate_trend)
        res[col] = tmp.seasonal + tmp.trend
    return res