from pandas import DataFrame

def data_normalization(df: DataFrame, range: int=20, include_cur_row: bool=False):
    '''
    Return the normalized data:
    + df: the dataframe to be normalized
    + range: the number of previous rows (or including the current row) to be considered in the normalization
    + include_cur_row: True if we consider the current row in the normalization process (calculate mean and std
    using the current row and (range-1) previous rows), False if we want to use all the passed data for normalization 
    processing ((calculate mean and std using (range) previous rows))
    '''
    
    df_roll = None
    if include_cur_row == False:
        df_roll = df.rolling(range, closed='left')
    else:
        df_roll = df.rolling(range)
    res_df = (df - df_roll.mean()) / df_roll.std()
    return res_df