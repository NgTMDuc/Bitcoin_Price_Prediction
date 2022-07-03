import numpy as np

def noise_filter(decomp_dict: dict, std_range: int=5):
    '''
    Return a dictionary containing the residuals of every columns after removing noises
    and fill NaN values with foward fill
    Parameters:
        decomp_dict: a dictionary containing original residual values of columns
        std_range: number of std away from mean that a value will be consider as noise
    '''
    decomp_dict_copy = decomp_dict.copy()
    for col, decomp in decomp_dict_copy.items():
        decomp_dropna = decomp.dropna()
        std = decomp_dropna.std()
        mean = decomp_dropna.mean()
        decomp[abs(decomp.values - mean) > std_range * std] = np.nan
        decomp = decomp.fillna(method='ffill')
        decomp_dict_copy[col] = decomp
       
    return decomp_dict_copy
       