import pandas as pd

from data.data_normalization import data_normalization
from seasonal_decompose import seasonal_decompose

# Read files
df_with_features = pd.read_csv("data/data_with_features_ver01.csv")
df_raw = pd.read_csv("data/raw_binance_btc_1h.csv")

df_with_features.drop(columns=["Unnamed: 0"], inplace=True)

# Get the residual and the seasonal_and_trend components
df_raw_resid = pd.DataFrame(seasonal_decompose.seasonal_decompose_get_resid(x=df_raw.drop(columns=["Time_UTC_Start"]), period=7))
df_raw_seasonal_and_trend = pd.DataFrame(seasonal_decompose.seasonal_decompose_get_seasonal_and_trend(x=df_raw.drop(columns=["Time_UTC_Start"]), period=7))

# Normalize data
df_raw_seasonal_and_trend = data_normalization(df=df_raw_seasonal_and_trend, range=20)
df_raw_resid = data_normalization(df=df_raw_resid, range=20)
df_with_features = data_normalization(df=df_with_features, range=20)

df_raw_resid["Time_UTC_Start"] = df_raw["Time_UTC_Start"]
df_raw_seasonal_and_trend["Time_UTC_Start"] = df_raw["Time_UTC_Start"]

print(df_raw_resid.head(35))
print(df_raw_seasonal_and_trend.head(35))

# Dropna
first_valid_index = max(df_raw_resid.first_valid_index(), df_raw_seasonal_and_trend.first_valid_index())

df_raw_resid = df_raw_resid[df_raw_resid.index >= first_valid_index]
df_raw_seasonal_and_trend = df_raw_seasonal_and_trend[df_raw_seasonal_and_trend.index >= first_valid_index]

print(df_raw_resid.head(5), df_raw_resid.shape)
print(df_raw_seasonal_and_trend.head(5), df_raw_seasonal_and_trend.shape)

# Cast to float32
df_raw_resid.loc[:,df_raw_resid.columns!="Time_UTC_Start"].astype('float32')
df_raw_seasonal_and_trend.loc[:,df_raw_seasonal_and_trend.columns!="Time_UTC_Start"].astype('float32')
df_with_features.loc[:,df_with_features.columns!="Time_UTC_Start"].astype('float32')

# Merging
df_resid_with_features = pd.merge(df_raw_resid, df_with_features, on='Time_UTC_Start', how='inner')
df_seasonal_and_trend_with_features = pd.merge(df_raw_seasonal_and_trend, df_with_features, on='Time_UTC_Start', how='inner')

print(df_resid_with_features.head(5), df_raw_resid.shape)
print(df_seasonal_and_trend_with_features.head(5), df_raw_seasonal_and_trend.shape)