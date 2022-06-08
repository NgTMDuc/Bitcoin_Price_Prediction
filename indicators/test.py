from Feature import Feature
from utils.label import label


# Dataframe with 15 minutes between each row
df_15min = label("../binance-BTCUSDT-1m.csv")


# We train the model with a half of df_15min
df = df_15min.iloc[int(len(df_15min)*0):, :]


a = Feature(df.copy())
a.MACD(inplace=True)
print(a.columns)
print(a[["MACD12-26"]].head(28))