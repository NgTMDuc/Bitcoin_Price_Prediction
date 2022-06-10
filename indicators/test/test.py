from Feature import Feature
from utils.label import label


# Dataframe with 15 minutes between each row
df = label("binance-BTCUSDT-1m.csv")


# We train the model with a half of df_15min
# df = df.iloc[int(len(df)*0):, :]


a = Feature(df.copy())
a.price_WMA(days=4, inplace=True)
print(a.columns)
print(a[["price_WMA4", "price"]].tail(10))