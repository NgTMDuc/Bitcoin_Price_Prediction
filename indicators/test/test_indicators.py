import pandas as pd

from Feature import Feature

a = Feature(pd.read_csv("../data/raw_binance_btc_1h.csv"))

# Get WMA price
a.price_WMA(days=4, inplace=True)

# Print all the columns of a
print(a.columns)

# Print price_WMA4 and close columns
print(a[["price_WMA4", "close"]].tail(10))
