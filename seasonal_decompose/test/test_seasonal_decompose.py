import pandas as pd
import seasonal_decompose as sm

df = pd.read_csv("test/test_df.csv")
a = sm.seasonal_decompose_get_resid(df,period=14)
print(a.keys())
print(a)
