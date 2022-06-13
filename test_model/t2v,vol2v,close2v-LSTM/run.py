import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
TF_ENABLE_ONEDNN_OPTS=0

res = []

file_path = "data_with_features_ver01.csv"
df = pd.read_csv(file_path)

for c in ["close", "high", "open", "low", "raw_money_flow", "volume", "typical_price", "number_of_trades", "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]:
    df[c] = df[c] - df[c].shift(1)

# Test
lst = [4,8,12,16,20,24,48,72]
for i in range(len(lst)-1):
    for j in range(i+1, len(lst)):
        df["diff_price_SMA%d-%d"%(lst[i],lst[j])] = (df["price_SMA%d"%lst[i]] - df["price_SMA%d"%lst[j]])
        df["diff_volume_SMA%d-%d"%(lst[i],lst[j])] = (df["volume_SMA%d"%lst[i]] - df["volume_SMA%d"%lst[j]])
        df["diff_price_EMA%d-%d"%(lst[i],lst[j])] = (df["price_EMA%d"%lst[i]] - df["price_EMA%d"%lst[j]])
        df["diff_volume_EMA%d-%d"%(lst[i],lst[j])] = (df["volume_EMA%d"%lst[i]] - df["volume_EMA%d"%lst[j]])
        df["diff_price_WMA%d-%d"%(lst[i],lst[j])] = (df["price_WMA%d"%lst[i]] - df["price_WMA%d"%lst[j]])
        df["diff_volume_WMA%d-%d"%(lst[i],lst[j])] = (df["volume_WMA%d"%lst[i]] - df["volume_WMA%d"%lst[j]])

df.drop(columns=["Unnamed: 0", "TR", "plus_DM", "minus_DM", "price_diff", "label", "PPO12-26"],inplace=True)
for i in lst:
    df.drop(columns=["ADX%d"%(i), "BOP%d"%i, "price_SMA%d"%i, "price_EMA%d"%i, "price_WMA%d"%i, "minus_DI%d"%(i), "MFI%d"%i, "plus_DI%d"%i, "ATR%d"%(i), "CCI%d"%i, "DX%d"%i, "mad%d"%(i), "smoothed_minus_DM%d"%i, "smoothed_plus_DM%d"%i, "volume_SMA%d"%i, "volume_EMA%d"%i, "volume_WMA%d"%i], inplace=True)


df["day"] = pd.to_datetime(df["Time_UTC_Start"]).dt.day/31
df["month"] = pd.to_datetime(df["Time_UTC_Start"]).dt.month/12
df["year"] = pd.to_datetime(df["Time_UTC_Start"]).dt.year/2018
df["hour"] = pd.to_datetime(df["Time_UTC_Start"]).dt.hour/24

df.drop(columns=["Time_UTC_Start"], inplace=True)

df.dropna(inplace=True)

def split_sequences(data, window_size, price_col_idx):
    x = []
    y = []
    for i in range(0,len(data)-window_size):
        x.append(data[i:i+window_size,:]) # Take window_size rows data before
        y.append(data[i+window_size,price_col_idx]) # To predict the current value of 'difference' column
    return np.array(x), np.array(y)

for steps in [i for i in range(3,20)]:
    print("Steps = " + str(steps))

    # Convert all dataframes to numpy
    train_df = df.iloc[0:int(len(df)*0.7),:].copy()
    train_data = train_df.to_numpy()
    num_rows_train, num_cols_train = train_data.shape

    val_df = df.iloc[int(len(df)*0.7):int(len(df)*0.8),:].copy()
    val_data = val_df.to_numpy()
    num_rows_val, num_cols_val = val_data.shape

    test_df = df.iloc[int(len(df)*0.8):].copy()
    test_data = test_df.to_numpy()
    num_rows_test, num_cols_test = test_data.shape


    # Make copies of test_data and train_data
    test_data_cpy = test_data.copy()
    train_data_cpy = train_data.copy()
    val_data_cpy = val_data.copy()

    # StandardScaler for training data, we take only {steps} row before to apply StandardScaler() in these rows
    for i in range(steps, num_rows_train):
        scaler = StandardScaler().fit(train_data_cpy[i-steps:i])
        train_data[i] = scaler.transform(train_data_cpy[i].reshape(1, -1))
        del scaler

    # # StandardScaler for validation data, we take only {steps} row before to apply StandardScaler() in these rows
    for i in range(steps, num_rows_val):
        scaler = StandardScaler().fit(val_data_cpy[i-steps:i])
        val_data[i] = scaler.transform(val_data_cpy[i].reshape(1, -1))
        del scaler

    train_data = train_data[steps+1:]
    train_df = train_df.iloc[steps+1:,:]

    for i in range(num_rows_test):
        if i > steps:
            scaler = StandardScaler().fit(test_data_cpy[i-steps:i])
            test_data[i] = scaler.transform(test_data_cpy[i].reshape(1, -1))
            del scaler
        else:
            scaler = StandardScaler().fit(np.array(train_data_cpy[-steps:].tolist() + test_data_cpy[:i].tolist()))
            test_data[i] = scaler.transform(test_data_cpy[i].reshape(1, -1))
            del scaler

    train_data = np.asarray(train_data).astype('float32')
    val_data = np.asarray(val_data).astype('float32')
    test_data = np.asarray(test_data).astype('float32')

    x_train, y_train = split_sequences(train_data, steps, df.columns.get_loc("close"))
    x_val, y_val = split_sequences(val_data, steps, df.columns.get_loc("close"))
    x_test, y_test = split_sequences(test_data, steps, df.columns.get_loc("close"))

    num_of_outputs = 16


    print('Build model %s...')


    class LSTM4_with_t2v(Model):

        def __init__(self, vol_idx, close_idx):
            super().__init__()
            self.vol_idx = vol_idx
            self.close_idx = close_idx

            self.t2v_first_col = Dense(1, activation=None)
            self.t2v_others_col = Dense(15, activation=None)

            self.vol2v_first_col = Dense(1, activation=None)
            self.vol2v_others_col = Dense(15, activation=None)

            self.close2v_first_col = Dense(1, activation=None)
            self.close2v_others_col = Dense(15, activation=None)

            self.LSTM1 = LSTM(
                num_of_outputs, return_sequences=True, recurrent_dropout=0.4)
            self.LSTM2 = LSTM(
                num_of_outputs, return_sequences=True, recurrent_dropout=0.4)
            self.LSTM3 = LSTM(
                num_of_outputs, return_sequences=False, recurrent_dropout=0.4)

            self.batch_norm = BatchNormalization()
            self.out = Dense(1)

        def call(self, inputs):
            t2v_x1 = self.t2v_first_col(inputs[:, :, :])
            t2v_x2 = tf.sin(self.t2v_others_col(inputs[:, :, :]))
            t2v = tf.concat([t2v_x1, t2v_x2], -1)

            vol2v_x1 = self.vol2v_first_col(
                inputs[:, :, self.vol_idx:self.vol_idx+1])
            vol2v_x2 = tf.sin(self.vol2v_others_col(
                inputs[:, :, self.vol_idx:self.vol_idx+1]))
            vol2v = tf.concat([vol2v_x1, vol2v_x2], -1)

            close2v_x1 = self.close2v_first_col(
                inputs[:, :, self.close_idx:self.close_idx+1])
            close2v_x2 = tf.sin(self.close2v_others_col(
                inputs[:, :, self.close_idx:self.close_idx+1]))
            close2v = tf.concat([close2v_x1, close2v_x2], -1)

            new_input = tf.concat([inputs[:, :, :-4], t2v], -1)
            new_input = tf.concat([new_input, vol2v], -1)
            new_input = tf.concat([new_input, close2v], -1)

            x1 = self.LSTM1(new_input)
            x1 = self.batch_norm(x1)

            x2 = self.LSTM2(x1)
            x2 = self.batch_norm(x2)

            x3 = self.LSTM3(x1)

            return self.out(x3)


    model = LSTM4_with_t2v(df.columns.get_loc(
        "volume"), df.columns.get_loc("close"))

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    model.compile(loss='mean_absolute_error', optimizer=opt)

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3,
                            verbose=1, mode='auto', restore_best_weights=True)

    # First test fit:
    print("First fit:")
    model.fit(x_train, y_train, batch_size=256,
            epochs=1, validation_data=(x_val, y_val))
    print("Done!")

    print('Train...')

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    model.compile(loss='mean_absolute_error', optimizer=opt)

    for i in range(1,3):
        print("Load %i:"%i)
        model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_val, y_val), callbacks=[monitor])

    
    print('Train...')

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.0001
    )

    model.compile(loss='mean_absolute_error', optimizer=opt)

    for i in range(1,2):
        print("Load %i:"%i)
        model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_val, y_val), callbacks=[monitor])


    y_pred = model.predict(x_test).reshape(-1)
    pred_data = [0 for _ in range(y_pred.shape[0])]
    test_df_close = test_df["close"][steps:]

    for i in range(y_pred.shape[0]):
        if i > steps:
            scaler = StandardScaler().fit(test_df_close[i-steps:i].to_numpy().reshape(-1,1))
            pred_data[i] = float(scaler.inverse_transform(y_pred[i].reshape(1, -1)).reshape(-1))
            del scaler
        else:
            scaler = StandardScaler().fit(pd.concat((train_df["close"][-steps:],test_df_close[:i])).values.reshape((-1,1)))
            pred_data[i] = float(scaler.inverse_transform(y_pred[i].reshape(1, -1)).reshape(-1))
            del scaler

    y_pred = pd.DataFrame(pred_data)
    y_test = pd.DataFrame(test_df_close.to_numpy())
        
    acc = ((y_pred*y_test)>=0).sum()/(y_test.shape[0])

    plt.figure(figsize=(16,9),dpi=90)
    plt.plot(y_pred[:100], label="predict")
    plt.plot(y_test[:100], label="real")
    plt.legend()
    plt.savefig(str(steps)+".jpg")

    res += [(steps, float(acc))]
    print(float(acc))
    pd.DataFrame(res).to_csv("acc.csv")