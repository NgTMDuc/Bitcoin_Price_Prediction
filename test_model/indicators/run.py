import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM, Input, RNN, GRU, Bidirectional, concatenate, BatchNormalization, Flatten, SimpleRNN

from tensorflow.keras import initializers
from scipy import stats

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Feature import Feature

import logging
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpus = tf.config.experimental.list_physical_devices('GPU')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

TF_ENABLE_ONEDNN_OPTS = 0

tf.keras.utils.enable_interactive_logging

df = pd.read_csv("data_with_features_ver1.csv")

df = df[df["Timestamp"] >= 1638506000000] # 1609506000000

df.drop(columns=["Time_UTC_Start", "Timestamp", "zeros", "price_up", "price_down", "plus_DM", "minus_DM"], inplace=True)
df.drop(columns=["price_up_SMA4", "price_down_SMA4","price_up_SMA8", "price_down_SMA8","price_up_SMA12", "price_down_SMA12"], inplace=True)
df.drop(columns=["price_up_SMA16", "price_down_SMA16","price_up_SMA24", "price_down_SMA24","price_up_SMA48", "price_down_SMA48"], inplace=True)
df.drop(columns=["smoothed_plus_DM4", "smoothed_minus_DM4", "plus_DI4",
        "minus_DI4", "smoothed_plus_DM8", "smoothed_minus_DM8"], inplace=True)
df.drop(columns=["plus_DI8", "minus_DI8","smoothed_plus_DM12", "smoothed_minus_DM12","plus_DI12", "minus_DI12"], inplace=True)
df.drop(columns=["RS4", "RSI4", "MFI8", "MFI4", "RS8", "MFI12", "RSI8"], inplace=True)


a = df.copy() - df.copy().shift(1)
df2 = a/df
df2["label"] = df["label"].copy()
df = df2
df.reset_index(drop=True)
df = df.iloc[1:,:]

# for col in df.columns:
#     print(col, (np.isfinite(df[col])==False).sum())


# df.replace([np.inf, -np.inf], 0, inplace=True)
# df.replace([np.inf], 10, inplace=True) 
# df.replace([-np.inf], -10, inplace=True) 
# print((np.isfinite(df) == False).sum().sum())

# Convert all dataframes to numpy
data_np = df.copy().to_numpy()

# Number of steps
steps = 5

# Make copies of test_data and train_data
data_np_cpy = data_np.copy()
label_np = data_np[:,10].copy()

# MinMaxScaler for training data, we take only {steps} row before to apply MinMaxScaler() in these rows
scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_np[0:steps])

for i in range(steps, data_np.shape[0]):
    if i % steps == 0:
        # Recalculate the scaler
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_np_cpy[i-steps:i])
    data_np[i] = scaler.transform(data_np_cpy[i].reshape(1, -1))

data_np = np.asarray(data_np).astype('float32')
data_np[:,10] = label_np

RNNs = {
    "GRU": GRU,
    "LSTM": LSTM,
    "RNN": SimpleRNN
}


def split_sequences(data, window_size, label_col_idx):
    x = []
    y = []
    for i in range(0, len(data)-window_size):
        x.append(data[i:i+window_size, :])  # Take window_size rows data before
        
        # To predict the current value of 'difference' column
        y.append(data[i+window_size, label_col_idx])
    return np.array(x), np.array(y)


def build_model(num_of_main_layers, name_of_RNN):
    model = Sequential()

    for _ in range(num_of_main_layers-1):
        model.add(RNNs[name_of_RNN](num_of_outputs, input_shape=(
            steps, data_np.shape[1]), return_sequences=True, dropout=0.1, recurrent_initializer=initializers.RandomNormal(stddev=0.01)))

    model.add(RNNs[name_of_RNN](num_of_outputs, input_shape=(
        steps, data_np.shape[1]), return_sequences=False, dropout=0.1, recurrent_initializer=initializers.RandomNormal(stddev=0.01)))

    model.add(Dense(1, activation='sigmoid'))

    return model


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=5,
                                                verbose=1, mode='auto', restore_best_weights=True)

callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=5,
                                                verbose=1, mode='auto', restore_best_weights=True)

num_of_outputs = 5

training_period = 500

threshold_down=0.5
threshold_up=0.5

def pre_label(x):
    global threshold_up
    global threshold_down
    if x > threshold_up and x < 0.9:
        return 1.
    if x < threshold_down and x > 0.1:
        return 0.
    return -1.


train_data_np, test_data_np = split_sequences(data_np, steps, df.columns.get_loc('label'))
total = 0
rate = 0.2

for name_of_RNN in ["LSTM", "GRU", "RNN"]:

    for num_of_main_layers in range(2, 5):

        for retrain_period in [100, 50, 20, 10]:

            model_name = str(num_of_main_layers) + name_of_RNN + \
                str(num_of_outputs) + ".h5"
            
            logging.basicConfig(level=logging.DEBUG, filename="log-%s.log" % (model_name), filemode="a+",
                                format="%(asctime)-15s %(levelname)-8s %(message)s")            
            
            logging.info('Build model %s...' % (model_name))

            model = build_model(num_of_main_layers, name_of_RNN)

            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            model.fit(train_data_np[0:10], test_data_np[0:10], batch_size=32, epochs=1)

            logging.info(model.summary())
            
            logging.info('Train...')

            good_predict = 0
            count = 0

            totalModels = 1

            actual_acc = []

            for i in range(0, train_data_np.shape[0]-retrain_period, retrain_period):
                models = []
                logging.info(
                    "Step:" + str(i) + "/" + str(train_data_np.shape[0]-retrain_period))
                for _ in range(totalModels):
                    print("Model " + str(_+1) + ":")
                    
                    model = build_model(num_of_main_layers, name_of_RNN)

                    model.compile(
                        loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy']
                        # run_eagerly=True
                    )
                    
                    next_i = i + training_period

                    if train_data_np.shape[0]-1-retrain_period < i+training_period:
                        next_i = min(train_data_np.shape[0]-1-retrain_period, i+training_period)
                        i = train_data_np.shape[0]-1-2*retrain_period-training_period

                    print(i, next_i)
                    
                    history = model.fit(train_data_np[i:next_i, :], test_data_np[i:next_i], batch_size=32,
                                        validation_data=(train_data_np[next_i-retrain_period:next_i, :], test_data_np[next_i-retrain_period:next_i]), epochs=1000, callbacks=[callback])

                    a = model.predict(
                        train_data_np[i:next_i, :])


                    tmp_mean = a.mean()
                    tmp_std = a.std()

                    logging.info("Train data characteristisc:")
                    logging.info(str(tmp_mean) + " " + str(tmp_std) + " " + str(tmp_mean/tmp_std))

                    a = model.predict(
                        train_data_np[next_i:next_i+retrain_period, :])

                    logging.info("\nTest data characteristisc:")
                    logging.info(str(np.mean(a)) + " " + str(np.std(a)) + " " + str(np.mean(a)/np.std(a)))
                    
                    models.append(model)
                    # plt.plot(a)
                    # plt.show()
                            
                            # maxx = 0
                            # t_maxx = 0

                            # for t in np.arange(threshold_down, threshold_up, 0.001):
                            #     threshold = t
                            #     a = model.predict(
                            #         train_data_np[i:next_i, :])
                            #     c = sum((np.vectorize(pre_label)
                            #             (a.reshape((a.shape[0]))) == test_data_np[i:next_i]))
                            #     if c > maxx:
                            #         t_maxx = t
                            #         maxx = c
                            # threshold_down = maxx
                            # threshold_up = maxx
                b = model.predict(
                        train_data_np[i:next_i-retrain_period, :]).reshape(-1)
                a = model.predict(train_data_np[next_i-retrain_period:next_i, :]).reshape(-1)

                threshold_down = np.quantile(b, rate)
                threshold_up = np.quantile(b, (1-rate))

                tmp_result = 0

                m,n = stats.ks_2samp(a, b)
                if n > 0.8:
                    threshold_down = np.quantile(b, rate)
                    threshold_up = np.quantile(b, (1-rate))
                m,n = stats.ks_2samp(a, b, alternative="greater")
                if n > 0.8:
                    threshold_down = np.quantile(b, rate)
                    threshold_up = 1
                m,n = stats.ks_2samp(a, b, alternative="less")
                if n > 0.8:
                    threshold_down = 0
                    threshold_up = np.quantile(b, (1-rate))

                for model in models[:-1]:
                    a += model.predict(
                        train_data_np[next_i:next_i+retrain_period, :])
                
                
                a/=totalModels
                print(a)
                check=False
                if retrain_period - sum((np.vectorize(pre_label)(a.reshape((a.shape[0]))) == -1.)) != 0:
                    check=True
                    tmp_result = sum((np.vectorize(pre_label)(a.reshape((a.shape[0]))) == test_data_np[next_i:next_i+retrain_period]))/(retrain_period - sum((np.vectorize(pre_label)(a.reshape((a.shape[0]))) == -1.)))
                    good_predict += sum((np.vectorize(pre_label)(
                        a.reshape((a.shape[0]))) == test_data_np[next_i:next_i+retrain_period]))
                    total += (retrain_period - sum((np.vectorize(pre_label)(a.reshape((a.shape[0]))) == -1.)))
                        
                if check == True:
                    actual_acc.append(tmp_result)
                    logging.info("Current accuraccy: " + str(tmp_result))

                if total != 0:
                    logging.info("Last acc: " + str(good_predict/total))
                logging.info("Total predictions: " + str(total))
            
            if count != 0:
                logging.info("Overall acc:" + str(good_predict/count))
            else:
                logging.info("Overall acc: no information")

            # Plot
            plt.figure(figsize=(16, 9))
            plt.plot(actual_acc, label="Overall test accuracy")
            pd.DataFrame(actual_acc).to_csv("actual_acc_%s_%s.csv" % (retrain_period, model_name))
            # plt.plot(history.history['val_accuracy'], label="Test acc")
            plt.legend()
            plt.xlabel("Steps")
            plt.ylabel("Accuracy")

            plt.savefig("retrain_period_%s_%s.png" %
                        (retrain_period, model_name))