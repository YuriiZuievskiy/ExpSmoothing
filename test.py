# comends by #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras import models, layers

from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel(r"C:\Users\38050\Downloads\Dataset_augmented.xlsx", sheet_name='Sheet1')
data_full = data[0].values  # we will work only with first column
data_full = data_full[::-1]  # reverse vector so that older values go first

lookback = 2  # we will use two previous values to predict current value
train_size = 15  # we leave 15 values for training and 4 last values for test
# but we actually need 6 for test with first to also in train
# because we need those two to predict first of actually test values
data_train, data_test = data_full[:train_size+1], data_full[train_size-lookback:]

# scale data to range [0,1]
scaler = MinMaxScaler()
data_train_scaled = scaler.fit_transform(data_train.reshape((-1, 1))).flatten()
data_test_scaled = scaler.transform(data_test.reshape((-1, 1))).flatten()

# reshape data to be a collection of sequences of length 2
X_train = np.vstack([data_train_scaled[i - lookback:i] for i in range(lookback, data_train_scaled.shape[0])]).reshape(
    (-1, lookback, 1))
y_train = data_train_scaled[lookback:].reshape((-1, 1))

X_test = np.vstack([data_test_scaled[i - lookback:i] for i in range(lookback, data_test_scaled.shape[0])]).reshape(
    (-1, lookback, 1))
y_test = data_test_scaled[lookback:].reshape((-1, 1))

# create model
model = models.Sequential()

model.add(layers.LSTM(units=50, return_sequences=True, input_shape=X_train.shape[1:]))
model.add(layers.LSTM(units=50, return_sequences=True))
model.add(layers.LSTM(units=50))
model.add(layers.Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
# train model
model.fit(X_train, y_train, epochs=500, batch_size=4, use_multiprocessing=True)

# plot models predictions on train and test sets
predict_train = scaler.inverse_transform(model.predict(X_train)).flatten()
predict_test = scaler.inverse_transform(model.predict(X_test)).flatten()
plt.figure(figsize=(12, 6))
plt.plot(range(data_train.shape[0]), data_train, label='train data')
plt.plot(range(data_train.shape[0] - 1, data_full.shape[0]), data_test[lookback:], label='test data')
plt.plot(range(lookback, data_train.shape[0]), predict_train, linestyle='--', linewidth=3, c='red',
         label='train prediction')
plt.plot(range(data_train.shape[0] - 1, data_full.shape[0]), predict_test, linestyle='--', linewidth=3, c='green',
         label='test prediction')
plt.xticks(range(0, len(data_full), 2), range(1, len(data_full) + 1, 2))
plt.legend()
plt.title(f'Test set MSE: {round(model.evaluate(X_test, y_test, verbose=0), 3)}')
plt.show()

