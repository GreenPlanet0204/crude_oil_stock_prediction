import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# np.random.seed(42)

# df = pd.read_excel('1.xlsx')
df = pd.read_csv('SPR.csv')
# BB_DS.csv
# df = pd.read_csv('2.csv')
Predict_Case = ['production']
df.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)
days_to_predict = 30


def data_split(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        x.append(a)
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)


dataset = df.filter([Predict_Case[0]])
dataset1 = dataset[::-1]


print(dataset1)
# print(dataset)
test_size = int(dataset1.shape[0] * 0.08)
all = dataset1[:]
train = dataset1[:-test_size]
test = dataset1[-test_size:]


scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(dataset1)
all = scaler.transform(dataset1)
train = scaler.transform(train)
test = scaler.transform(test)

look_back = 18
X_all, Y_all = data_split(all, look_back=look_back)
X_train, Y_train = data_split(train, look_back=look_back)
X_test, Y_test = data_split(test, look_back=look_back)

# batch_size = 30
model = Sequential()
# model.add((LSTM(units=30, activation='relu', return_sequences=True, input_shape=(1,look_back))))
model.add(LSTM(units=30, activation='relu',
          return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(units=20))
model.add(Dense(10))
# model.add(Dense(1))
model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='mse', optimizer='adam')
model.summary()

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
# print(X_train.shape)
# x = x.reshape(1, x.shape[0], x.shape[1])
# history = model.fit(X_train, Y_train, epochs=30, batch_size=6, validation_data=(X_test, Y_test), verbose=1, shuffle=False)
history = model.fit(X_train, Y_train, epochs=100, batch_size=16,
                    validation_data=(X_test, Y_test), verbose=1, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

model.save("model_spr"+".h5")

X_data1 = X_train[0:1]


def change(datax1, pred):
    for j in range(look_back-1):
        datax1[0, 0, j] = datax1[0, 0, j+1]
        # print("ssssefef",datax[i])
    datax1[0, 0, look_back-1] = pred
    return datax1


len = len(X_all)
print('length', len)
print('shpae', X_all.shape)
index = len - 110
X = X_all[index:]
print(X[:-1])

future_predict = []
for x in X:

    pred = model.predict(x.reshape((1, 1, look_back)))
    # pred1 = change(X_data1, pred)
    # print(k, "awdwd", pred, "adawdawd")
    pred = np.array(pred).flatten()
    future_predict.append(pred)

all_predict_flatten = np.array(scaler.inverse_transform(
    np.array(future_predict))).flatten().astype('int')

all_predict_flatten = np.absolute(all_predict_flatten)
ypoints = all_predict_flatten
xpoints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

# plt.plot(xpoints, ypoints)
# plt.show()
# print(all_predict_flatten)
print('Future Prediction up to ' + str(days_to_predict) +
      ' days Based on ' + ' :', all_predict_flatten)

model.reset_states()
