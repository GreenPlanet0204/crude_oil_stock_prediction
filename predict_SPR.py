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
from keras.models import load_model

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


# print(dataset1)
# print(dataset)
test_size = int(dataset1.shape[0] * 0.3)
all = dataset1[:]
train = dataset1[:-test_size]
test = dataset1[-test_size:]


scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(dataset1)
all = scaler.transform(dataset1)
print(all)
# print(all)
# train = scaler.transform(train)
# test = scaler.transform(test)

look_back = 18

X_all, Y_all = data_split(all, look_back=look_back)
model = load_model('model_spr.h5')


def change(datax1, pred):
    for j in range(look_back-1):
        datax1[0, 0, j] = datax1[0, 0, j+1]
        # print("ssssefef",datax[i])
    datax1[0, 0, look_back-1] = pred
    return datax1


len = len(X_all)
print('length', len)
print('shpae', X_all.shape)
index = len - 55
X = X_all[-18:]
print(X.shape)
future_predict = []
for x in X:
    # print(x)
    pred = model.predict(x.reshape((1, 1, look_back)))
    # pred1 = change(X_data1, pred)
    # print(k, "awdwd", pred, "adawdawd")
    pred = np.array(pred).flatten()
    future_predict.append(pred)

all_predict_flatten = np.array(scaler.inverse_transform(
    np.array(future_predict))).flatten().astype('int')

all_predict_flatten = np.absolute(all_predict_flatten)
# ypoints = all_predict_flatten
# xpoints = np.array([1, 2, 3])

# plt.plot(xpoints, ypoints)
# plt.show()
# print(all_predict_flatten)
print('Future Prediction up to ' + str(days_to_predict) +
      ' days Based on ' + ' :', all_predict_flatten)

# model.reset_states()
