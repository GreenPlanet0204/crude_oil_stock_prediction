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
df = pd.read_csv('SPR_2.csv')
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
