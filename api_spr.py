from dateutil.relativedelta import relativedelta
from datetime import date
import requests
from datetime import datetime
# import schedule
import time
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


oil_list = []
date_list = []
dt = datetime.now()
current_day = dt.date()
print('Datetime is:', current_day)
weekday = dt.weekday()
print('Day of a week is:', weekday)

# url = "https://api.eia.gov/v2/petroleum/sum/sndw/data/?api_key=M1TKk38zIbimr79GC7ZYRGk9Ermw1p2fuMkF8uER&frequency=weekly&data[0]=value&facets[series][]=WCSSTUS1&start=2022-03-03&end=2023-04-07&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"


defore_day = date.today() - relativedelta(weeks=+55)
print(defore_day)

list_oil = []
url = "https://api.eia.gov/v2/petroleum/sum/sndw/data/?frequency=weekly&data[0]=value&facets[process][]=SAS&facets[duoarea][]=NUS&facets[product][]=EPC0&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=18&api_key=M1TKk38zIbimr79GC7ZYRGk9Ermw1p2fuMkF8uER"

response = requests.get(url)

# print(response.json()['response']['data'][0]['value'])
# df = pd.read_excel('1.xlsx')
df = pd.read_csv('SPR.csv')
# BB_DS.csv
# df = pd.read_csv('2.csv')
Predict_Case = ['production']
print(df)
df.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)
dataset = df.filter([Predict_Case[0]])
dataset1 = dataset[::-1]
print("dataset1", dataset1)

for i in range(0, 18, 1):
    # print(response.json()['response']['data'][i]['value'])
    # print(response.json()['response']['data'][i]['period'])
    production = response.json()['response']['data'][i]['value']
    date = response.json()['response']['data'][i]['period']
    oil_list.append({
        'date': date,
        'production': production
    })
    # print(i)
    # print('11111111')

Predict_Case = ['production']
days_to_predict = 30


def data_split(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        x.append(a)
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)


df_oil = pd.DataFrame(oil_list, columns=['date', 'production'])
df_oil.set_index('date', inplace=True)
df_oil.index = pd.to_datetime(df_oil.index)
oil_dataset = df_oil.filter([Predict_Case[0]])
dataset2 = oil_dataset[::-1]
print(dataset2)

print(dataset2.shape)
# all = oil_dataset[:]
# print(all.shape)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(dataset1)
all = scaler.transform(dataset2)
model = load_model('model_spr.h5')
look_back = 18
X_data1 = all
X_all, Y_all = data_split(all, look_back=look_back)
print(X_data1)


def change(datax1, pred):
    for j in range(look_back-1):
        datax1[0, 0, j] = datax1[0, 0, j+1]
    datax1[0, 0, look_back-1] = pred
    return datax1


future_predict = []
for x in X_all:
    pred = model.predict(x.reshape(1, 1, look_back))
    pred = np.array(pred).flatten()
    future_predict.append(pred)
all_predict_flatten = np.array(scaler.inverse_transform(
    np.array(future_predict))).flatten().astype('int')
# all_predict_flatten = np.absolute(all_predict_flatten)
ypoints = all_predict_flatten
# xpoints = np.array([1, 2, 3, 4])

# plt.plot(xpoints, ypoints)
# plt.show()
print('Future Prediction up to ' + str(days_to_predict) +
      ' days Based on ' + ' :', all_predict_flatten)


# def job(t):
#     print("I'm working...", str(datetime.now()), t)
# for i in ["05:00"]:
#     schedule.every().tuesday.at(i).do(job, i)
# while True:
#     schedule.run_pending()
#     time.sleep(30)
