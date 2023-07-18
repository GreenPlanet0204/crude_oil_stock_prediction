import requests
from datetime import datetime
import schedule
import time
oil_list = []
date_list = []
dt = datetime.now()
current_day = dt.date()
print('Datetime is:', current_day)
weekday = dt.weekday()
print('Day of a week is:', weekday)

# url = "https://api.eia.gov/v2/petroleum/sum/sndw/data/?api_key=M1TKk38zIbimr79GC7ZYRGk9Ermw1p2fuMkF8uER&frequency=weekly&data[0]=value&facets[series][]=WCSSTUS1&start=2022-03-03&end=2023-04-07&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"

from datetime import date
from dateutil.relativedelta import relativedelta

defore_day = date.today() - relativedelta(weeks=+55)
print(defore_day)

list_oil = []
url = "https://api.eia.gov/v2/petroleum/sum/sndw/data/?api_key=M1TKk38zIbimr79GC7ZYRGk9Ermw1p2fuMkF8uER&frequency=weekly&data[0]=value&facets[series][]=WCSSTUS1&start=" + str(defore_day) + "&end="  + str(current_day) +  "&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"

response  = requests.get(url)

# print(response.json()['response']['data'][0]['value'])

for i in range(0, 54, 1):
    # print(response.json()['response']['data'][i]['value'])
    # print(response.json()['response']['data'][i]['period'])
    production = response.json()['response']['data'][i]['value']
    date = response.json()['response']['data'][i]['period']
    oil_list.append(production)
    date_list.append(date)
    # print(i)
    # print('11111111')

print(oil_list)
print(date_list)








# def job(t):
#     print("I'm working...", str(datetime.now()), t)
# for i in ["05:00"]:
#     schedule.every().tuesday.at(i).do(job, i)
# while True:
#     schedule.run_pending()
#     time.sleep(30)




