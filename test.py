# Author: Ethan Lee
# creadte at 2019/4/29
# Email: ethan2lee.coder@gmail.com
# Author: Ethan Lee
# creadte at 2019/4/29
# Email: ethan2lee.coder@gmail.com
import pandas_datareader.data as web
import tushare as ts
import datetime
from sklearn.linear_model import LinearRegression
start = datetime.datetime(2017,1,1)#获取数据的时间段-起始时间
end = datetime.date.today()#获取数据的时间段-结束时间
stock = web.DataReader("601398.SS", "yahoo", start, end)
stock.to_csv("data.csv")


data = ts.get_hist_data("601398")


X_train = []
y_train = []
for i in range(10, data.shape[0]):
    tmp = []
    tmp.append(data.ix[i]['open'])
    tmp.append(data.ix[i]['high'])
    tmp.append(data.ix[i]['close'])
    tmp.append(data.ix[i]['low'])
    X_train.append(tmp)
    tmp = (data.ix[i]['close'] - data.ix[i]['open']) / data.ix[i]['open']
    y_train.append(tmp)

X_test = X_train[:10]

# print(X_train.shape)
# print(y_train.shape)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred =linreg.predict(X_test)

print(y_pred)
