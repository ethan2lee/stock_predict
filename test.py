# Author: Ethan Lee
# creadte at 2019/4/29
# Email: ethan2lee.coder@gmail.com
# Author: Ethan Lee
# creadte at 2019/4/29
# Email: ethan2lee.coder@gmail.com
import numpy as np
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
X =np.array(X_train)
test=X[-10:]


def generator(data, lookback, delay, min_index, max_index, shuffle=False,

              batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

lookback = 14
step = 6
delay = 12
batch_size = 128
train_gen = generator(X,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(X,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(X,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(X) - 300001 - lookback) // batch_size

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, X.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
 steps_per_epoch=500,
 epochs=20,
 validation_data=val_gen,
 validation_steps=val_steps)

