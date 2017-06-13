import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import math

from keras.models import Sequential

from keras.layers.recurrent import LSTM
from keras import  regularizers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

matplotlib.style.use('ggplot')

# Stocks information 176-1
# date,open,high,low,close,volume
dataset_st = pd.read_csv('ts1.csv', usecols=[1,2,3,4,5],
                          engine='python')
# Stock+tweets
# date,tweets,pos,neg,neutr,open,high,low,volume,close
df = pd.read_csv('full_tesla.csv',
                        usecols=[0,1,2,3,4,5,6,7,8,9],
                        engine='python')

# correlation for stocks+tweets
dfTS = pd.DataFrame(df)
corr_tw = dfTS.corr()

print corr_tw

# stocks and tweets visualization
close_price = df.ix[:, 'close'].tolist()
tweets = df.ix[:, 'tweets'].tolist()
close_price = [(np.array(x) - np.mean(x)) / np.std(x) for x in close_price]
tweets = [(np.array(x) - np.mean(x)) / np.std(x) for x in tweets]
plt.plot(tweets, color='black', label = 'Amount of tweets')
plt.plot(close_price, color='green', label = 'Close price')
plt.show()

# LEARNING
dataset = df.values
X = dataset[:,1:9]
Y = dataset[:,9]

#X = [(np.array(x) - np.mean(x)) / np.std(x) for x in X]

print X
print Y

def build_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='linear'))
    model.add(Dense(6, kernel_initializer='normal', activation='linear'))
    model.add(Dense(1, kernel_initializer='normal'))

    #Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
model = build_model()
history = model.fit(X, Y,
                    batch_size=128,
                    epochs=200,
                    callbacks=[reduce_lr],
                    validation_split=0.15)

predicted = model.predict(np.array(X))
original = Y

plt.plot(original, color='black', label = 'Original data')
plt.plot(predicted, color='blue', label = 'Predicted data')
plt.legend(loc='best')
plt.title('Actual and predicted')
plt.show()


#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
