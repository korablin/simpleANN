import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import math

from keras.models import Sequential

from keras.layers.recurrent import LSTM
from keras import  regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

from sklearn import preprocessing

matplotlib.style.use('ggplot')

# Stocks information
# date,open,high,low,volume,close
df_st = pd.read_csv('data/ts1.csv', usecols=[1,2,3,4,5],
                    engine='python')
# Stock+tweets
# date,tweets,pos,neg,neutr,open,high,low,volume,close
df = pd.read_csv('data/full_tesla.csv',
                 usecols=[0,1,2,3,4,5,6,7,8,9],
                 engine='python')

# correlation for stocks+tweets
dfTS = pd.DataFrame(df)
corr_tw = dfTS.corr()

print corr_tw

# stocks and tweets visualization
volume = df.ix[:, 'volume'].tolist()
tweets = df.ix[:, 'tweets'].tolist()
vol = preprocessing.scale(np.array(volume))
tw = preprocessing.scale(np.array(tweets))

plt.plot(tw, color='black', label = 'Amount of tweets')
plt.plot(vol, color='green', label = 'Traded volume')
plt.legend(loc='best')
plt.title('Number of tweets and traded volume for TSLA')
plt.show()

# LEARNING
dataset = df.values
X = dataset[:,1:9]
Y = dataset[:,9]

min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(np.array(X))
Y_scaled = min_max_scaler.fit_transform(np.array(Y))

def build_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='linear'))
    model.add(BatchNormalization())
    model.add(Dense(6, kernel_initializer='normal', activation='linear'))
    model.add(Dense(1, kernel_initializer='normal'))

    #Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
model = build_model()
history = model.fit(X_scaled, Y_scaled,
                    batch_size=128,
                    epochs=150,
                    callbacks=[reduce_lr],
                    validation_split=0.1)

predicted = model.predict(np.array(X_scaled))
original = Y_scaled

plt.plot(original, color='black', label = 'Original price')
plt.plot(predicted, color='blue', label = 'Predicted price')
plt.legend(loc='best')
plt.title('Actual and predicted')
plt.show()

