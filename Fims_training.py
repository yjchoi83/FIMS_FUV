import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import sys,csv
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler

from keras.layers import Dense, Dropout, Activation,LeakyReLU,PReLU
from sklearn import preprocessing
from keras.models import Sequential
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
from keras import initializers

from scipy.optimize import curve_fit
from keras import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from math import  log

dt = datetime.datetime.now()
dt = dt.strftime("%m%d%H%M%S")

##### 1. loading files #####
# training data loading
data_set_train = pd.read_csv('./dataset/train_log.csv',header=None)
data_set_train = pd.DataFrame(data_set_train)
data_set_train = data_set_train.values[:]

# test data loading (log scale)
data_set_test = pd.read_csv('./dataset/test_log_all.csv',header=None)
data_set_test = pd.DataFrame(data_set_test)
data_set_test = data_set_test.values[:]

# test data loading (raw)
data_set_test_raw = pd.read_csv('./dataset/test_raw_all.csv',header=None)
data_set_test_raw = pd.DataFrame(data_set_test_raw)
data_set_test_raw = data_set_test_raw.values[:]

##### 2. data set #####
all_train = data_set_train[:,1:9]
all_test = data_set_test[:,1:9]

# define variable types
all_train = all_train.astype('float32')
all_test = all_test.astype('float32')

##### 3. normalization #####
scaler = StandardScaler()
scaler.fit(all_train)
scaled_all_train = scaler.transform(all_train)
scaled_x_train = scaled_all_train[:,1:8]	# seven input parameters
scaled_y_train = scaled_all_train[:,0]		# target parameter (FIMS observation)

# application of nomalization to the test set
scaled_all_test = scaler.transform(all_test)
scaled_x_test = scaled_all_test[:,1:8]		# seven input parameters

##### 4. modeling #####
# three layers: tanh + relu + relu
model = Sequential()
model.add(Dense(16, kernel_initializer='RandomUniform', activation='tanh', input_shape=(7,)))
model.add(Dropout(0.03))
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.03))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(1))

# model compile: loass function = mean squared error
model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.0001), metrics=[metrics.mae])

# training
history = model.fit(scaled_x_train, scaled_y_train, epochs=1000)

##### 5. plottng result #####
# accuracy of training
plt.figure(1)
plt.title('training accuracy')
plt.plot(history.history['mean_absolute_error'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

##### 6. inverse transformation #####
scaled_y_pred = model.predict(scaled_x_test)
scaled_all_test[:,0] = list(scaled_y_pred.flatten())
all_pred = scaler.inverse_transform(scaled_all_test)
y_pred = all_pred[:,0]
y_pred = list(y_pred.flatten())

##### 7. converting log-scale into continuum unit #####
y_pred = np.exp(y_pred)/1000

##### 8. saving result #####
result = open('./dataset/result.csv', 'w', encoding="utf-8")
# data order: pixel_number + fims_observation + fims_prediction + gal_lon + gal_lat + rass1 + rass2 + H_alpha + E(B-V) + N(HI) + galex_oservation
for i in range(len(y_pred)):
	result.write(str(data_set_test[i,0]) +',' +str(data_set_test_raw[i,1]) +',' +str(y_pred[i]) + ',' + str(data_set_test[i,2]) + ',' + str(data_set_test[i,3])+ ',' + str(data_set_test_raw[i,4]) +
	',' + str(data_set_test_raw[i,5]) + ',' + str(data_set_test_raw[i,6])+ ',' + str(data_set_test_raw[i,7]) + ',' + str(data_set_test_raw[i,8]) +','  + str(data_set_test[i,9]) + '\n')
result.close()
