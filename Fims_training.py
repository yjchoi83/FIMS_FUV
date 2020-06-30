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
#from keras.layers.advanced_activations import PReLU, LeakyReLU
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

# data train loading
data_set_train = pd.read_csv('./dataset/train_log_lb_final.csv',header=None)
data_set_train = pd.DataFrame(data_set_train)
data_set_train = data_set_train.values[1:]


# data test loading
data_set_test = pd.read_csv('./dataset/test_log_lb_final.csv',header=None)
data_set_test = pd.DataFrame(data_set_test)
data_set_test = data_set_test.values[1:]

#data observed loading
obj_set_test = pd.read_csv('./dataset/f_g_union.csv',header=None)
obj_set_test = pd.DataFrame(obj_set_test)
obj_set_test = obj_set_test.values[1:]


# data set 

X_train = data_set_train[:,2:9]
y_train= data_set_train[:,1]

X_test = data_set_test[:,2:9]
y_galex = data_set_test[:,9]

obs_galex = obj_set_test[:,1]
obs_fims = obj_set_test[:,0]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_galex = y_galex.astype('float32')
obs_galex = obs_galex.astype('float32')
obs_fims = obs_fims.astype('float32')

# # normalization
scaler = StandardScaler()
scaler_train = scaler.fit(X_train)
X_train = scaler_train.fit_transform(X_train)

scaler_test = scaler.fit(X_test)
X_test = scaler_test.transform(X_test)

# model
model = Sequential()
model.add(Dense(16, kernel_initializer='RandomUniform', activation='tanh', input_shape=(7,)))
model.add(Dropout(0.3))
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(1))




# model compile 
model.compile(  loss='mean_squared_error',    optimizer=RMSprop(lr=0.0001),   metrics=[metrics.mae])

# training
history = model.fit(X_train, y_train,epochs=20)

# # result 1 : accuracy of training
plt.figure(1)
plt.title('training accuracy')
plt.plot(history.history['mean_absolute_error'])
#score = model.evaluate(X_test, y_test)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# training
y_pred = model.predict(X_test)


# # # return output value before scaling and result save
# inv_pred = np.zeros(shape=(len(y_pred), 5) )
# inv_pred[:,0] = y_pred[:,0]
# #inv_pred[:,:] = y_test[:,:]
# y_pred = scaler_test.inverse_transform(inv_pred)[:,0]

y_pred = list(y_pred.flatten())



#galex_final = galex_final
#
#
# for i in range(len(y_galex)):
#
#     if str(y_galex[i]) == 'nan':
#         continue
#     else:
#         y_pred_final.append(y_pred[i])
#         galex_final.append(float(y_galex[i]))

y_pred = np.exp(y_pred)/1000
#result = open('./dataset/result/result_{}.csv'.format(dt), 'w', encoding="utf-8")
result = open('./dataset/result/result_mk.csv', 'w', encoding="utf-8")
for i in range(len(y_pred)):
    result.write(str(y_pred[i]) + ',' + str(y_galex[i]) + '\n')
result.close()

# y_pred_final = np.exp(y_pred)
# galex_final = np.array(galex_final) *(1e-3)

#
# xmin = min(y_pred)
# xmax = max(y_pred)
# ymin = min(float(y_galex))
# ymax = max(float(y_galex))

#x_range = np.arange(0.0, 10000.0, len(y_pred))
#
# temp = pd.read_csv('./dataset/f_g_union.csv',header=None)
# temp = pd.DataFrame(temp)
# temp = temp.values[1:]
#

def fitFunc(x,a,b,c):
    return a*np.power(x,b)+c

pred_fitParams,pred_fitCovariance = curve_fit(fitFunc,y_pred,y_galex)
print("predicted parameter",pred_fitParams)
print("predicted covariance", pred_fitCovariance)
pred_sigma = [pred_fitCovariance[0,0],pred_fitCovariance[1,1],pred_fitCovariance[2,2]]

obs_fitParams,obs_fitCovariance = curve_fit(fitFunc,obs_fims,obs_galex)
print("observed parameter",obs_fitParams)
print("observed covariance", obs_fitCovariance)
obs_sigma = [obs_fitCovariance[0,0],obs_fitCovariance[1,1],obs_fitCovariance[2,2]]



x = np.arange(0,max(y_pred))
# # result 2 : true vs predict
plt.figure(2)
plt.title('Objected vs predict')
#plt.scatter(y_pred,y_galex)
plt.scatter(obs_fims,obs_galex,marker ='+',color='red',s=3)
plt.scatter(y_pred,y_galex,marker ='x',color='blue',s=3)
#plt.plot(x,x)
plt.xlabel("Predicted FIMS")
plt.ylabel("GALEX")
plt.yscale('log')
plt.xscale('log')
# plt.plot(obs_fims,fitFunc(obs_fims,*obs_fitParams),'r-',label ="({0:.3f}*x**{1:.3f}) + {2:.3f}".format(*obs_fitParams))
# plt.plot(y_pred,fitFunc(y_pred,*pred_fitParams),'b-',label ="({0:.3f}*x**{1:.3f}) + {2:.3f}".format(*pred_fitParams))
plt.plot(obs_fims,fitFunc(obs_fims,*obs_fitParams),'r-',label ="({0:.3f}*x**{1:.3f}+ {2:.3f}),R:{0:2f} ".format(*obs_fitParams,*obs_fitCovariance))
plt.plot(y_pred,fitFunc(y_pred,*pred_fitParams),'b-',label ="({0:.3f}*x**{1:.3f}+ {2:.3f}),R:{0:2f} ".format(*pred_fitParams,*pred_fitCovariance))
plt.legend(loc='lower right')
plt.show()
plt.savefig('./dataset/result/scatter_mk.png'.format(dt))
plt.close()



loss = []

for i in range(len(y_pred)):
      loss.append(abs(float(y_pred[i]) - float(y_galex[i])) )

print('error = ', sum(loss) / len(y_pred))

