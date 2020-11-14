import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import sys,csv
import datetime

##### reading allsky file #####
data_set_train = pd.read_csv('./dataset/allskymap_raw.csv',header=None)
data_set_train = pd.DataFrame(data_set_train)
data_set_train = data_set_train.values[1:]

index = data_set_train[:,0]
l = data_set_train[:,1]
b = data_set_train[:,2]
non_fims = data_set_train[:,3]
non_galex =data_set_train[:,4]
non_rass1 =data_set_train[:,5]
non_rass2 = data_set_train[:,6]
non_ha = data_set_train[:,7]
non_ebv = data_set_train[:,8]
non_hi = data_set_train[:,9]

l = l.astype('float32')
b = b.astype('float32')
non_fims = non_fims.astype('float32')
non_galex = non_galex.astype('float32')
non_rass1 = non_rass1.astype('float32')
non_rass2 = non_rass2.astype('float32')
non_ha = non_ha.astype('float32')
non_ebv = non_ebv.astype('float32')
non_hi = non_hi.astype('float32')

##### dividing into training set and test set #####
# raw data files
result_train = open('./dataset/train_raw.csv', 'w', encoding="utf-8")
result_pred = open('./dataset/test_raw.csv', 'w', encoding="utf-8")
result_pred_all = open('./dataset/test_raw_all.csv', 'w', encoding="utf-8")

# log scale conversion files
result_train_log = open('./dataset/train_log.csv', 'w', encoding="utf-8")
result_pred_log = open('./dataset/test_log.csv', 'w', encoding="utf-8")
result_pred_all_log = open('./dataset/test_log_all.csv', 'w', encoding="utf-8")

##### dividing data sets and saving #####
# data order: pixel_number + fims_observation + fims_prediction + gal_lon + gal_lat + rass1 + rass2 + H_alpha + E(B-V) + N(HI) + galex_oservation
for i in range(len(non_fims)):

	if str(non_rass1[i]) != 'nan' and str(non_rass2[i]) != 'nan' and str(non_ha[i]) != 'nan' and str(non_ebv[i]) != 'nan' and str(non_hi[i]) != 'nan' :

		# entire data set (raw)
		result_pred_all.write(str(index[i]) +',' +str(non_fims[i]) + ',' + str(l[i]) + ',' + str(b[i])+ ',' + str(non_rass1[i]) +
		',' + str(non_rass2[i]) + ',' + str(non_ha[i])+ ',' + str(non_ebv[i]) + ',' + str(non_hi[i]) +','  + str(non_galex[i]) + '\n')

		# test set
		if str(non_fims[i]) == 'nan':
			result_pred.write(str(index[i]) +',' +str(non_fims[i]) + ',' + str(l[i]) + ',' + str(b[i])+ ',' + str(non_rass1[i]) +
			',' + str(non_rass2[i]) + ',' + str(non_ha[i])+ ',' + str(non_ebv[i]) + ',' + str(non_hi[i]) +','  + str(non_galex[i]) + '\n')
			result_pred_log.write(str(index[i]) +',' +str(non_fims[i]) + ',' + str(l[i]) + ',' + str(b[i])+ ',' + str(np.log(10*non_rass1[i])) +
			',' + str(np.log(10*non_rass2[i])) + ',' + str(np.log(10000*non_ha[i]))+ ',' + str(np.log(100000*non_ebv[i])) + ',' + str(np.log(non_hi[i]/1e16)) +','  + str(non_galex[i]) + '\n')
			# entire data set (log)
			result_pred_all_log.write(str(index[i]) +',' +str(non_fims[i]) + ',' + str(l[i]) + ',' + str(b[i])+ ',' + str(np.log(10*non_rass1[i])) +
			',' + str(np.log(10*non_rass2[i])) + ',' + str(np.log(10000*non_ha[i]))+ ',' + str(np.log(100000*non_ebv[i])) + ',' + str(np.log(non_hi[i]/1e16)) +','  + str(non_galex[i]) + '\n')
		# training set
		else:
			result_train.write(str(index[i]) +',' +str(non_fims[i]) + ',' + str(l[i]) + ',' + str(b[i])+ ',' + str(non_rass1[i]) +
			',' + str(non_rass2[i]) + ',' + str(non_ha[i])+ ',' + str(non_ebv[i]) + ',' + str(non_hi[i]) +','  + str(non_galex[i]) + '\n')
			result_train_log.write(str(index[i]) +',' +str(np.log(1000*non_fims[i])) + ',' + str(l[i]) + ',' + str(b[i])+ ',' + str(np.log(10*non_rass1[i])) +
			',' + str(np.log(10*non_rass2[i])) + ',' + str(np.log(10000*non_ha[i]))+ ',' + str(np.log(100000*non_ebv[i])) + ',' + str(np.log(non_hi[i]/1e16)) +','  + str(non_galex[i]) + '\n')
			# entire data set (log)
			result_pred_all_log.write(str(index[i]) +',' +str(np.log(1000*non_fims[i])) + ',' + str(l[i]) + ',' + str(b[i])+ ',' + str(np.log(10*non_rass1[i])) +
			',' + str(np.log(10*non_rass2[i])) + ',' + str(np.log(10000*non_ha[i]))+ ',' + str(np.log(100000*non_ebv[i])) + ',' + str(np.log(non_hi[i]/1e16)) +','  + str(non_galex[i]) + '\n')

# flie closing
result_pred_all_log.close()
result_pred_log.close()
result_train_log.close()

result_pred_all.close()
result_pred.close()
result_train.close()
