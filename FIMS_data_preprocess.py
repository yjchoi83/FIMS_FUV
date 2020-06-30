import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import sys,csv
import datetime

# temp_a = pd.read_csv('./dataset/test_log_final.csv',header = None)
# temp_a = pd.DataFrame(temp_a)
# temp_a = temp_a.values[0:]
#
# temp_b = pd.read_csv('./dataset/temp_com.csv',header = None)
# temp_b = pd.DataFrame(temp_b)
# temp_b = temp_b.values[0:]
#
# a = temp_a[:,0]
# b = temp_b[:,0:3]
#
# result = open('./dataset/result/temp.csv', 'w', encoding="utf-8")
#
# for i in range(len(a)):
#     for j in range(len(b)):
#         if str(a[i]) == str(b[j]):
#             result.write(str(b[j,0]) + ','+ str(b[j,1]) + ',' + str(b[j,2]) + '\n')
#
#         else:
#             continue
#
# result.close()

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
non_euv = data_set_train[:,8]
non_hi = data_set_train[:,9]
#
#
#
#result_train = open('./dataset/train_raw.csv', 'w', encoding="utf-8")
result_pred = open('./dataset/test_raw_new.csv', 'w', encoding="utf-8")
# for i in range(len(non_fims)):
#
#     if str(non_fims[i]) == 'nan' :
#         continue
#     else:
#         result_train.write(str(index[i])+','+str(non_fims[i]) + ',' + str(non_galex[i])+',' + str(non_rass1[i]) +
#         ',' + str(non_rass2[i])+',' + str(non_ha[i])+',' + str(non_euv[i])+',' + str(non_hi[i])+ '\n')
#
# #
for i in range(len(non_fims)):

    # if str(non_fims[i]) == 'nan' :
    #     result_pred.write(str(index[i]) +',' + str(non_fims[i])+ ',' + str(non_galex[i]) + ',' + str(non_rass1[i]) +
    #                           ',' + str(non_rass2[i]) + ',' + str(non_ha[i]) + ',' + str(non_euv[i]) + ',' + str(
    #             non_hi[i]) + '\n')

    if str(non_fims[i]) == 'nan' :
        result_pred.write(str(index[i]) +',' +str(non_fims[i]) + ',' + str(l[i]) + ',' + str(b[i])+ ',' + str(non_rass1[i]) +
                              ',' + str(non_rass2[i]) + ',' + str(non_ha[i])+ ',' + str(non_euv[i]) + ',' + str(
                non_hi[i]) +','  + str(non_galex[i]) + '\n')


result_pred.close()
# result_pred.close()

#
# data_set_train = pd.read_csv('./dataset/allskymap_raw.csv',header=None)
# data_set_train = pd.DataFrame(data_set_train)
# data_set_train = data_set_train.values[1:]
#
# test_galex = open('./dataset/test_final.csv', 'w', encoding="utf-8")
#
# temp = pd.read_csv('./dataset/test_raw.csv',header=None)
# temp = pd.DataFrame(temp)
# temp = temp.values[0:]
#
# index = temp[:,0]
# non_fims = temp[:,1]
# non_galex =temp[:,2]
# non_rass1 =temp[:,3]
# non_rass2 = temp[:,4]
# non_ha = temp[:,5]
# non_euv = temp[:,6]
# non_hi = temp[:,7]
#
# for i in range(len(non_galex)):
#
#     if str(non_galex[i]) == 'nan' :
#         continue
#
#     else:
#         test_galex.write(str(index[i]) +',' + str(non_fims[i])+ ',' + str(non_galex[i]) + ',' + str(non_rass1[i]) +
#             ',' + str(non_rass2[i]) + ',' + str(non_ha[i]) + ',' + str(non_euv[i]) + ',' + str(
#                 non_hi[i]) + '\n')
#
# test_galex.close()


#
# union = open('./dataset/f_g_union.csv', 'w', encoding="utf-8")
#
# temp = pd.read_csv('./dataset/fims_galex_union.csv',header=None)
# temp = pd.DataFrame(temp)
# temp = temp.values[0:]
#
#
# non_fims = temp[:,0]
# non_galex =temp[:,1]
#
# for i in range(len(non_galex)):
#
#     if str(non_fims[i]) != 'nan' and str(non_galex[i]) != 'nan':
#         union.write(str(non_fims[i]) + ','+ str( non_galex[i]) + '\n')
#
# union.close()

