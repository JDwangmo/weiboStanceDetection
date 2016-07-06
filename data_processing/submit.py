#encoding=utf8
from __future__ import print_function

__author__ = 'jdwang'
__date__ = 'create date: 2016-07-05'
__email__ = '383287471@qq.com'

from data_util import DataUtil
final_test_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/NLPCC2016_Stance_Detection_Task_A_Testdata.txt'
final_test_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/data_processing/result/TaskA_all_testdata_15000.csv'

# 去进行分类的句子
final_test_classify_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskA_all_testdata_14966.csv'
clasify_result_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/cnn/randEmbedding_cnn/result/temp.csv'



data_util = DataUtil()

final_test_data = data_util.load_data(final_test_file_path)
print(final_test_data.head())
print(final_test_data.shape)
# final_test_data = final_test_data[[]]
print(final_test_data[final_test_data['WORDS'].isnull()].shape)
final_test_data = final_test_data[final_test_data['WORDS'].notnull()]
data_util.save_data(final_test_data,'result/TaskA_all_testdata_15000_A.csv')
# print(final_test_data.tail())
# print(final_test_data.sort_values(by=['ID']).tail())

quit()

final_test_classify_data = data_util.load_data(final_test_classify_file_path)
clasify_result_data = data_util.load_data(clasify_result_file_path)
final_test_classify_data[u'STANCE'] = clasify_result_data[u'PREDICT']
print(final_test_classify_data.head())

final_test_classify_data[u'WORDS'] = final_test_classify_data[u'TEXT'].apply(data_util.segment_sentence)
# print(final_test_data.head())

# print(final_test_classify_data[u'﻿ID'])
# print(final_test_data[u'﻿ID'])
# print(final_test_classify_data[u'WORDS'])
# final_test_data.loc[final_test_data['WORDS'].isnull(),'STANCE'] = 'NONE'
data_util.save_data(final_test_classify_data,'result/TaskA_all_testdata_15000.csv')

print(final_test_classify_data.shape)
quit()
quit()
print(clasify_result_data.shape)
print(final_test_classify_data.head())
final_test_classify_data['STANCE'] = clasify_result_data['PREDICT']
print(final_test_classify_data.head())
print(final_test_data.columns)
quit()

# print(final_test_data[final_test_data['WORDS'].isnull()].shape)
# print(final_test_data[final_test_data['STANCE']=='NONE'].shape)
# print(final_test_data[final_test_data['WORDS'].isnull()])

# data_util.save_data(final_test_data[final_test_data['WORDS'].isnull()],'result/null_data.csv')
data_util.save_data(final_test_data,'result/TaskA_all_testdata_15000.csv')
print(final_test_data.head())
print(final_test_data.shape)
