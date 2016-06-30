#encoding=utf8
"""
    微博立场分析baseline，使用CNN-rand模型，具体见: https://github.com/JDwangmo/coprocessor#randomembedding_padding_cnn
    训练数据： train_data/train_data_2090.csv，
    测试数据： train_data/test_data_896.csv
    实验设置：


"""
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-23'
__email__ = '383287471@qq.com'

import numpy as np
import pandas as pd
import logging
import timeit
import yaml
config = yaml.load(file('./config.yaml'))
config = config['randEmbedding_cnn']
verbose = config['verbose']
logging.basicConfig(filename=''.join(config['log_file_path']),filemode = 'w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
start_time = timeit.default_timer()
print('='*30)
print config['describe']
print('='*30)
print 'start running!'
logging.debug('='*30)
logging.debug(config['describe'])
logging.debug('='*30)
logging.debug('start running!')
logging.debug('='*20)

from deep_learning.cnn.randEmbedding_cnn.randEmbedding_cnn_model import RandEmbeddingCNN
from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
from data_processing.data_util import DataUtil
import os


# -------------- region start : 1. 加载训练集和测试集 -------------
if verbose > 2 :
    logging.debug('-' * 20)
    print '-' * 20
    logging.debug('1. 加载训练集和测试集')
    print '1. 加载训练集和测试集'
# -------------- code start : 开始 -------------

data_util = DataUtil()

train_data = data_util.load_data(config['train_data_filte_path'])
test_data = data_util.load_data(config['test_data_filte_path'])

# -------------- code start : 结束 -------------
if verbose > 2 :
    logging.debug('-' * 20)
    print '-' * 20
# -------------- region end : 1. 加载训练集和测试集 ---------------

# -------------- region start : 2. 转换数据格式，以可以进行分类 -------------
if verbose > 2 :
    logging.debug('-' * 20)
    print '-' * 20
    logging.debug('2. 转换数据格式，以可以进行分类')
    print '2. 转换数据格式，以可以进行分类'
# -------------- code start : 开始 -------------

label_to_index = {'FAVOR':0,'AGAINST':1,'NONE':2}
index_to_label = ['FAVOR','AGAINST','NONE']

# 将TARGET和TEXT字段进行拼接
train_X = (train_data['TARGET'] +','+ train_data['TEXT']).as_matrix()
test_X = (test_data['TARGET'] +','+ test_data['TEXT']).as_matrix()

train_y = train_data['STANCE'].map(label_to_index).as_matrix()
test_y = test_data['STANCE'].map(label_to_index).as_matrix()


feature_encoder = FeatureEncoder(train_data=train_X,
                                 sentence_padding_length=config['sentence_padding_length'],
                                 verbose=0,
                                 need_segmented=True,
                                 full_mode=True,
                                 replace_number=True,
                                 remove_stopword=True,
                                 lowercase=True,
                                 )
feature_encoder.print_sentence_length_detail()
feature_encoder.print_model_descibe()

# -------------- code start : 结束 -------------
if verbose > 2 :
    logging.debug('-' * 20)
    print '-' * 20
# -------------- region end : 2. 转换数据格式，以可以进行分类 ---------------

# -------------- region start : 3. 初始化CNN模型并训练 -------------
if verbose > 2 :
    logging.debug('-' * 20)
    print '-' * 20
    logging.debug('3. 初始化CNN模型并训练')
    print '3. 初始化CNN模型并训练'
# -------------- code start : 开始 -------------

model_file_path = ''.join([str(item) for item in config['model_file_path']])

word_embedding_dim = config['word_embedding_dim']
rand_embedding_cnn = RandEmbeddingCNN(
    rand_seed=1337,
    verbose=verbose,
    input_dim=feature_encoder.train_data_dict_size+1,
    word_embedding_dim=word_embedding_dim,
    input_length = config['sentence_padding_length'],
    num_labels = len(label_to_index),
    conv_filter_type = config['conv_filter_type'],
    k=config['kmax_k'],
    embedding_dropout_rate=config['embedding_dropout_rate'],
    output_dropout_rate=config['output_dropout_rate'],
    nb_epoch=int(config['cnn_nb_epoch']),
    earlyStoping_patience = config['earlyStoping_patience'],
)
rand_embedding_cnn.print_model_descibe()

if config['refresh_all_model'] or not os.path.exists(model_file_path):
    # 训练模型
    rand_embedding_cnn.fit((feature_encoder.train_padding_index, train_y),
                           (map(feature_encoder.encoding_sentence, test_X), test_y))
    # 保存模型
    rand_embedding_cnn.save_model(model_file_path)
else:
    # 从保存的pickle中加载模型
    rand_embedding_cnn.model_from_pickle(model_file_path)



# -------------- code start : 结束 -------------
if verbose > 2 :
    logging.debug('-' * 20)
    print '-' * 20
# -------------- region end : 3. 初始化CNN模型并训练 ---------------

# -------------- region start : 4. 预测 -------------
if verbose > 1 :
    logging.debug('-' * 20)
    print '-' * 20
    logging.debug('4. 预测')
    print '4. 预测'
# -------------- code start : 开始 -------------



print index_to_label[rand_embedding_cnn.predict(feature_encoder.encoding_sentence('你好吗'))]

y_pred,is_correct,accu = rand_embedding_cnn.accuracy((map(feature_encoder.encoding_sentence,test_X),test_y))

test_data[u'IS_CORRECT'] = is_correct
test_data[u'PREDICT'] = [index_to_label[item] for item in y_pred]

result_file_path = ''.join(config['result_file_path'])

data_util.save_data(test_data[[u'﻿ID',u'TARGET',u'TEXT',u'STANCE',u'PREDICT',u'IS_CORRECT']],
                    path=result_file_path)



# -------------- code start : 结束 -------------
if verbose > 1 :
    logging.debug('-' * 20)
    print '-' * 20
# -------------- region end : 4. 预测 ---------------




end_time = timeit.default_timer()
print 'end! Running time:%ds!'%(end_time-start_time)
logging.debug('='*20)
logging.debug('end! Running time:%ds!'%(end_time-start_time))