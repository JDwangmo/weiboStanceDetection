#encoding=utf8
from __future__ import print_function

__author__ = 'jdwang'
__date__ = 'create date: 2016-07-03'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
import yaml
config = yaml.load(file('./config.yaml'))
config = config['dcnn']
verbose = config['verbose']
logging.basicConfig(filename=''.join(config['log_file_path']),filemode = 'w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
start_time = timeit.default_timer()
print('='*30)
print(config['describe'])
print('='*30)
print('start running!')
logging.debug('='*30)
logging.debug(config['describe'])
logging.debug('='*30)
logging.debug('start running!')
logging.debug('='*20)


from deep_learning.cnn.dynamic_cnn.dynamic_cnn_model import DynamicCNN
from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
from data_processing.data_util import DataUtil
import os

# -------------- region start : 1. 加载训练集和测试集 -------------

data_util = DataUtil()

train_data,test_data,label_to_index,index_to_label = data_util.load_train_test_data(config)

# -------------- region end : 1. 加载训练集和测试集 ---------------

# -------------- region start : 2. 转换数据格式，以可以进行分类 -------------
if verbose > 2:
    logging.debug('-' * 20)
    print('-' * 20)
    logging.debug('2. 转换数据格式，以可以进行分类')
    print('2. 转换数据格式，以可以进行分类')
# -------------- code start : 开始 -------------


# 将TARGET和TEXT字段进行拼接
train_X = (train_data['TARGET'] + ',' + train_data['TEXT']).as_matrix()
test_X = (test_data['TARGET'] + ',' + test_data['TEXT']).as_matrix()

train_y = train_data['STANCE'].map(label_to_index).as_matrix()
test_y = test_data['STANCE'].map(label_to_index).as_matrix()
feature_encoder = FeatureEncoder(
    sentence_padding_length=config['sentence_padding_length'],
    verbose=0,
    need_segmented=config['need_segmented'],
    full_mode=True,
    remove_stopword=True,
    replace_number=True,
    lowercase=True,
    zhs2zht=True,
    remove_url=True,
    padding_mode='center',
    add_unkown_word=True,
    mask_zero=True
)
train_X_features = feature_encoder.fit_transform(train_data=train_X)
test_X_features = feature_encoder.transform(test_X)

feature_encoder.print_sentence_length_detail
print(feature_encoder.vocabulary_size)
# print ','.join(sorted(feature_encoder.vocabulary))
# quit()
feature_encoder.print_model_descibe()
# -------------- code start : 结束 -------------
if verbose > 2:
    logging.debug('-' * 20)
    print('-' * 20)
# -------------- region end : 2. 转换数据格式，以可以进行分类 ---------------



for seed in config['rand_seed']:

    # -------------- region start : 3. 初始化CNN模型并训练 -------------
    if verbose > 2:
        logging.debug('-' * 20)
        print('-' * 20)
        logging.debug('3. 初始化CNN模型并训练')
        print('3. 初始化CNN模型并训练')
    # -------------- code start : 开始 -------------
    # 设置文件地址
    model_file_path = ''.join([str(item) for item in config['model_file_path']])
    model_file_path = model_file_path%seed

    result_file_path = ''.join([str(item) for item in config['result_file_path']])
    result_file_path = result_file_path % seed


    print(model_file_path)
    print(result_file_path)

    dcnn_model = DynamicCNN(
        rand_seed=seed,
        verbose=config['verbose'],
        batch_size=32,
        vocab_size=feature_encoder.vocabulary_size,
        word_embedding_dim=config['word_embedding_dim'],
        input_length=config['sentence_padding_length'],
        num_labels=len(label_to_index),
        conv_filter_type=[[100, 2, 'full'],
                          [100, 4, 'full'],
                          # [100, 8, 'full'],
                          ],
        ktop=config['ktop'],
        embedding_dropout_rate=0.5,
        output_dropout_rate=0.5,
        nb_epoch=int(config['cnn_nb_epoch']),
        earlyStoping_patience=config['earlyStoping_patience'],
        )
    dcnn_model.print_model_descibe()

    if config['refresh_all_model'] or not os.path.exists(model_file_path):
        # 训练模型
        dcnn_model.fit((train_X_features, train_y),
                       (test_X_features, test_y))
        # 保存模型
        dcnn_model.save_model(model_file_path)
    else:
        # 从保存的pickle中加载模型
        dcnn_model.model_from_pickle(model_file_path)

    # -------------- code start : 结束 -------------
    if verbose > 2:
        logging.debug('-' * 20)
        print('-' * 20)
    # -------------- region end : 3. 初始化CNN模型并训练 ---------------

    # -------------- region start : 4. 预测 -------------
    if verbose > 1:
        logging.debug('-' * 20)
        print('-' * 20)
        logging.debug('4. 预测')
        print('4. 预测')
    # -------------- code start : 开始 -------------



    print(index_to_label[dcnn_model.predict(feature_encoder.transform_sentence('你好吗'))])

    y_pred, is_correct, accu,f1 = dcnn_model.accuracy((map(feature_encoder.transform_sentence, test_X), test_y))
    logging.debug('F1(macro)为：%f'%(np.average(f1[:-1])))
    print('F1(macro)为：%f' % (np.average(f1[:-1])))
    test_data[u'IS_CORRECT'] = is_correct
    test_data[u'PREDICT'] = [index_to_label[item] for item in y_pred]
    # data_util.save_data(test_data,'tmp.tmp')
    # quit()

    data_util.save_data(test_data,
                        path=result_file_path)

    # -------------- code start : 结束 -------------
    if verbose > 1:
        logging.debug('-' * 20)
        print('-' * 20)
    # -------------- region end : 4. 预测 ---------------






end_time = timeit.default_timer()
print('end! Running time:%ds!'%(end_time-start_time))
logging.debug('='*20)
logging.debug('end! Running time:%ds!'%(end_time-start_time))