# encoding=utf8
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
logging.basicConfig(filename=''.join(config['log_file_path']), filemode='w',
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
start_time = timeit.default_timer()
print('=' * 30)
print config['describe']
print('=' * 30)
print 'start running!'
logging.debug('=' * 30)
logging.debug(config['describe'])
logging.debug('=' * 30)
logging.debug('start running!')
logging.debug('=' * 20)

from deep_learning.cnn.wordEmbedding_cnn.wordEmbedding_cnn_model import WordEmbeddingCNN
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
    print '-' * 20
    logging.debug('2. 转换数据格式，以可以进行分类')
    print '2. 转换数据格式，以可以进行分类'
# -------------- code start : 开始 -------------

# 将TARGET和TEXT字段进行拼接
train_X = (train_data['TARGET'] + ',' + train_data['TEXT']).as_matrix()
test_X = (test_data['TARGET'] + ',' + test_data['TEXT']).as_matrix()

train_y = train_data['STANCE'].map(label_to_index).as_matrix()
test_y = test_data['STANCE'].map(label_to_index).as_matrix()

feature_encoder = FeatureEncoder(train_data=train_X,
                                 sentence_padding_length=config['sentence_padding_length'],
                                 verbose=0,
                                 need_segmented=config['need_segmented'],
                                 full_mode=True,
                                 replace_number=True,
                                 remove_stopword=True,
                                 lowercase=True,
                                 padding_mode='center',
                                 add_unkown_word=True,
                                 mask_zero=True,
                                 zhs2zht=True,
                                 remove_url=True,
                                 )

train_X_feature = feature_encoder.train_padding_index
test_X_feature = map(feature_encoder.encoding_sentence, test_X)

feature_encoder.print_sentence_length_detail()
print feature_encoder.train_data_dict_size
# print ','.join(sorted(feature_encoder.vocabulary))
# quit()
feature_encoder.print_model_descibe()
# -------------- code start : 结束 -------------
if verbose > 2:
    logging.debug('-' * 20)
    print '-' * 20
# -------------- region end : 2. 转换数据格式，以可以进行分类 ---------------



for seed in config['rand_seed']:

    # -------------- region start : 3. 初始化CNN模型并训练 -------------
    if verbose > 2:
        logging.debug('-' * 20)
        print '-' * 20
        logging.debug('3. 初始化CNN模型并训练')
        print '3. 初始化CNN模型并训练'
    # -------------- code start : 开始 -------------
    # 设置文件地址
    model_file_path = ''.join([str(item) for item in config['model_file_path']])
    model_file_path = model_file_path%seed

    result_file_path = ''.join([str(item) for item in config['result_file_path']])
    result_file_path = result_file_path % seed

    train_cnn_feature_file_path = ''.join([str(item) for item in config['train_cnn_feature_file_path']])
    train_cnn_feature_file_path = train_cnn_feature_file_path % seed

    test_cnn_feature_file_path = ''.join([str(item) for item in config['test_cnn_feature_file_path']])
    test_cnn_feature_file_path = test_cnn_feature_file_path % seed

    print model_file_path
    print result_file_path
    print train_cnn_feature_file_path
    print test_cnn_feature_file_path


    rand_embedding_cnn = WordEmbeddingCNN(
        rand_seed=seed,
        verbose=verbose,
        optimizers=config['optimizers'],
        input_dim=feature_encoder.train_data_dict_size + 1,
        word_embedding_dim=config['word_embedding_dim'],
        input_length=config['sentence_padding_length'],
        num_labels=len(label_to_index),
        conv_filter_type=config['conv_filter_type'],
        k=config['kmax_k'],
        embedding_dropout_rate=config['embedding_dropout_rate'],
        output_dropout_rate=config['output_dropout_rate'],
        nb_epoch=int(config['cnn_nb_epoch']),
        earlyStoping_patience=config['earlyStoping_patience'],
    )
    rand_embedding_cnn.print_model_descibe()

    if config['refresh_all_model'] or not os.path.exists(model_file_path):
        # 训练模型
        rand_embedding_cnn.fit((train_X_feature, train_y),
                               (test_X_feature, test_y))
        # 保存模型
        rand_embedding_cnn.save_model(model_file_path)
    else:
        # 从保存的pickle中加载模型
        rand_embedding_cnn.model_from_pickle(model_file_path)

    # -------------- code start : 结束 -------------
    if verbose > 2:
        logging.debug('-' * 20)
        print '-' * 20
    # -------------- region end : 3. 初始化CNN模型并训练 ---------------

    # -------------- region start : 4. 预测 -------------
    if verbose > 1:
        logging.debug('-' * 20)
        print '-' * 20
        logging.debug('4. 预测')
        print '4. 预测'
    # -------------- code start : 开始 -------------

    print index_to_label[rand_embedding_cnn.predict(feature_encoder.encoding_sentence('你好吗'))]

    y_pred, is_correct, accu,f1 = rand_embedding_cnn.accuracy((map(feature_encoder.encoding_sentence, test_X), test_y))
    logging.debug('F1(macro)为：%f'%(np.average(f1[:-1])))
    print 'F1(macro)为：%f'%(np.average(f1[:-1]))
    test_data[u'IS_CORRECT'] = is_correct
    test_data[u'PREDICT'] = [index_to_label[item] for item in y_pred]
    # data_util.save_data(test_data,'tmp.tmp')
    # quit()
    data_util.save_data(test_data,
                        path=result_file_path)


    # -------------- region start : 生成深度特征编码 -------------
    if verbose > 1 :
        logging.debug('-' * 20)
        print('-' * 20)
        logging.debug('生成深度特征编码')
        print('生成深度特征编码')
    # -------------- code start : 开始 -------------



    # -------------- code start : 结束 -------------
    if verbose > 1 :
        logging.debug('-' * 20)
        print('-' * 20)
    # -------------- region end : 生成深度特征编码 ---------------
    if config['genernate_cnn_feature']:
        train_cnn_feature = np.asarray(map(rand_embedding_cnn.get_conv1_feature,train_X_feature))
        test_cnn_feature = np.asarray(map(rand_embedding_cnn.get_conv1_feature,test_X_feature))
        print train_cnn_feature.shape
        print test_cnn_feature.shape
        np.savetxt(train_cnn_feature_file_path,
                   train_cnn_feature,
                   fmt='%.5f',
                   delimiter=',')
        np.savetxt(test_cnn_feature_file_path,
                   test_cnn_feature,
                   fmt='%.5f',
                   delimiter=',')


    # -------------- code start : 结束 -------------
    if verbose > 1:
        logging.debug('-' * 20)
        print '-' * 20
    # -------------- region end : 4. 预测 ---------------




end_time = timeit.default_timer()
print 'end! Running time:%ds!' % (end_time - start_time)
logging.debug('=' * 20)
logging.debug('end! Running time:%ds!' % (end_time - start_time))
