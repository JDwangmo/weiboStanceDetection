#encoding=utf8
"""
    微博立场分析baseline，使用CNN-rand模型，具体见
"""
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-23'
__email__ = '383287471@qq.com'

import numpy as np
import pandas as pd
import logging
from deep_learning.cnn.randEmbedding_cnn.randEmbedding_cnn_model import RandEmbeddingCNN
from deep_learning.cnn.randEmbedding_cnn.feature_encoder import FeatureEncoder
from data_processing.data_util import DataUtil
import os
data_util = DataUtil()

train_data = data_util.load_data('/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/train_data_2090.csv')
test_data = data_util.load_data('/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/test_data_896.csv')

model_file = '/home/jdwang/PycharmProjects/nlp_util/deep_learning/cnn/randEmbedding_cnn/model/' \
             'model_2000epcho_2rd.pkl'


# quit()
print train_data.head()
# train_X = train_data['TEXT'].as_matrix()
# train_X = train_data['WORDS'].as_matrix()
# test_X = test_data['WORDS'].as_matrix()
# test_X = test_data['TEXT'].as_matrix()
train_X = (train_data['TARGET'] +','+ train_data['TEXT']).as_matrix()
test_X = (test_data['TARGET'] +','+ test_data['TEXT']).as_matrix()
label_to_index = {'FAVOR':0,'AGAINST':1,'NONE':2}
index_to_label = ['FAVOR','AGAINST','NONE']

train_y = train_data['STANCE'].map(label_to_index).as_matrix()
test_y = test_data['STANCE'].map(label_to_index).as_matrix()

sentence_padding_length = 150
word_embedding_dim = 300

feature_encoder = FeatureEncoder(train_data=train_X,
                                 sentence_padding_length=sentence_padding_length,
                                 verbose=0,
                                 need_segmented=True
                                 )
# print feature_encoder.train_padding_index[:5]
# print map(feature_encoder.encoding_sentence,test_X)
# quit()
rand_embedding_cnn = RandEmbeddingCNN(
    # rand_seed=1337,
    verbose=1,
    input_dim=feature_encoder.train_data_dict_size+1,
    word_embedding_dim=word_embedding_dim,
    input_length = sentence_padding_length,
    num_labels = len(label_to_index),
    conv_filter_type = [[100,2,word_embedding_dim,'valid'],
                        [100,4,word_embedding_dim,'valid'],
                        # [100,6,word_embedding_dim,'valid'],
                        [100,8,word_embedding_dim,'valid'],
                        # [100,10,word_embedding_dim,'valid'],
                        ],
    k=3,
    embedding_dropout_rate=0.5,
    output_dropout_rate=0.5,
    nb_epoch=2000,
    earlyStoping_patience = 50,
)

if not os.path.exists(model_file):
    # 训练模型
    rand_embedding_cnn.fit((feature_encoder.train_padding_index, train_y),
                           (map(feature_encoder.encoding_sentence,test_X),test_y))
    # 保存模型
    rand_embedding_cnn.save_model(model_file)
else:
    # 从保存的pickle中加载模型
    rand_embedding_cnn.model_from_pickle(model_file)

print index_to_label[rand_embedding_cnn.predict(feature_encoder.encoding_sentence('你好吗'))]

y_pred,is_correct,accu = rand_embedding_cnn.accuracy((map(feature_encoder.encoding_sentence,test_X),test_y))

test_data[u'IS_CORRECT'] = is_correct
test_data[u'PREDICT'] = [index_to_label[item] for item in y_pred]

data_util.save_data(test_data[[u'﻿ID',u'TARGET',u'TEXT',u'STANCE',u'PREDICT',u'IS_CORRECT']],
                    path='/home/jdwang/PycharmProjects/nlp_util/deep_learning/cnn/randEmbedding_cnn/result/rand_cnn.csv')