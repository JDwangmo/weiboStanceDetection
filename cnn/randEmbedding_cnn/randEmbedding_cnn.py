#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-23'
__email__ = '383287471@qq.com'

import numpy as np
import pandas as pd
import logging
from deep_learning.cnn.randEmbedding_cnn.randEmbedding_cnn_model import RandEmbeddingCNN
from deep_learning.cnn.randEmbedding_cnn.feature_encoder import FeatureEncoder
from data_processing.data_util import DataUtil

data_util = DataUtil()

train_data = data_util.load_data('/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/train_data_2090.csv')
test_data = data_util.load_data('/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/test_data_896.csv')
print train_data.head()
train_X = train_data['WORDS'].as_matrix()
test_X = test_data['WORDS'].as_matrix()
label_to_index = {'FAVOR':0,'AGAINST':1,'NONE':2}
train_y = train_data['STANCE'].map(label_to_index).as_matrix()
test_y = test_data['STANCE'].map(label_to_index).as_matrix()

sentence_padding_length = 150
word_embedding_dim = 50

feature_encoder = FeatureEncoder(train_data=train_X,
                                 sentence_padding_length=sentence_padding_length,
                                 verbose=0,
                                 need_segmented=False)
# print feature_encoder.train_padding_index[:5]
# print map(feature_encoder.encoding_sentence,test_X)
# quit()
rand_embedding_cnn = RandEmbeddingCNN(
    rand_seed=1337,
    verbose=1,
    input_dim=feature_encoder.train_data_dict_size+1,
    word_embedding_dim=word_embedding_dim,
    input_length = sentence_padding_length,
    num_labels = len(label_to_index),
    conv_filter_type = [[100,2,word_embedding_dim,'valid'],
                        [100,4,word_embedding_dim,'valid'],
                        [100,6,word_embedding_dim,'valid'],
                        [100,8,word_embedding_dim,'valid'],
                        [100,10,word_embedding_dim,'valid'],
                        ],
    dropout_rate = 0.5,
    nb_epoch=100,
    earlyStoping_patience = 50,
)
# 训练模型
rand_embedding_cnn.fit((feature_encoder.train_padding_index, train_y),
                       (map(feature_encoder.encoding_sentence,test_X),test_y))
# 保存模型
rand_embedding_cnn.save_model('model/model.pkl')

# 从保存的pickle中加载模型
# rand_embedding_cnn.model_from_pickle('model/model.pkl')
print rand_embedding_cnn.predict(feature_encoder.encoding_sentence('你好吗'))