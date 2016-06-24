#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-23'
__email__ = '383287471@qq.com'

import numpy as np
import pandas as pd
import logging

from jiebanlp.toolSet import seg
from gensim.corpora.dictionary import Dictionary

class RandEmbeddingCNN(object):
    '''
        一层CNN模型,随机初始化词向量,CNN-rand模型.
    '''
    def __init__(self, rand_seed=1337, verbose=0):
        '''
            初始化参数
        :param rand_seed: 随机种子
        :type rand_seed: int
        :param verbose: 数值越大,输出更详细的信息
        :param input_dim: onehot输入的维度,即 字典大小+1,+1是为了留出0给填充用
        :param input_dim: int
        '''
        # todo 初始化参数,filter size等
        self.verbose = verbose
        self.rand_seed = rand_seed
        self.verbose = verbose

        self.input_dim = None

    def build_model(self):
        '''
            构建CNN模型
        :return:
        '''
        if self.rand_seed is not None:
            np.random.seed(self.rand_seed)
        from keras.layers import Embedding, Convolution2D, Input, Activation, MaxPooling2D, Reshape, Dropout, Dense, \
            Flatten, Merge
        from keras.callbacks import EarlyStopping
        from keras.models import Sequential, Model, model_from_json
        from keras.utils import np_utils
        from keras import backend as K

        input_dim = len(feature_encoder.__gensim_dict__.keys()) + 1
        word_embedding_length = config['word_embedding_length']
        input_length = config['padding_length']
        # print input_dim

        cnn_model = Sequential()
        # 构建第一层卷积层
        conv_layers = []
        nn_conv_filters = len(config['conv_filter_type'])
        for items in config['conv_filter_type']:

            nb_filter, nb_row, nb_col, border_mode = items
            m = Sequential()
            m.add(Convolution2D(nb_filter,
                                nb_row,
                                nb_col,
                                border_mode=border_mode,
                                input_shape=(1,
                                             input_length,
                                             word_embedding_length)
                                ))
            m.add(Activation('relu'))
            # 1-max
            if border_mode == 'valid':
                pool_size = (input_length - nb_row + 1, 1)
            elif border_mode == 'same':
                pool_size = (input_length, 1)
            m.add(MaxPooling2D(pool_size=pool_size, name='1-max'))
            conv_layers.append(m)

        cnn_model.add(Merge(conv_layers, mode='concat', concat_axis=1))
        cnn_model.add(Flatten())

        # print cnn_model.summary()
        # quit()
        model_input = Input((input_length,), dtype='int64')
        embedding = Embedding(input_dim=input_dim,
                              output_dim=word_embedding_length,
                              input_length=input_length,
                              # mask_zero = True,
                              init='uniform')(model_input)

        embedding_4_dim = Reshape((1, input_length, word_embedding_length))(embedding)

        conv1_output = cnn_model([embedding_4_dim] * nn_conv_filters)

        full_connected_layers = Dense(output_dim=len(label_to_index), init="glorot_uniform", activation='relu')(
            conv1_output)

        dropout_layers = Dropout(p=config['dropout_rate'])(full_connected_layers)
        # dropout_layers = full_connected_layers
        softmax_output = Activation("softmax")(dropout_layers)

        model = Model(input=[model_input], output=[softmax_output])
        model_output = K.function([model_input, K.learning_phase()],
                                  [softmax_output])
        print model.summary()

        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        logging.debug('开始训练...')
        print '开始训练...'

        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        pass

    def train(self):
        # todo 模型
        pass

    def predict(self):
        # todo 预测
        pass



class FEATURE_ENCODER(object):
    '''
    输入的特征编码器,onehot编码
    '''
    def __init__(self,train_data=None,
                 full_mode=True,
                 remove_stopword=True,
                 sentence_padding = 7,
                 verbose=0):
        '''

        :param train_data: 训练句子
        :type train_data: np.array([])
        '''
        self.__full_mode__ = full_mode
        self.__remove_stopword__ = remove_stopword
        self.__verbose__ = verbose
        self.__sentence_padding__ = sentence_padding



        if train_data is not None:
            self.__train_data__ = train_data
            self.build_encoder(train_data)

    def __seg__(self, sentence):
        '''
        对句子进行分词,使用jieba分词
        :param sentence: 句子
        :type sentence: str
        :return:
        '''
        sentence_to_seg = lambda x: seg(x,
                                        sep=' ',
                                        full_mode=self.__full_mode__,
                                        remove_stopword=self.__remove_stopword__,
                                        verbose=self.__verbose__
                                        )
        return sentence_to_seg(sentence)

    def build_dictionary(self):
        logging.debug('=' * 20)
        logging.debug('首先,构建训练库字典,然后将每个词映射到一个索引,再将所有句子映射成索引的列表')

        # 构建训练库字典
        # 将训练库所有句子切分成列表,构成 2D的训练文档,每个单元是一个token,
        # 比如: [['今年','你','多少岁'],['你', '二十四','小时','在线','吗'],...]

        train_document = map(lambda x: x.split(),self.__seg_sentence__)

        gensim_dict = Dictionary.from_documents(train_document)

        logging.debug('训练库字典为:%d' % (len(gensim_dict.keys())))
        print '训练库字典为:%d' % (len(gensim_dict.keys()))

        # 更新字典,再字典中添加特殊符号,其中
        # U表示未知字符,即OOV词汇
        gensim_dict.add_documents([[u'UNKOWN']])
        logging.debug('更新字典,再字典中添加特殊符号(UNKOWN),之后字典大小为:%d' % (len(gensim_dict.keys())))
        print '更新字典,再字典中添加特殊符号,之后字典大小为:%d' % (len(gensim_dict.keys()))

        logging.debug(u'字典有:%s' % (','.join(gensim_dict.token2id.keys())))
        print u'字典有:%s' % (','.join(gensim_dict.token2id.keys()))
        self.__gensim_dict__ = gensim_dict

    def sentence_to_index(self,sentence):
        """
        将 sentence 转换为 index,如果 token为OOV词,则分配为 UNKOWN
        :type sentence: str
        :param sentence: 以空格分割
        :return:
        """
        unknow_token_index = self.__gensim_dict__.token2id[u'UNKOWN']
        # 将训练库中所有句子的每个词映射到索引上,变成索引列表
        # 注意这里把所有索引都加1,目的是为了保留 索引0(用于补充句子),在神经网络上通过mask_zero忽略,实现变长输入
        index = [self.__gensim_dict__.token2id.get(item, unknow_token_index) + 1 for item in sentence.split()]
        return index

    def sentence_padding(self,sentence):
        '''
        将不等长的句子都对齐,超出padding_length长度的句子截断,小于的则补0
        :type sentence: list
        :param sentence: 以索引列表表示的句子
        :type padding_length: int
        :param padding_length: 补齐长度
        :return:
        '''

        padding_length = self.__sentence_padding__
        # print sentence
        sentence_length = len(sentence)
        if sentence_length > padding_length:
            # logging.debug(u'对句子进行截断:%s' % (sentence))

            sentence = sentence[:padding_length]

            # logging.debug(u'对句子进行截断后:%s' % (' '.join(seg[:padding_length])))
            # print(u'对句子进行截断后:%s' % (' '.join(seg[:padding_length])))
        elif sentence_length < padding_length:
            should_padding_length = padding_length - sentence_length
            left_padding = np.asarray([0] * (should_padding_length / 2))
            right_padding = np.asarray([0] * (should_padding_length - len(left_padding)))
            sentence = np.concatenate((left_padding, sentence, right_padding), axis=0)

        return sentence

    def build_encoder(self,train_data=None):
        logging.debug('=' * 20)
        if train_data is not None:
            self.__train_data__ = train_data
        if train_data is None and self.__train_data__ is None:
            logging.debug('没有输入训练数据!')
            print '没有输入训练数据!'
            quit()

        logging.debug('对数据进行分词...')
        logging.debug('-' * 20)

        self.__seg_sentence__ = map(self.__seg__,self.__train_data__)

        print ','.join(self.__seg_sentence__)

        # 构建训练库字典
        self.build_dictionary()

        # 将训练库中所有句子的每个词映射到索引上,变成索引列表
        train_index = map(self.sentence_to_index, self.__seg_sentence__)
        # print train_index[:5]

        # 将不等长的句子都对齐,超出padding_length长度的句子截断,小于的则补0
        train_padding_index = np.asarray(map(self.sentence_padding, train_index))
        self.__train_padding_index__ = train_padding_index


    def encoding_sentence(self,sentence):
        # 跟训练数据一样的操作
        # 分词
        seg_sentence = self.__seg__(sentence)
        sentence_index = self.sentence_to_index(seg_sentence)
        sentence_padding_index = self.sentence_padding(sentence_index)
        return sentence_padding_index
