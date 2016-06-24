#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-24'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit

from data_processing_util.jiebanlp.jieba_util import Jieba_Util

class DataUtil(object):
    def __init__(self):
        # 初始化jieba工具
        self.jieba_util = Jieba_Util()
        pass

    def load_data(self,path):
        '''
            读取数据
        :param path: 数据文件的路径
        :return:
        '''
        data = pd.read_csv(path,
                           sep='\t',
                           header=0,
                           encoding='utf8',
                           )
        return data

    def save_data(self,data,path):
        '''
            保存数据
        :param path: 数据文件的路径
        :return:
        '''
        data = data.to_csv(path,
                           sep='\t',
                           header=True,
                           index=False,
                           encoding='utf8',
                           )

    def data_detail(self,data, has_stance=True):
        '''
            展示数据的详细信息
        :param data: Dateframe对象
        :param has_stance: 是否有STANCE字段
        :return: 无
        '''

        logging.debug('data的个数为：%d' % (len(data)))
        logging.debug('data的sample数据：')
        logging.debug(data.head())

        logging.debug('data的target和个数分别为：')
        logging.debug(data['TARGET'].value_counts())
        if has_stance:
            logging.debug('统计每个Target下各个类型立场的数量...')
            group = data.groupby(by=['TARGET', 'STANCE'])
            logging.debug(group.count())
        else:
            logging.debug('没有STANCE字段')

        logging.debug('数据各个字段情况...')
        # print data.info()
        for column in data.columns:
            # 统计每个字段是否有数据是空串
            # 先将所有空字符串用nan替换
            data[column] = data[column].replace(r'^\s*$', np.nan, regex=True)
            count_null = sum(data[column].isnull())
            if count_null != 0:
                logging.warn(u'%s字段有空值，个数：%d,建议使用processing_na_value()方法进一步处理！' % (column, count_null))
                null_data_path = './null_data.csv'
                logging.warn(u'将缺失值数据输出到文件：%s' % (null_data_path))
                data[data[column].isnull()].to_csv(null_data_path,
                                                   index=None,
                                                   encoding='utf8',
                                                   sep='\t')

    def processing_na_value(self,data,clear_na=True,fill_na = False,fill_char = 'NULL',columns=None):
        '''
            处理数据的空值

        :param data:  Dateframe对象
        :param clear_na: bool,是否去掉空值数据
        :param fill_na: bool，是否填充空值
        :param fill_char: str，填充空置的字符
        :param column: list，需要处理的字段，默认为None时，对所有字段处理
        :return: Dateframe对象
        '''
        logging.debug('[def processing_na_value()] 对缺失值进行处理....')
        for column in data.columns:
            if columns == None or column in columns:
                data[column] = data[column].replace(r'^\s*$', np.nan, regex=True)
                count_null = sum(data[column].isnull())
                if count_null != 0:
                    logging.warn(u'%s字段有空值，个数：%d' % (column, count_null))
                    if clear_na:
                        logging.warn(u'对数据的%s字段空值进行摘除'%(column))
                        data = data[data[column].notnull()].copy()
                    else:
                        if fill_na:
                            logging.warn(u'对数据的%s字段空值进行填充，填充字符为：%s'%(column,fill_char))
                            data[column] = data[column].fillna(value=fill_char)

        return data

    def segment_sentence(self,sentence):
        segmented_sentence = self.jieba_util.seg(sentence=sentence,
                                                 sep=' ',
                                                 full_mode=False,
                                                 remove_stopword=True,
                                                 replace_number=False)
        return segmented_sentence

    def split_train_test(self,data, train_split=0.7):
        '''
            将数据切分成训练集和验证集

        :param data:
        :param train_split: float，取值范围[0,1],设置训练集的比例
        :return: dev_data,test_data
        '''
        logging.debug('对数据随机切分成train和test数据集，比例为：%f' % (train_split))
        num_train = len(data)
        num_dev = int(num_train * train_split)
        num_test = num_train - num_dev
        logging.debug('全部数据、训练数据和测试数据的个数分别为：%d,%d,%d' % (num_train, num_dev, num_test))
        rand_list = np.random.RandomState(0).permutation(num_train)
        # print rand_list
        # print rand_list[:num_dev]
        # print rand_list[num_dev:]
        dev_data = data.iloc[rand_list[:num_dev]].sort_index()
        test_data = data.iloc[rand_list[num_dev:]].sort_index()
        # print dev_data
        # print test_data
        return dev_data, test_data



if __name__ == '__main__':
    train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                            'evasampledata4-TaskAA.txt'
    data_util = DataUtil()
    data = data_util.load_data(train_dataA_file_path)
    data_util.data_detail(data, has_stance=True)

    print data.shape
    # 将有空数值的数据去除
    data = data_util.processing_na_value(data, clear_na=True)

    print data.shape
    # print train_data.head()
    # 分词
    data['WORDS'] = data['TEXT'].apply(data_util.segment_sentence)
    # 统计句子的长度,按词(分完词)统计
    data['LENGTH'] = data['WORDS'].apply(lambda x:len(x.split()))
    # 句子长度情况
    print data['LENGTH'].value_counts().sort_index()
    print data.head()
    # print data['WORDS'][:5]

    # 将数据随机切割成训练集和测试集
    train_data,test_data = data_util.split_train_test(data,train_split=0.7)

    print train_data.shape
    print test_data.shape
    data_util.data_detail(test_data)
    # print train_data['TARGET'].value_counts()
    # print test_data['TARGET'].value_counts()
    # print data['TARGET'].value_counts()
    # 保存数据
    data_util.save_data(data, 'result/all_data_2986.csv')
    data_util.save_data(train_data, 'result/train_data_2090.csv')
    data_util.save_data(test_data, 'result/test_data_896.csv')







