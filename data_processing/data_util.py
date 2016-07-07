#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-24'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
from sklearn.feature_extraction.text import CountVectorizer


from data_processing_util.jiebanlp.jieba_util import Jieba_Util

class DataUtil(object):
    '''
        微博立场分析数据处理工具类，包含以下函数：
            1. load_data：加载csv格式的数据
            2. save_data：保存csv格式的数据
            3. print_data_detail： 打印数据详情
            4. processing_na_value：处理空值数据
            5. segment_sentence：分词
            6. split_train_test：切分训练集和测试集
            7.
    '''
    def __init__(self):
        # 初始化jieba工具
        self.jieba_util = Jieba_Util()

    def load_data(self,path,header=True):
        '''
            读取数据
        :param path: 数据文件的路径
        :return:
        '''
        if header:
            data = pd.read_csv(path,
                               sep='\t',
                               header=0,
                               encoding='utf8',
                               )
        else:
            data = pd.read_csv(path,
                               sep='\t',
                               header=None,
                               encoding='utf8',
                               )
        return data

    def load_train_test_data(self,
                             config = None
                             ):
        '''
            加载训练数据和测试数据,已经标签索引

        :param config:  一些配置信息
        :param config: dict
        :return:
        '''

        # -------------- region start : 1. 加载训练集和测试集 -------------
        if config['verbose'] > 2:
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('1. 加载训练集和测试集')
            print '1. 加载训练集和测试集'
        # -------------- code start : 开始 -------------
        train_data_file_path = (config['train_data_file_path']) % config['train_data_type']
        test_data_file_path = (config['test_data_file_path']) % config['test_data_type']
        logging.debug(train_data_file_path)
        print train_data_file_path
        logging.debug(test_data_file_path)
        print test_data_file_path

        data_util = DataUtil()
        train_data = data_util.load_data(train_data_file_path)
        test_data = data_util.load_data(test_data_file_path)

        # -------------- code start : 结束 -------------
        if config['verbose'] > 2:
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1. 加载训练集和测试集 ---------------

        # 生成类别索引
        label_to_index = {u'FAVOR': 0, u'AGAINST': 1, u'NONE': 2}
        index_to_label = [u'FAVOR', u'AGAINST', u'NONE']

        return train_data,test_data,label_to_index,index_to_label

    def save_data(self,data,path):
        '''
            保存数据
        :param path: 数据文件的路径
        :return:
        '''
        data.to_csv(path,
                    sep='\t',
                    header=True,
                    index=False,
                    encoding='utf8',
                    )

    def print_data_detail(self, data, has_stance=True):
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
                null_data_path = './result/null_data.csv'
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
                                                 full_mode=True,
                                                 remove_stopword=True,
                                                 replace_number=True,
                                                 lowercase = True,
                                                 zhs2zht=True,
                                                 remove_url=True,
                                                 )
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


    def count_word_freq(self,data):
        '''
            统计每个词 在各个类别中的次数,每个词有四个统计项：
                1. FAVOR：	在favor类别中的出现的次数
                2. AGAINST：在AGAINST类别中的出现的次数
                3. NONE	： 在NONE类别中的出现的次数
                4. FREQ	： 在所有类别中的出现的次数，即FAVOR+AGAINST+NONE
                5. SUPPORT： 最高词频词频项/（FREQ）

        :param data:
        :return:
        '''
        from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder

        feature_encoder = FeatureEncoder(train_data=data['WORDS'].as_matrix(),
                                         verbose=0,
                                         padding_mode='none',
                                         need_segmented=False,
                                         full_mode=True,
                                         remove_stopword=True,
                                         replace_number=True,
                                         lowercase=True,
                                         remove_url=True,
                                         sentence_padding_length=7,
                                         add_unkown_word=False,
                                         mask_zero=False,
                                         zhs2zht=True,
                                         )

        # print feature_encoder.train_padding_index
        train_X_features = feature_encoder.to_onehot_array()

        np.save('result/train_X_feature',train_X_features)

        print train_X_features.shape
        print train_X_features[:5]
        vocabulary = feature_encoder.vocabulary
        print ','.join(vocabulary)
        print feature_encoder.vocabulary_size
        np.save('result/vocabulary',vocabulary)

        freq = np.sum(train_X_features,axis=0)
        favor_freq = np.sum(train_X_features[data['STANCE'].as_matrix()==u'FAVOR'],axis=0)
        against_freq = np.sum(train_X_features[data['STANCE'].as_matrix()==u'AGAINST'],axis=0)
        none_freq = np.sum(train_X_features[data['STANCE'].as_matrix()==u'NONE'],axis=0)



        support = np.nan_to_num([max(favor,against,none)/(1.0*(favor+against+none)) for favor,against,none in zip(favor_freq,against_freq,none_freq)])
        print freq
        print favor_freq
        print against_freq
        print none_freq
        count_data = pd.DataFrame(data={
            u'WORD':vocabulary,
            u'FAVOR':favor_freq,
            u'AGAINST':against_freq,
            u'NONE':none_freq,
            u'SUPPORT':support,
            u'FREQ':freq,
        })
        count_data = count_data.sort_values(by=[u'SUPPORT',u'FREQ','WORD'],ascending=False)
        count_data = count_data[[u'WORD',u'FAVOR',u'AGAINST',u'NONE',u'FREQ',u'SUPPORT']]
        count_data.to_csv('result/word_count.csv',
                          sep='\t',
                          index=False,
                          header=True,
                          encoding='utf8',
                          )
        print count_data.head()


def preprocess_dataAA():
    '''
        数据 evasampledata4-TaskAA.txt 预处理主流程
    :return:
    '''
    # 读取数据
    train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/evasampledata4-TaskAA.txt'

    data_util = DataUtil()

    data = data_util.load_data(train_dataA_file_path)
    data_util.print_data_detail(data, has_stance=True)
    # -------------- region start : 1. 处理空值数据 -------------
    logging.debug('-' * 20)
    print '-' * 20
    logging.debug('1. 处理空值数据')
    print '1. 处理空值数据'

    print '原始数据有%d条句子'%(len(data))
    # 将TEXT字段有空数值的数据去除，并保存
    data = data_util.processing_na_value(data, clear_na=True,columns=[u'TEXT'])
    print '去除TEXT字段有空数值的数据之后，剩下%d条句子'%(len(data))

    logging.debug('-' * 20)
    print '-' * 20
    # -------------- region end : 1. 处理空值数据 ---------------

    # 分词
    data['WORDS'] = data['TEXT'].apply(data_util.segment_sentence)
    # 保存数据
    data_util.save_data(data, 'result/TaskAA_all_data_3000.csv')

    # 将其他字段有空数值的数据去除
    data = data_util.processing_na_value(data, clear_na=True)
    output_file_path = 'result/TaskAA_all_data_%d.csv' % len(data)
    print '去除其他字段有空数值的数据之后，剩下%d条句子,输出到：%s'%(len(data),output_file_path)
    # 保存数据
    data_util.save_data(data, output_file_path)

    # 统计句子的长度,按词(分完词)统计
    data['LENGTH'] = data['WORDS'].apply(lambda x: len(x.split()))
    # 句子长度情况
    print data['LENGTH'].value_counts().sort_index()
    print data.head()
    # print data['WORDS'][:5]

    # 将数据随机切割成训练集和测试集

    train_data, test_data = data_util.split_train_test(data, train_split=0.7)

    print train_data.shape
    print test_data.shape
    data_util.print_data_detail(test_data)
    # print train_data['TARGET'].value_counts()
    # print test_data['TARGET'].value_counts()
    # print data['TARGET'].value_counts()
    # 保存数据
    data_util.save_data(data, 'result/TaskAA_all_data_2986.csv')
    data_util.save_data(train_data, 'result/TaskAA_train_data_2090.csv')
    data_util.save_data(test_data, 'result/TaskAA_test_data_896.csv')


def preprocess_dataAR():
    '''
        数据 evasampledata4-TaskAR.txt 预处理主流程

    :return:
    '''

    train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/evasampledata4-TaskAR.txt'

    data_util = DataUtil()

    data = data_util.load_data(train_dataA_file_path)
    data_util.print_data_detail(data, has_stance=False)
    print data.shape

    data_util.save_data(data, 'result/TaskAR_all_data_2997.csv')
    # 将有空数值的数据去除
    data = data_util.processing_na_value(data, clear_na=True)
    logging.debug('去除空值数据后剩下%d条数据。'%len(data))
    print '去除空值数据后剩下%d条数据。'%len(data)

    # 分词
    data['WORDS'] = data['TEXT'].apply(data_util.segment_sentence)
    # print data.head()
    # 检验分完词之后是否出现空值
    count_null_data = sum(data['WORDS'].isnull())
    logging.debug('WORDS出现空值数：%d'%count_null_data)
    print 'WORDS出现空值数：%d'%count_null_data
    data = data_util.processing_na_value(data, clear_na=True,columns=['WORDS'])
    logging.debug('去除WORDS空值数据后剩下%d条数据。' % len(data))
    print '去除WORDS空值数据后剩下%d条数据。' % len(data)
    data_util.save_data(data, 'result/TaskAR_all_data_2997.csv')

def preprocess_testA():
    '''
        数据 NLPCC2016_Stance_Detection_Task_A_Testdata.txt 预处理主流程
    :return:
    '''

    train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/NLPCC2016_Stance_Detection_Task_A_Testdata.txt'
    # train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskAA_testdata_Mhalf_896.csv'
    # train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskA_all_testdata_15000.csv'
    # output_file_path = 'result/TaskA_all_testdata_15000.csv'
    output_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskAA_all_data_3000_Mhalf.csv'
    # output_file_path = 'result/TaskAA_testdata_Mhalf_896.csv'

    data_util = DataUtil()

    data = data_util.load_data(train_dataA_file_path)

    data['ID'] = data['ID'].astype(dtype=int)

    data_util.print_data_detail(data, has_stance=False)
    print data.shape
    print data.head()
    # quit()
    # 将有空数值的数据去除
    # data = data_util.processing_na_value(data,
    #                                      clear_na=True,
    #                                      columns=[u'TEXT'],
    #                                      )
    logging.debug('去除TEXT字段空值数据后剩下%d条数据。'%len(data))
    print '去除空值数据后剩下%d条数据。'%len(data)
    # print '有%d个句子是已经标注'%sum(data[u'PREDICT'].notnull())

    # -------------- region start : 对句子开始分词 -------------
    logging.debug('-' * 20)
    print '-' * 20
    logging.debug('对句子开始分词')
    print '对句子开始分词'

    # 分词
    data['WORDS'] = data['TEXT'].apply(data_util.segment_sentence)
    print data.head()

    # 检验分完词之后是否出现空值
    count_null_data = sum(data['WORDS'].isnull())
    logging.debug('WORDS出现空值数：%d' % count_null_data)
    print 'WORDS出现空值数：%d' % count_null_data
    # data = data_util.processing_na_value(data, clear_na=True, columns=['WORDS'])
    # logging.debug('去除WORDS空值数据后剩下%d条数据。' % len(data))
    print '去除WORDS空值数据后剩下%d条数据。' % len(data)

    logging.debug('-' * 20)
    print '-' * 20
    # -------------- region end : 对句子开始分词 ---------------


    data_util.save_data(data, output_file_path)


def padding_dataAA():
    '''
        文件 train_data/TaskAA_all_data_3000.csv 补全漏标标签 处理过程

    :return:
    '''
    train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/evasampledata4-TaskAA.txt'
    # train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskAA_testdata_Mhalf_896.csv'
    # train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskA_all_testdata_15000.csv'
    # output_file_path = 'result/TaskA_all_testdata_15000.csv'
    output_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskAA_all_data_3000.csv'
    # output_file_path = 'result/TaskAA_testdata_Mhalf_896.csv'

    data_util = DataUtil()

    data = data_util.load_data(train_dataA_file_path)

    # data['ID'] = data['ID'].astype(dtype=int)

    # data[data['STANCE'].isnull()]['STANCE'] =u'NONE'
    # print data[data['STANCE'].isnull()]['STANCE']
    data_util.print_data_detail(data, has_stance=True)
    # print data.loc[3001]
    data.loc[data['STANCE'].isnull(),'STANCE'] = 'NONE'

    data_util.print_data_detail(data, has_stance=True)
    print data.shape
    # print data.head()
    # quit()
    # print data['STANCE'][data['STANCE'].isnull()]
    # quit()
    data['WORDS'] = data['TEXT'].apply(data_util.segment_sentence)
    # 将有空数值的数据去除
    # data = data_util.processing_na_value(data,
    #                                      clear_na=True,
    #                                      columns=[u'TEXT'],
    #                                      )
    # logging.debug('去除TEXT字段空值数据后剩下%d条数据。' % len(data))
    # print '去除空值数据后剩下%d条数据。' % len(data)
    # print '有%d个句子是已经标注'%sum(data[u'PREDICT'].notnull())

    data_util.save_data(data, output_file_path)




if __name__ == '__main__':
    # 数据 evasampledata4-TaskAA.txt 预处理主流程
    # preprocess_dataAA()

    # 数据 evasampledata4-TaskAR.txt 预处理主流程
    # preprocess_dataAR()

    # NLPCC2016_Stance_Detection_Task_A_Testdata.txt预处理主流程
    # preprocess_testA()

    # 文件train_data / TaskAA_all_data_3000.csv 补全漏标标签处理过程
    padding_dataAA()
