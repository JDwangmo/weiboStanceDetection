#encoding=utf8
import io

__author__ = 'jdwang'
__date__ = 'create date: 2016-06-30'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
from data_processing.data_util import DataUtil
verbose = 1
word_count_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/word_count_17211.csv'
train_X_feature_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/data_processing/result/train_X_feature.npy'
vocabulary_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/data_processing/result/vocabulary.npy'
train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskAA_all_data_2986.csv'

selected_keywords_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/cue_pharse/result/selected_keywords.csv'

vocabulary = np.load(vocabulary_file_path)
token_to_id = {item:idx for idx,item in enumerate(vocabulary)}

train_X_feature = np.load(train_X_feature_file_path)
data_util = DataUtil()

train_dataA = data_util.load_data(train_dataA_file_path,header=True)


def count_word_freq():
    '''
        对文件（train_data/TaskAA_all_data_2986.csv）统计词频。
        统计每个词 在各个类别中的次数,每个词有五个统计项：
            1. FAVOR：	在favor类别中的出现的次数
            2. AGAINST：在AGAINST类别中的出现的次数
            3. NONE	： 在NONE类别中的出现的次数
            4. FREQ	： 在所有类别中的出现的次数，即FAVOR+AGAINST+NONE
            5. SUPPORT： 最高词频词频项/（FREQ）
        步骤如下：
            1. 将所有句子转成onehot编码
            2. 统计每个词的5种统计值

    :return:
    '''
    # -------------- region start : 1. 将所有句子转成onehot编码,并保存数据 -------------
    if verbose > 1 :
        logging.debug('-' * 20)
        print '-' * 20
        logging.debug('1. 将所有句子转成onehot编码,并保存数据')
        print '1. 将所有句子转成onehot编码,并保存数据'
    # -------------- code start : 开始 -------------

    from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
    # print train_dataA.head()
    print train_dataA.shape
    feature_encoder = FeatureEncoder(train_data=train_dataA['WORDS'].as_matrix(),
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

    np.save('result/train_X_feature', train_X_features)

    print train_X_features.shape
    print train_X_features[:5]
    vocabulary = feature_encoder.vocabulary
    print ','.join(vocabulary)
    print '字典个数有：%d'%feature_encoder.train_data_dict_size
    np.save('result/vocabulary', vocabulary)

    # -------------- code start : 结束 -------------
    if verbose > 1 :
        logging.debug('-' * 20)
        print '-' * 20
    # -------------- region end : 1. 将所有句子转成onehot编码,并保存数据 ---------------

    # -------------- region start : 2. 统计每个词的5种统计值 -------------
    if verbose > 1 :
        logging.debug('-' * 20)
        print '-' * 20
        logging.debug('2. 统计每个词的5种统计值')
        print '2. 统计每个词的5种统计值'
    # -------------- code start : 开始 -------------

    # 总词频
    freq = np.sum(train_X_features, axis=0)
    # favor类中词频
    favor_freq = np.sum(train_X_features[train_dataA['STANCE'].as_matrix() == u'FAVOR'], axis=0)
    # against类中词频
    against_freq = np.sum(train_X_features[train_dataA['STANCE'].as_matrix() == u'AGAINST'], axis=0)
    # none类中词频
    none_freq = np.sum(train_X_features[train_dataA['STANCE'].as_matrix() == u'NONE'], axis=0)
    # 支持度 ：最高词频/总词频
    support = np.nan_to_num([max(favor, against, none) / (1.0 * (favor + against + none)) for favor, against, none in
                             zip(favor_freq, against_freq, none_freq)])

    print freq
    print favor_freq
    print against_freq
    print none_freq

    count_data = pd.DataFrame(data={
        u'WORD': vocabulary,
        u'FAVOR': favor_freq,
        u'AGAINST': against_freq,
        u'NONE': none_freq,
        u'SUPPORT': support,
        u'FREQ': freq,
    })
    # -------------- code start : 结束 -------------
    if verbose > 1 :
        logging.debug('-' * 20)
        print '-' * 20
    # -------------- region end : 2. 统计每个词的5种统计值 ---------------

    quit()
    count_data = count_data.sort_values(by=[u'SUPPORT', u'FREQ', 'WORD'], ascending=False)
    count_data = count_data[[u'WORD', u'FAVOR', u'AGAINST', u'NONE', u'FREQ', u'SUPPORT']]
    count_data.to_csv('result/word_count.csv',
                      sep='\t',
                      index=False,
                      header=True,
                      encoding='utf8',
                      )
    print count_data.head()


def process_1gram():
    # -------------- region start : 1.获取1-gram 的词频列表，并选出特征词和候选特征词 -------------
    logging.debug('-' * 20)
    print '-' * 20
    logging.debug('1.获取1-gram 的词频列表，并选出特征词和候选特征词')
    print '1.获取1-gram 的词频列表，并选出特征词和候选特征词'

    data_word_count = data_util.load_data(word_count_file_path)

    # support>=0.8 & frequency>=5
    mask1 = (data_word_count['SUPPORT'].as_matrix()>=0.85) * (data_word_count['FREQ'].as_matrix()>=5)

    # support>=0.75 & frequency>=10
    mask2 = (data_word_count['SUPPORT'].as_matrix()>=0.8) * (data_word_count['FREQ'].as_matrix()>=10)

    # support>=0.6 & frequency>=5
    mask3 = (data_word_count['SUPPORT'].as_matrix()>=0.6) * (data_word_count['FREQ'].as_matrix()>=5)

    # 1281个候选词
    candiate_words = data_word_count[mask3]
    # 命中的关键词
    keywords = data_word_count[mask1+mask2]
    # 剩余需要进一步处理的词
    rest_words = data_word_count[-(mask2+mask1)*mask3]

    print candiate_words.shape
    print keywords.shape
    print rest_words.shape
    print rest_words.head()
    keywords = keywords.sort_values(by=[u'SUPPORT', u'FREQ', u'WORD'], ascending=False)
    data_util.save_data(keywords,'result/selected_keywords.csv')

    logging.debug('-' * 20)
    print '-' * 20
    # -------------- region end : 1.获取1-gram 的词频列表，并选出特征词和候选特征词 ---------------

    return candiate_words,keywords,rest_words

def process_2gram(candiate_words,rest_words):
    '''
        由候选1gram组合出2gram，再对2gram统计，符合条件的选出作为特征词
    :return:
    '''

    data = data_util.load_data(train_dataA_file_path)
    # 组合出候选2gram
    candiate_2gram = []
    for i in rest_words['WORD'].tolist():
        for j in candiate_words['WORD'].tolist():
            # print ','.join(sorted([i, j]))
            if i!=j:
                candiate_2gram.append([i, j])

    print len(candiate_2gram)

    count_2gram_count = lambda vector,x,y: int(vector[token_to_id[x]]>0 and vector[token_to_id[y]]>0)

    with io.open('result/selected_2gram_count.csv','w',encoding='utf8') as fout:
        fout.write(u'WORD\tFREQ\tFAVOR\tAGAINST\tNONE\tSUPPORT\n')
        counter = 0
        filter1 = np.asarray([data['STANCE'].as_matrix() == u'FAVOR']).flatten()
        filter2 = np.asarray([data['STANCE'].as_matrix() == u'NONE']).flatten()
        filter3 = np.asarray([data['STANCE'].as_matrix() == u'AGAINST']).flatten()
    #     # print filter1
        for x,y in candiate_2gram:
    #         # print x,y
    #         # print count_2gram_count(train_X_feature[0],x,y)
            count = np.asarray(map(lambda vector:count_2gram_count(vector,x,y),train_X_feature))
            freq = np.sum(count)
            # print count
            favor_freq = np.sum(count[filter1])
            none_freq = np.sum(count[filter2])
            against_freq = np.sum(count[filter3])
            support = np.nan_to_num(max(favor_freq, against_freq, none_freq) / (1.0 * (favor_freq+ against_freq+ none_freq)))
            mask2 = (support >= 0.8) * (freq >= 10)
            # support>=0.6 & frequency>=5
            mask3 = (support >= 0.85) * (freq >= 5)
            # print mask2+mask3
            if mask2+mask3:
                print x,y
                print(freq, favor_freq, against_freq, none_freq, support)
                fout.write(u'%s,%s\t%d\t%d\t%d\t%d\t%f\n'%(x,y,freq, favor_freq, against_freq, none_freq, support))

            counter +=1
            if counter%1000==0:
                print '第%d個數據...'%counter


    data_2gram_count = data_util.load_data('result/selected_2gram_count.csv')
    # print data_2gram_count[u'SUPPORT']
    # print data_2gram_count.shape
    print data_2gram_count.columns
    data_2gram_count = data_2gram_count.sort_values(by=[u'SUPPORT', u'FREQ', u'WORD'], ascending=False)
    data_2gram_count = data_2gram_count[[u'WORD', u'FAVOR', u'AGAINST', u'NONE', u'FREQ', u'SUPPORT']]
    print data_2gram_count.head()
    data_2gram_count.to_csv('result/selected_2gram_count.csv',
                      sep='\t',
                      index=False,
                      header=True,
                      encoding='utf8',
                      )

def test():
    test_data = data_util.load_data('/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/dataA_150len.csv')
    test_data['WORDS'] = test_data['TEXT'].apply(data_util.segment_sentence)
    data_util.save_data(test_data,'/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskAR_full_3000.csv')
    quit()
    for item in test_data['TEXT'].apply(data_util.segment_sentence).as_matrix():
        seg = item.split()
        print item
        # todo
        # [i for item in seg]
        quit()
    print test_data

if __name__ == '__main__':

    # test()
    count_word_freq()
    # candiate_words,keywords,rest_words = process_1gram()
    # process_2gram(candiate_words,rest_words)
    # data_util.save_data(keywords,selected_keywords_file_path)


