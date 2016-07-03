#encoding=utf8
import pandas as pd
import logging
import jieba
import re
import itertools
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )
# 最大句子填充完长度,如果超过就截断，如果少就填充
MAX_SENTENCE_LENGTH = 150
START_CHAR = u'[start]'
END_CHAR = u'[end]'
PAD_CHAR = u'[none]'

def data_detail(data,has_stance = True):
    '''
    展示数据的详细信息
    :param data: Dateframe对象
    :param has_stance: 是否有STANCE字段
    :return: 无
    '''
    logging.debug('data的个数为：%d'%(len(data)))
    logging.debug('data的sample数据：')
    logging.debug( data.head())

    logging.debug('data的target和个数分别为：')
    logging.debug(data['TARGET'].value_counts())
    if has_stance:
        logging.debug('统计每个Target下各个类型立场的数量...')
        group = data.groupby(by=['TARGET','STANCE'])
        logging.debug( group.count())
    else:
        logging.debug('没有STANCE字段')

    logging.debug('数据各个字段情况...')
    # print data.info()
    for column in data.columns:
        # 统计每个字段是否有数据是空串
        # 先将所有空字符串用nan替换
        data[column] = data[column].replace(r'^\s*$',np.nan,regex=True)
        count_null = sum(data[column].isnull())
        if count_null!=0:
            logging.warn('%s字段有空值，个数：%d,建议使用processing_na_value()方法进一步处理！'%(column,count_null))
            null_data_path = './null_data.csv'
            logging.warn('将缺失值数据输出到文件：%s'%(null_data_path))
            data[data[column].isnull()].to_csv(null_data_path,
                                               index=None,
                                               sep='\t')


def processing_na_value(data,clear_na=True,fill_na = False,fill_char = 'NULL',columns=None):
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
                logging.warn('%s字段有空值，个数：%d' % (column, count_null))
                if clear_na:
                    logging.warn('对数据的%s字段空值进行摘除'%(column))
                    data = data[data[column].notnull()].copy()
                else:
                    if fill_na:
                        logging.warn('对数据的%s字段空值进行填充，填充字符为：%s'%(column,fill_char))
                        data[column] = data[column].fillna(value=fill_char)

    return data

def split_train_test(data,train_split = 0.7):
    '''
    将数据切分成训练集和验证集
    :param data:
    :param train_split: float，取值范围[0,1],设置训练集的比例
    :return: dev_data,test_data
    '''
    logging.debug('对数据随机切分成train和test数据集，比例为：%f'%(train_split))
    num_train = len(data)
    num_dev = int(num_train * train_split)
    num_test = num_train - num_dev
    logging.debug('全部数据、训练数据和测试数据的个数分别为：%d,%d,%d'%(num_train,num_dev,num_test))
    rand_list = np.random.RandomState(0).permutation(num_train)
    # print rand_list
    # print rand_list[:num_dev]
    # print rand_list[num_dev:]
    dev_data = data.iloc[rand_list[:num_dev]].sort_index()
    test_data = data.iloc[rand_list[num_dev:]].sort_index()
    # print dev_data
    # print test_data
    return dev_data,test_data

def clean_data(data,columns = None,filter_char = False):
    '''
    清理数据：过滤部分字符，分词切句
    :param data: Dateframe对象
    :param columns:list，需要处理的字段，默认为None时，对所有字段处理
    :param filter_char:bool，是否过滤字符
    :return:
    '''
    logging.debug('[def clean_data()]开始清理数据...')
    if columns == None:
        logging.warn('注意：现在对所有字段进行clean和segment,please spec the columns！')

    logging.warn('是否开启过滤字符功能：%s'%(filter_char))

    for column in data.columns:
        if columns == None or column in columns:
            logging.warn('现在处理字段：%s...'%(column))
            if data[column].dtype != object:
                logging.warn('字段(%s)不是字符串类型，直接跳过不处理！'%(column))
                continue
            # 原始句子列表
            # 把空值使用nan替换，不然nan数据去进行下一步关于字符串的操作时会报错
            texts = data[column].replace(np.nan,'')
            # print origin
            if filter_char:
                # 使用正则过滤掉非中文的字符
                pattern = re.compile(u'[^\u4e00-\u9fa5]+')
                filter_not_chinese_text = lambda x : pattern.sub(' ',x.decode('utf8'))
                texts = texts.apply(filter_not_chinese_text)
                data['CLEAN_%s'%(column.upper())] = texts

            # 检查，是否出现空句子
            for i, items in enumerate(texts):
                # print items
                if len(items.strip())==0:
                    logging.warn('第%d句出现空句子'%(i+1))
            # 分词处理
            segment_text = lambda x:\
                ','.join([item for item in jieba.cut(x,cut_all=True) if len(item.strip())!=0])
            segment_sentences = texts.apply(segment_text)
            data['SEGMENT_%s'%(column.upper())] = segment_sentences

            max_length = max([len(item.split(',')) for item in segment_sentences])
            logging.debug(u'最长: ' +','.join([item for item in segment_sentences
                                             if len(item.split(',')) ==max_length]))
            logging.debug('最大句子长度为%d'%(max_length))

    return data


def to_vector(data):
    '''
    为data创建一个字典
    将为data中句子生成句子向量，并返回字典频率、target正、负、中的概率等
    :param data:
    :return: data,(freq_pos,freq_neg,freq_non),target_dict
    '''
    logging.debug('开始转换数据成向量的形式...')
    sentences = [item.split(',') for item in data['SEGMENT_TEXT']]
    # 由于部分句子过滤后出现空串，所以字典会出现空串，这里将空串这个移除
    logging.debug('移除空串')
    [item.remove('') for item in sentences if item.__contains__('')]
    # 创建字典
    dictionary = set(itertools.chain(*sentences))
    # 增加两个特殊字符，开始标记和结束标记
    start_char = START_CHAR
    end_char = END_CHAR
    pad_char = PAD_CHAR
    logging.info(u'增加两个特殊字符，开始标记(%s)和结束标记(%s)'%(start_char,end_char))
    logging.info(u'增加一个特殊字符，填充标记(%s)'%(pad_char))
    dictionary.add(start_char)
    dictionary.add(end_char)
    dictionary.add(pad_char)

    dict_size = len(dictionary)
    logging.debug('字典大小为：%d'%(dict_size))
    idx2word = list(dictionary)
    word2idx = { item:idx for idx,item in enumerate(dictionary)}
    logging.debug(u'开始标记(%s)、结束标记(%s)和填充标记(%s)的字典索引分别为：%d，%d，%d'%(
        start_char,
        end_char,
        pad_char,
        word2idx[start_char],
        word2idx[end_char],
        word2idx[pad_char]
    ))
    # print word2idx
    # 将句子转成整数的字典索引
    # 如果出现未知字符，则使用none字符填充
    sentence_to_index = lambda x : [word2idx.get(item,word2idx[pad_char]) for item in [start_char]+x+[end_char]]
    indexs = [ sentence_to_index(items) for items in sentences]
    # print indexs[0]
    vectors = []
    for items in indexs:
        temp = np.zeros(dict_size,dtype=int)
        temp[items]=1
        vectors.append(temp)
    vectors = np.array(vectors)

    logging.debug('向量shape：%d,%d'%( vectors.shape))
    # print data.head()
    count_pos = sum(vectors[(data['STANCE']=='FAVOR').as_matrix()])
    count_neg = sum(vectors[(data['STANCE']=='AGAINST').as_matrix()])
    count_non = sum(vectors[(data['STANCE']=='NONE').as_matrix()])
    count_all = sum(vectors)


    pos_word =  [i for i,item in enumerate(count_pos) if item!=0]
    neg_word =  [i for i,item in enumerate(count_neg) if item!=0]
    non_word =  [i for i,item in enumerate(count_non) if item!=0]
    logging.debug(u'正例词(%d个):'%(len(pos_word))+ ','.join([idx2word[i] for i in pos_word]))
    logging.debug(u'负例词(%d个):'%(len(neg_word)) + ','.join([idx2word[i] for i in neg_word]))
    logging.debug(u'中立词(%d个):'%(len(non_word)) +','.join([idx2word[i] for i in non_word]))

    freq_pos = np.nan_to_num(count_pos/(count_all*1.0))
    freq_neg = np.nan_to_num(count_neg/(count_all*1.0))
    freq_non = np.nan_to_num(count_non/(count_all*1.0))

    index_to_freq = lambda x:np.asarray([[freq_pos[item],
                                          freq_neg[item],
                                          freq_non[item]] for item in x])
    sentences_freqs = [index_to_freq(item) for item in indexs]

    # print sentences_freqs[0].shape
    # 统计每个Target下各个类型立场的数量
    group = data.groupby(by=['TARGET'])
    target_dict = {}
    for target,g in group:
        # print target
        g_count_pos =  sum(g['STANCE'] == 'FAVOR')
        g_count_neg =  sum(g['STANCE'] == 'AGAINST')
        g_count_non =  sum(g['STANCE'] == 'NONE')
        g_count_all = len(g)
        target_dict[target] = np.array([g_count_pos,g_count_neg,g_count_non])/(g_count_all*1.0)
        # print target_dict[target]
    # 原始句子最大长度172
    max_sentence_length = MAX_SENTENCE_LENGTH
    logging.info('设置句子长度为：%d'%(max_sentence_length))
    vector_prob = []
    indexs_padding = []
    for target,sent_freq,index in zip(data['TARGET'],sentences_freqs,indexs):
        # print sent_freq
        # print len(sent_freq)
        # print len(index)

        if len(sent_freq)<max_sentence_length:
            # 句子长度小于最长句子长度，将其padding
            # print len(sent)
            padding_length = max_sentence_length - len(sent_freq)

            # print '需要填充%d'%(padding_length)
            sentence_after_padding = np.concatenate([sent_freq,padding_length*[target_dict[target]]])
            vector_prob.append(sentence_after_padding)
            # print sentence_after_padding
            # print len(sentence_after_padding)
            # 使用none字符填充
            index = np.pad(index,(0,padding_length),mode='constant',constant_values=word2idx[pad_char])
            indexs_padding.append(np.asarray(index))
        elif len(sent_freq)>max_sentence_length:
            # 句子长度超过最长句子长度，将其截断
            # 计算每个词的信息熵
            # print sent
            entropy = np.zeros(len(sent_freq))
            for items in sent_freq.transpose():
                entropy += -np.nan_to_num(np.log2(items))*items
            # 只取信息熵大的前n个词
            best_n_word = np.argsort(entropy)[:max_sentence_length]
            # 重新排序，不要打乱原有句子词序
            best_n_word.sort()
            # print best_n_word
            sent_freq = sent_freq[best_n_word]
            vector_prob.append(sent_freq)
            index = np.asarray(index)[best_n_word]
            indexs_padding.append(index)

        else:
            vector_prob.append(sent_freq)
            indexs_padding.append(np.asarray(index))


    # print len(vector_prob)

    vector_prob = [item.flatten() for item in vector_prob]
    array_to_str = lambda x: ','.join(['%.5f' % (item) for item in x])
    vector_prob = [array_to_str(item) for item in vector_prob]

    int_array_to_str = lambda x: ','.join([str(item) for item in x])
    indexs_padding = [int_array_to_str(item) for item in indexs_padding]

    data['VECTOR_PROB'] = vector_prob
    data['INDEXS_PADDING'] = indexs_padding

    return data,word2idx,(freq_pos,freq_neg,freq_non),target_dict

def testdata_to_vector(data,word2idx,freq,target_dict):
    '''
    生成句子向量，使用已有的字典等
    :param data:
    :param freq: (freq_pos,freq_neg,freq_non)
    :param target_dict:
    :return:
    '''
    freq_pos, freq_neg, freq_non = freq
    # 增加一个元素在最后的原因，是预留给OOV的词的，如果该词是OOV，则赋予概率为0
    # freq_pos.__add__(0)
    # freq_neg.__add__(0)
    # freq_non.__add__(0)
    logging.debug('开始使用已有字典转换数据成向量的形式...')
    sentences = [item.split(',') for item in data['SEGMENT_TEXT']]
    # 将句子转成整数的字典索引
    # 如果出现未知字符，则使用none字符填充
    sentence_to_index = lambda x: [word2idx.get(item, word2idx[PAD_CHAR]) for item in [START_CHAR] + x + [END_CHAR]]
    indexs = [sentence_to_index(items) for items in sentences]

    index_to_freq = lambda x: np.asarray([[freq_pos[item], freq_neg[item], freq_non[item]] for item in x])
    sentences_freqs = [index_to_freq(item) for item in indexs]


    # word_to_freq = lambda x: \
    #     np.asarray([[freq_pos[word2idx.get(item,-1)],
    #                  freq_neg[word2idx.get(item,-1)],
    #                  freq_non[word2idx.get(item,-1)]] for item in x])
    # sentences_freqs = [word_to_freq(item) for item in sentences]
    # 对句子补全或截断
    vector_prob = []
    indexs_padding = []
    max_sentence_length = MAX_SENTENCE_LENGTH
    for target, sent_freq, index in zip(data['TARGET'], sentences_freqs, indexs):
        # print sent_freq
        # print len(sent_freq)
        # print len(index)

        if len(sent_freq) < max_sentence_length:
            # 句子长度小于最长句子长度，将其padding
            # print len(sent)
            padding_length = max_sentence_length - len(sent_freq)

            # print '需要填充%d'%(padding_length)
            sentence_after_padding = np.concatenate([sent_freq, padding_length * [target_dict[target]]])
            vector_prob.append(sentence_after_padding)
            # print sentence_after_padding
            # print len(sentence_after_padding)
            # 使用none字符填充
            index = np.pad(index, (0, padding_length), mode='constant', constant_values=word2idx[PAD_CHAR])
            indexs_padding.append(np.asarray(index))
        elif len(sent_freq) > max_sentence_length:
            # 句子长度超过最长句子长度，将其截断
            # 计算每个词的信息熵
            # print sent
            entropy = np.zeros(len(sent_freq))
            for items in sent_freq.transpose():
                entropy += -np.nan_to_num(np.log2(items)) * items
            # 只取信息熵大的前n个词
            best_n_word = np.argsort(entropy)[:max_sentence_length]
            # 重新排序，不要打乱原有句子词序
            best_n_word.sort()
            # print best_n_word
            sent_freq = sent_freq[best_n_word]
            vector_prob.append(sent_freq)
            index = np.asarray(index)[best_n_word]
            indexs_padding.append(index)

        else:
            vector_prob.append(sent_freq)
            indexs_padding.append(np.asarray(index))
    # print len(vector_prob)
    # print vector_prob[0].shape
    # quit()
    vector_prob = [item.flatten() for item in vector_prob]
    float_array_to_str = lambda x: ','.join(['%.5f' % (item) for item in x])
    vector_prob = [float_array_to_str(item) for item in vector_prob]
    # print vector_prob[0]
    int_array_to_str = lambda x: ','.join([str(item) for item in x])
    indexs_padding = [int_array_to_str(item) for item in indexs_padding]


    data['VECTOR_PROB'] = vector_prob
    data['INDEXS_PADDING'] = indexs_padding
    return data



def main_processing_dataA():
    '''
    dataA的主处理过程
    :return:
    '''
    train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                            'evasampledata4-TaskAA.txt'


    train_dataA = pd.read_csv(train_dataA_file_path,
                              sep='\t',
                              header=0)
    logging.debug('show the detail of task A')
    # print_data_detail(train_dataA)
    # 过滤字符和分词
    train_dataA = clean_data(train_dataA,
                             columns=['TEXT'],
                             filter_char=True
                             )
    train_dataA = clean_data(train_dataA,
                             columns=['TARGET'],
                             filter_char=False
                             )

    # 处理空值数据
    train_dataA = processing_na_value(train_dataA,
                                      clear_na=True,
                                      fill_na=True,
                                      fill_char='NONE',
                                      columns=None)


    dev_dataA, test_dataA = split_train_test(train_dataA, train_split=0.7)
    #
    dev_dataA,word2idx,freq,target_dict = to_vector(dev_dataA)
    # print train_dataA.head()
    test_dataA = testdata_to_vector(test_dataA,word2idx,freq,target_dict)
    # print test_dataA.head()

    # print dev_dataA.head()
    dev_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                              'dev_dataA_%dlen.csv'%(MAX_SENTENCE_LENGTH)
    logging.debug('将dev data集保存到：%s'%(dev_dataA_result_path))
    dev_dataA.to_csv(dev_dataA_result_path,
                       sep='\t',
                       index=None,
                       encoding='utf8'
                       )
    # print dev_dataA.shape
    test_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                              'test_dataA_%dlen.csv'%(MAX_SENTENCE_LENGTH)
    logging.debug('将test data集保存到：%s'%(test_dataA_result_path))
    test_dataA.to_csv(test_dataA_result_path,
                       sep='\t',
                       index=None,
                       encoding='utf8'
                       )
    # print test_dataA.shape
    logging.debug('保存完成！')

def main_processing_dataB():
    '''
    该数据处理流程主要是：对空值进行处理，切分句子，保存到文件
    :return:
    '''
    train_dataB_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                            'evasampledata4-TaskAR.txt'
    train_dataB = pd.read_csv(train_dataB_file_path,
                              sep='\t',
                              header=0)
    logging.debug('show the detail of task B')
    data_detail(train_dataB,has_stance=False)
    # 过滤字符和分词
    train_dataB = clean_data(train_dataB,
                             columns=['TEXT'],
                             filter_char= True
                             )
    train_dataB = clean_data(train_dataB,
                             columns=['TARGET'],
                             filter_char= False
                             )

    train_dataB = processing_na_value(train_dataB,
                                      clear_na=True,
                                      fill_na=False,
                                      fill_char='',
                                      columns=['SEGMENT_TEXT']
                                      )

    dataB_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                            'dataA_%dlen.csv' % (MAX_SENTENCE_LENGTH)


    logging.debug('将data集保存到：%s' % (dataB_result_path))
    train_dataB.to_csv(dataB_result_path,
                       sep='\t',
                       index=None,
                       encoding='utf8'
                       )

if __name__ =='__main__':

    # main_processing_dataA()
    main_processing_dataB()
