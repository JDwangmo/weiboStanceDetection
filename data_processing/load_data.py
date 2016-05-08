#encoding=utf8
import pandas as pd
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )

target2idx = {'FAVOR':1,'AGAINST':0,'NONE':2}

def load_data(file_path,return_label = False):
    '''
    加载数据的概率向量
    :param file_path:
    :param return_label: 是否返回标签
    :return:
    '''
    logging.debug('加载data集from：%s' % (file_path))
    data = pd.read_csv(file_path,
                       sep='\t',
                       encoding='utf8',
                       header=0
                       )

    image_size = (150,3)

    str_to_array = lambda x: np.asarray(x.split(','),dtype=float).reshape(image_size)
    X = np.asarray([str_to_array(item) for item in data['VECTOR_PROB']])
    logging.debug('data的shape为：%d,%d,%d'%(X.shape))


    if return_label:
        logging.debug('返回标签')
        y = data['STANCE'].apply(lambda x:target2idx[x])
        return X,y

    return X

def load_data_indexs(file_path,return_label = False):
    '''
    加载数据的字典索引
    :param file_path:
    :param return_label: 是否返回标签
    :return:
    '''
    logging.debug('加载data集from：%s' % (file_path))
    data = pd.read_csv(file_path,
                       sep='\t',
                       encoding='utf8',
                       header=0
                       )


    str_to_array = lambda x: np.asarray(x.split(','), dtype=int)
    X = np.asarray([str_to_array(item) for item in data['INDEXS_PADDING']])
    logging.debug('data的shape为：%d,%d' % (X.shape))

    if return_label:
        logging.debug('返回标签')
        y = data['STANCE'].apply(lambda x: target2idx[x])
        return X, y

    return X


if __name__=='__main__':
    dev_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                            'dev_data_150len.csv'

    X_dev,y_dev = load_data(dev_dataA_result_path,
                    return_label = True
                    )

    X_dev,y_dev = load_data_indexs(dev_dataA_result_path,
                                   return_label = True
                                   )
    # print X_dev.shape
    # print y_dev.shape

    test_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                             'test_data_150len.csv'

    X_test, y_test = load_data(test_dataA_result_path,
                     return_label=True
                     )
    # print X_test.shape
    # print y_test.shape