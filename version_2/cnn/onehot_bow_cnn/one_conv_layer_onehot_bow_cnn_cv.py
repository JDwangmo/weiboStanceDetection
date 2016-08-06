# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-01'
    Email:   '383287471@qq.com'
    Describe: 单卷积层的CNN（BOC）
        1. 读取数据 S数据集
        2. 交叉验证
"""

config={
    'verbose':1,
}

from version_2.data_processing.data_util import DataUtil

data_util = DataUtil()
train_data, test_data = data_util.load_train_test_data(config)
label_to_index, index_to_label = data_util.get_label_index()

train_x = train_data['TEXT'].as_matrix()
train_y = train_data['STANCE_INDEX'].as_matrix()
test_x = test_data['TEXT'].as_matrix()
test_y = test_data['STANCE_INDEX'].as_matrix()

from deep_learning.cnn.wordEmbedding_cnn.example.one_conv_layer_onehot_bow_cnn import OnehotBowCNNWithOneConv

input_length = 120
OnehotBowCNNWithOneConv.cross_validation(
    train_data=(train_x, train_y),
    test_data=(test_x, test_y),
    input_length=input_length,
    feature_type = 'word',
    num_filter_list=[80, 100, 110, 150, 200],
    # num_filter_list=[80],
    # region_size_list=range(1,14),
    region_size_list=[14],
    verbose=1,
    word2vec_to_solve_oov=False,
    word2vec_model_file_path = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/vector1000000_50dim.gem'
)
