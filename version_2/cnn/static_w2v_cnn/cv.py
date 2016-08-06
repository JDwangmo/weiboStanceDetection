# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-06'
    Email:   '383287471@qq.com'
    Describe:
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

from deep_learning.cnn.wordEmbedding_cnn.example.one_conv_layer_wordEmbedding_cnn import WordEmbeddingCNNWithOneConv

input_length = 120
word_embedding_dim = 50
WordEmbeddingCNNWithOneConv.cross_validation(
    train_data=(train_x, train_y),
    test_data=(test_x, test_y),
    feature_type = 'word',
    input_length=input_length,
    num_filter_list=[10],
    # num_filter_list=[10,30,50, 80, 100, 110, 150, 200, 300,500,1000],
    verbose=1,
    # word2vec_model_file_path = data_util.transform_word2vec_model_name('%dd_weibo_100w' % word_embedding_dim),
    word2vec_model_file_path = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/vector1000000_50dim.gem'


)