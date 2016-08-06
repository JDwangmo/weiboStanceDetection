# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-06'
    Email:   '383287471@qq.com'
    Describe: RF(BOW/BOC)
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



from traditional_classify.bow_rf.bow_rf_model import BowRandomForest

BowRandomForest.cross_validation(
    train_data=(train_x, train_y),
    test_data=(test_x, test_y),
    shuffle_data = False,
    # n_estimators_list = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,1000,2000,3000,4000,5000],
    n_estimators_list = range(10,1010,10),
    # n_estimators_list = [610],
    verbose=0,
    feature_type = 'word',
    word2vec_to_solve_oov=False,
    word2vec_model_file_path = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/vector1000000_50dim.gem'


)