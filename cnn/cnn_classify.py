#encoding=utf8
import pandas as pd
from data_processing.load_data import load_data
from keras.models import model_from_json
from keras.utils import np_utils

target2idx = {'FAVOR':1,'AGAINST':0,'NONE':2}
idx2target = target2idx.keys()

test_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                         'test_data.csv'

X_test, y_test = load_data(test_dataA_result_path,
                           return_label=True
                           )
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1],X_test.shape[2])
y_test_onehot = np_utils.to_categorical(y_test,3)
# 多少次迭代
nb_epoch = 100
# 多少种类型的卷积核
num_wins = 8
model_architecture = '/home/jdwang/PycharmProjects/weiboStanceDetection/cnn/model/' \
                     'cnn_model_architecture_%depoch_%dwin.json'%(nb_epoch,num_wins)
model_weights = '/home/jdwang/PycharmProjects/weiboStanceDetection/cnn/model/' \
                'cnn_model_weights_%depoch_%dwin.h5'%(nb_epoch,num_wins)

# load
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)


# print model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy']
              )


y_pred = model.predict_classes([X_test]*num_wins)
is_correct = sum(y_pred==y_test)
accuary = is_correct / (len(y_test)*1.0)
print '正确的个数：%d,准确率：%f'%(is_correct,accuary)

test_data = pd.read_csv('/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/test_data.csv',
                       sep='\t',
                       encoding='utf8',
                       header=0
                       )
test_data['IS_CORRECT'] =  (y_pred==y_test)
test_data['PREDICT'] =  [idx2target[item] for item in y_pred]
result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/cnn/result/' \
              'cnn_result_%depoch_%dwin.csv'%(nb_epoch,num_wins)

test_data[[u'﻿ID','TARGET','TEXT','STANCE','IS_CORRECT','PREDICT']].to_csv(
    result_path,
    sep='\t',
    index=None,
    encoding='utf8'
)