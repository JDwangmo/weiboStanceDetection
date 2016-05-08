#encoding=utf8

from data_processing.load_data import load_data_indexs
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from keras.utils import np_utils
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )


dev_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                        'dev_data_150len.csv'


X_dev, y_dev = load_data_indexs(dev_dataA_result_path,
                                return_label=True
                                )

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))