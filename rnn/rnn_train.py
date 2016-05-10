#encoding=utf8

from data_processing.load_data import load_data_indexs
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Embedding,Activation,Input,merge,Dropout,Merge
from keras.utils import np_utils
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )

MAX_SENTENCE_LENGTH = 150
DICT_SIZE = 15465

dev_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                        'dev_dataA_150len.csv'


X_dev, y_dev = load_data_indexs(dev_dataA_result_path,
                                return_label=True
                                )

y_dev_onehot = np_utils.to_categorical(y_dev,3)


test_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                         'test_dataA_150len.csv'

X_test, y_test = load_data_indexs(test_dataA_result_path,
                                  return_label=True
                                  )

y_test_onehot = np_utils.to_categorical(y_test,3)


print('Build model...')
model = Sequential()
# this is the placeholder tensor for the input sequences
# input_sentence = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
# # this embedding layer will transform the sequences of integers
# # into vectors of size 128
# embedded = Embedding(DICT_SIZE, 128)(input_sentence)
#
# # apply forwards LSTM
# forwards = LSTM(64)(embedded)
# # apply backwards LSTM
# backwards = LSTM(64, go_backwards=True)(embedded)
#
# # concatenate the outputs of the 2 LSTMs
# merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
# Merge()
# after_dp = Dropout(0.5)(merged)
# output = Dense(1, activation='sigmoid')(after_dp)
#
# model = Model(input=input_sentence, output=output)

m1 = Sequential()
m1.add(Embedding(DICT_SIZE,300, dropout=0.9,input_length=MAX_SENTENCE_LENGTH))
m1.add(LSTM(128, dropout_W=0.8,
            dropout_U=0.8,
            input_shape=(MAX_SENTENCE_LENGTH,300),
            return_sequences=True))
m2 = Sequential()
m2.add(Embedding(DICT_SIZE,300, dropout=0.9,input_length=MAX_SENTENCE_LENGTH))
m2.add(LSTM(128, dropout_W=0.8,
            dropout_U=0.8,input_shape=(MAX_SENTENCE_LENGTH,300),
            go_backwards=True,
            return_sequences=True))
model.add(Merge([m1,m2],mode='concat'))
print model.summary()
quit()
model.add(LSTM(128))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
batch_size = 32
nb_epoch = 15
model.fit(X_dev, y_dev_onehot,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          validation_data=(X_test, y_test_onehot)
          )


# save
json_string = model.to_json()
model_architecture = '/home/jdwang/PycharmProjects/weiboStanceDetection/rnn/model/' \
                     'rnn_model_architecture_%depoch_%dlen.json'%(nb_epoch,MAX_SENTENCE_LENGTH)
open(model_architecture, 'w').write(json_string)
model_weights = '/home/jdwang/PycharmProjects/weiboStanceDetection/rnn/model/' \
                'rnn_model_weights_%depoch_%dlen.h5'%(nb_epoch,MAX_SENTENCE_LENGTH)
model.save_weights(model_weights,overwrite=True)

