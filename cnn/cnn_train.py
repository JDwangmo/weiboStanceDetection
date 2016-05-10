#encoding=utf8

from data_processing.load_data import load_data__prob
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Activation,Flatten,Dense,Dropout,Merge,Input
from keras.utils import np_utils
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )



dev_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                        'dev_dataA_150len.csv'

X_dev, y_dev = load_data__prob(dev_dataA_result_path,
                               return_label=True
                               )
X_dev = X_dev.reshape(X_dev.shape[0],1,X_dev.shape[1],X_dev.shape[2])
y_dev_onehot = np_utils.to_categorical(y_dev,3)


test_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                         'test_dataA_150len.csv'

X_test, y_test = load_data__prob(test_dataA_result_path,
                                 return_label=True
                                 )
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1],X_test.shape[2])
y_test_onehot = np_utils.to_categorical(y_test,3)

# cnn arch
image_row,image_col = 150,3
# 创建一个模型
model = Sequential()
sequence = Input(shape=(1, image_row, image_col), dtype='float32')
# 多窗口核卷积层
cnn_layers = []
n_wins = [5,10,20,25,30,40,45,50]
logging.debug('使用%d种卷积核，大小分别为：%s'%(len(n_wins),str(n_wins)))
for win_size in n_wins:

    m = Sequential()
    layers = [
        Convolution2D(64, win_size, 3, input_shape=(1, image_row, image_col), border_mode='valid'),
        Activation('relu'),
        MaxPooling2D(pool_size=(10, 1)),
        Dropout(0.6),
        Convolution2D(32, 4, 1),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 1)),
        Dropout(0.5),
        Flatten(),
        # Dense(60),
        # Dropout(0.8),
    ]
    [m.add(layer) for layer in layers]
    cnn_layers.append(m)


model.add(Merge(layers=cnn_layers,
                concat_axis=1,
                mode='concat'
                )
          )
# 隐含层
hidden_layers = [
    Dense(60),
    Dropout(0.8),
    Activation('relu'),
    Dense(3),
    Activation('softmax')
]
[model.add(layer) for layer in hidden_layers]


# 单窗口核卷积层
# layers = [
#     Convolution2D(32,5,3,input_shape=(1,image_row,image_col),border_mode='valid'),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(10,1)),
#     # Dropout(0.8),
#     Convolution2D(32,5,1),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2,1)),
#     # Dropout(0.8),
#     Flatten(),
#     Dense(60),
#     # Dropout(0.8),
#     Activation('relu'),
#     Dense(3),
#     Activation('softmax')
# ]
# [model.add(layer) for layer in layers]

# print model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy']
              )
# train

nb_epoch = 5
model.fit([X_dev]*len(n_wins),
          y_dev_onehot,
          batch_size=32,
          nb_epoch=nb_epoch,
          validation_data=([X_test]*len(n_wins),y_test_onehot),
          shuffle=True,
          verbose=1
          )
# print model.evaluate(X_test,y_test_onehot)

# print model.predict_classes(X_test)


# save
json_string = model.to_json()
model_architecture = '/home/jdwang/PycharmProjects/weiboStanceDetection/cnn/model/' \
                     'cnn_model_architecture_%depoch_%dwin.json'%(nb_epoch,len(n_wins))
open(model_architecture, 'w').write(json_string)
model_weights = '/home/jdwang/PycharmProjects/weiboStanceDetection/cnn/model/' \
                'cnn_model_weights_%depoch_%dwin.h5'%(nb_epoch,len(n_wins))
model.save_weights(model_weights,overwrite=True)

