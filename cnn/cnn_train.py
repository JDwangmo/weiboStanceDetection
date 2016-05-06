#encoding=utf8

from data_processing.load_data import load_data
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Activation,Flatten,Dense,Dropout
from keras.utils import np_utils

dev_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                        'dev_data.csv'

X_dev, y_dev = load_data(dev_dataA_result_path,
                         return_label=True
                         )
X_dev = X_dev.reshape(X_dev.shape[0],1,X_dev.shape[1],X_dev.shape[2])
y_dev_onehot = np_utils.to_categorical(y_dev,3)


test_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                         'test_data.csv'

X_test, y_test = load_data(test_dataA_result_path,
                           return_label=True
                           )
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1],X_test.shape[2])
y_test_onehot = np_utils.to_categorical(y_test,3)

# cnn arch
image_row,image_col = 173,3
model = Sequential()
layers = [
    Convolution2D(32,5,3,input_shape=(1,image_row,image_col),border_mode='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(10,1)),
    Convolution2D(32,5,1),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,1)),
    # Dropout(0.8),
    Flatten(),
    Dense(60),
    # Dropout(0.8),
    Activation('relu'),
    Dense(3),
    Activation('softmax')
]
[model.add(layer) for layer in layers]

# print model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy']
              )
# train
nb_epoch = 100
model.fit([X_dev],
          y_dev_onehot,
          batch_size=32,
          nb_epoch=nb_epoch,
          validation_data=(X_test,y_test_onehot),
          shuffle=True,
          verbose=1
          )
# print model.evaluate(X_test,y_test_onehot)

# print model.predict_classes(X_test)


# save
json_string = model.to_json()
model_architecture = '/home/jdwang/PycharmProjects/weiboStanceDetection/cnn/model/' \
                     'cnn_model_architecture_%depoch.json'%(nb_epoch)
open(model_architecture, 'w').write(json_string)
model_weights = '/home/jdwang/PycharmProjects/weiboStanceDetection/cnn/model/' \
                'cnn_model_weights_%depoch.h5'%(nb_epoch)
model.save_weights(model_weights,overwrite=True)

