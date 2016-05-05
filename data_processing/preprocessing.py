#encoding=utf8
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )

train_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                       'evasampledata4-TaskAA.txt'

train_dataB_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                       'evasampledata4-TaskAR.txt'

train_dataA = pd.read_csv(train_dataA_file_path,
                          sep='\t',
                          header=0)
train_dataB = pd.read_csv(train_dataB_file_path,
                          sep='\t',
                          header=0)

logging.debug('taskA的个数为：%d'%(len(train_dataA)))
logging.debug('taskB的个数为：%d'%(len(train_dataB)))
logging.debug('taskA的sample数据：')
logging.debug( train_dataA.head())
logging.debug('taskB的sample数据：')
logging.debug(train_dataB.head())

logging.debug('taskA的taget有：')
logging.debug(train_dataA['TARGET'].value_counts())
logging.debug('taskB的taget有：')
logging.debug(train_dataB['TARGET'].value_counts())

group = train_dataA.groupby(by=['TARGET','STANCE'])
print group.count()
