# encoding=utf8
import logging
from data_processing.load_data import load_data_segment
from gensim.models import Word2Vec
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                  'dataA_150len.csv'

dataB_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                  'dataB_150len.csv'

dataA = load_data_segment(dataA_file_path)
dataB = load_data_segment(dataB_file_path)

sentences = np.concatenate((dataA,dataB),axis=0)
# train
size = 100
model = Word2Vec(sentences=sentences,
                 size = 100,
                 min_count = 1,
                 workers = 10
                 )
# save
vector_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/word2vec/model/' \
                   'vector_%dsize.bin'%(size)

model.save_word2vec_format(vector_file_path,binary=True)
logging.info('save the word2vec model')