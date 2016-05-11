#encoding=utf8

import logging
from data_processing.load_data import load_data_segment
from gensim.models import Word2Vec
import numpy as np
import timeit
start = timeit.default_timer()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname):
            yield line


# sentences = MySentences('/media/jdwang/My Book/复旦大学/SMP-Weibo/sample/lv3_82_seg.csv')

end = timeit.default_timer()

# train
size = 100
sentences = MySentences('/media/jdwang/My Book/复旦大学/SMP-Weibo/sample/lv3_82_seg.csv')
model = Word2Vec(sentences=sentences,
                 size = 50,
                 min_count = 1,
                 workers = 10
                 )
print '用时：%ds'%(end - start)
# quit()
# save
vector_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/word2vec/model/' \
                   'weibo_%dsize.bin'%(size)

model.save_word2vec_format(vector_file_path,binary=True)
logging.info('save the word2vec model')
