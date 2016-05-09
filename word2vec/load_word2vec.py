# encoding=utf8
import logging
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

size = 100

vector_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/word2vec/model/' \
                   'vector_%dsize.bin'%(size)

model = Word2Vec.load_word2vec_format(vector_file_path ,binary=True)
print model.most_similar([u'热情'])
print ','.join([item for item,_ in model.most_similar([u'热情'])])


