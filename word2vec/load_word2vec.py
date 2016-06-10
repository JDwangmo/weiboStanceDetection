# encoding=utf8
import logging
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# size = 50
print '------'

vector_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/word2vec/model/' \
                   'vector1000000.gem'

model = Word2Vec.load(vector_file_path)
print '------'
word = u'呃'
print model.most_similar([word])
print ','.join([item for item,_ in model.most_similar([word])])
# print ','.join([item for item,_ in model.most_similar([word])])


