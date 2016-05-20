# encoding=utf8
import logging
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# size = 50

vector_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/word2vec/model/' \
                   'weibo_50size.gem'

model = Word2Vec.load(vector_file_path)
word = u'恩'
print model.most_similar([word])
print ','.join([item for item,_ in model.most_similar([word])])
# print ','.join([item for item,_ in model.most_similar([word])])


