#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-07-02'
__email__ = '383287471@qq.com'

import numpy as np
import pandas as pd
import logging
import timeit


def train():

    taskAA_data = pd.read_csv(TaskAA_data_file,
                encoding='utf8',
                sep='\t',
                header=0)
    taskAR_data = pd.read_csv(TaskAR_data_file,
                        encoding='utf8',
                        sep='\t',
                        header=0)
    TaskA_all_test_data = pd.read_csv(TaskA_all_test_file,
                        encoding='utf8',
                        sep='\t',
                        header=0)
    TaskA_all_test_data = TaskA_all_test_data[TaskA_all_test_data['WORDS'].notnull()]

    print taskAA_data.shape
    print taskAR_data.shape
    print TaskA_all_test_data.shape
    # print data2.head()
    if type==5997:
        sentences = np.concatenate((taskAA_data['WORDS'].as_matrix(),taskAR_data['WORDS'].as_matrix()),axis=0)
    else:
        sentences = np.concatenate((taskAA_data['WORDS'].as_matrix(),taskAR_data['WORDS'].as_matrix(),TaskA_all_test_data['WORDS'].as_matrix()),axis=0)
    print sentences.shape


    util.train(sentences)
    util.save('vector/train_data_full_%d_%ddim_%diter_%s.gem'%(len(sentences),
                                                               size,
                                                               iter,
                                                               train_method,
                                                               ))

def test():
    # type = 20963
    model = util.load('vector/train_data_full_%d_%ddim_%diter_%s.gem'%(type,
                                                                       size,
                                                                       iter,
                                                                       train_method
                                                                       ))

    most_similar_words = model.most_similar(u'iphone')
    print most_similar_words
    print ','.join([i for i, j in most_similar_words])
    most_similar_words = model.most_similar(u'喜欢')
    print most_similar_words
    print ','.join([i for i, j in most_similar_words])
    most_similar_words = model.most_similar(u'讨厌')
    print most_similar_words
    print ','.join([i for i, j in most_similar_words])
    most_similar_words = model.most_similar(u'底层')
    print most_similar_words
    print ','.join([i for i, j in most_similar_words])


from data_processing_util.word2vec_util.word2vec_util import Word2vecUtil
# def train_3000
TaskAA_data_file = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskAA_all_data_3000.csv'
TaskAR_data_file = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskAR_all_data_2997.csv'
TaskA_all_test_file = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/TaskA_all_testdata_15000.csv'

type,size,iter,train_method= 5997,50,50,'skip'
type,size,iter,train_method= 20963,50,50,'skip'

type,size,iter,train_method= 5997,50,50,'cbow'
type,size,iter,train_method= 20963,50,50,'cbow'

type,size,iter,train_method= 20963,300,50,'cbow'

# type,size,iter,train_method= 5997,50,5,'skip'
# type,size,iter,train_method= 20963,50,5,'skip'
#
# type,size,iter,train_method= 5997,50,5,'cbow'
# type,size,iter,train_method= 20963,50,5,'cbow'


util = Word2vecUtil(size=size,
                    train_method=train_method,
                    iter=iter,
                    )

train()
test()


'''
[(u'\u914d\u7f6e', 0.922458291053772), (u'\u8fd9\u6b3e', 0.9009464979171753), (u'\u62ff\u5230', 0.8862029910087585), (u'\u82f1\u5bf8', 0.8822333812713623), (u'\u4ef7\u683c\u6bd4', 0.8798030018806458), (u'ipad', 0.877876341342926), (u'\u57fa\u672c\u4e00\u81f4', 0.8750308752059937), (u'\u82f9\u679c', 0.8719356656074524), (u'pro', 0.8665716052055359), (u'\u7ffb\u65b0', 0.8638173937797546)]
配置,这款,拿到,英寸,价格比,ipad,基本一致,苹果,pro,翻新
[(u'\u5f90\u6d77', 0.8884794116020203), (u'\u75bc\u7231', 0.8816609382629395), (u'\u770b\u5f97\u51fa', 0.8788071274757385), (u'\u4e0d\u8fc7', 0.8751739263534546), (u'\u60ca\u8bb6', 0.8714367151260376), (u'\u6000\u604b', 0.866191565990448), (u'\u89c9\u5f97', 0.8654507398605347), (u'\u53e4\u602a', 0.8633217811584473), (u'\u6709\u70b9', 0.857458233833313), (u'\u5f88', 0.8566928505897522)]
徐海,疼爱,看得出,不过,惊讶,怀恋,觉得,古怪,有点,很
[(u'\u4eba\u7406', 0.9271324872970581), (u'\u8212\u5766', 0.9260797500610352), (u'\u653e\u8fc7', 0.9249367713928223), (u'\u8bdd', 0.9237825274467468), (u'\u5fc3\u91cc', 0.9217459559440613), (u'\u5ac1', 0.920974612236023), (u'\u6bcf\u6b21', 0.9208803772926331), (u'\u5bb3\u6015', 0.9200080633163452), (u'\u5b9e\u8bdd', 0.9199990034103394), (u'\u53cd\u6b63', 0.9188157916069031)]
人理,舒坦,放过,话,心里,嫁,每次,害怕,实话,反正
[(u'\u666e\u901a', 0.9671438932418823), (u'\u6700\u5e95\u5c42', 0.9643466472625732), (u'\u65b9\u6cd5', 0.9580934643745422), (u'\u627e\u51fa', 0.9523003101348877), (u'\u4ed6\u4eba', 0.9500033855438232), (u'\u767e\u59d3\u751f\u6d3b', 0.9421230554580688), (u'\u4e0b\u5c42', 0.9415550827980042), (u'\u5c0f\u770b\u4eba', 0.9411033391952515), (u'\u5e73\u6c11', 0.9399583339691162), (u'\u4e00\u5207', 0.9395464658737183)]
普通,最底层,方法,找出,他人,百姓生活,下层,小看人,平民,一切

    skip 50d,5iter
[(u'se', 0.9440841674804688), (u's', 0.8551413416862488), (u'iphone5s', 0.8486535549163818), (u'\u82f1\u5bf8', 0.8455590009689331), (u'iphonese', 0.8415039777755737), (u'\u5916\u89c2', 0.824216365814209), (u'\u914d\u7f6e', 0.8215358853340149), (u'iphone6s', 0.8196059465408325), (u'\u8bbe\u8ba1', 0.8169116973876953), (u'iphone5se', 0.8165431618690491)]
se,s,iphone5s,英寸,iphonese,外观,配置,iphone6s,设计,iphone5se
[(u'\u610f\u5411', 0.7531678080558777), (u'\u5c0f\u5de7', 0.7438930869102478), (u'\u89c9\u5f97', 0.739881157875061), (u'\u9002\u5408', 0.7385444641113281), (u'\u592a\u5c0f', 0.7341982126235962), (u'\u60ec\u610f', 0.7274749875068665), (u'\u5b9e\u8bdd', 0.7262848615646362), (u'\u7279\u610f', 0.7252548933029175), (u'\u771f\u5207', 0.724941611289978), (u'\u8bf4\u5b9e\u8bdd', 0.7216389179229736)]
意向,小巧,觉得,适合,太小,惬意,实话,特意,真切,说实话
[(u'\u54b3\u55fd', 0.8587285876274109), (u'\u534a\u591c\u4e09\u66f4', 0.8510130047798157), (u'\u7a00\u996d', 0.8487115502357483), (u'\u5435', 0.8462694883346558), (u'\u8212\u5766', 0.8456923365592957), (u'\u6e05\u9759', 0.8454961776733398), (u'\u603b\u4e4b', 0.84461510181427), (u'\u89c9\u5f97', 0.8402853012084961), (u'\u76f2\u76ee', 0.8391507863998413), (u'\u9690\u9690\u4f5c\u75db', 0.8386300206184387)]
咳嗽,半夜三更,稀饭,吵,舒坦,清静,总之,觉得,盲目,隐隐作痛
[(u'\u793e\u4f1a\u5e95\u5c42', 0.9058681726455688), (u'\u6700\u5e95\u5c42', 0.895221471786499), (u'\u666e\u901a\u767e\u59d3', 0.8735599517822266), (u'\u8eab\u5904', 0.8649846315383911), (u'\u538b\u8feb', 0.8559167981147766), (u'\u5265\u593a', 0.8540997505187988), (u'\u5e73\u6c11\u767e\u59d3', 0.8509180545806885), (u'\u6c11\u767e', 0.8495404720306396), (u'\u751f\u5b58\u7a7a\u95f4', 0.8415502309799194), (u'\u9636\u5c42', 0.8395096659660339)]
社会底层,最底层,普通百姓,身处,压迫,剥夺,平民百姓,民百,生存空间,阶层
'''