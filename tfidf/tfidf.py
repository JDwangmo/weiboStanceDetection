#encoding=utf8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.grid_search import GridSearchCV
from data_processing.load_data import load_data


dev_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                  'dev_dataA_150len.csv'


dev_data = load_data(dev_dataA_file_path)

test_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                  'test_dataA_150len.csv'


test_data = load_data(test_dataA_file_path)

X_train = [items for items in dev_data['SEGMENT_TEXT'] + ',' +dev_data['SEGMENT_TARGET']]
X_train = [' '.join(items.split(',')) for items in X_train]
X_eval = [items for items in test_data['SEGMENT_TEXT'] + ',' +test_data['SEGMENT_TARGET']]
X_eval = [' '.join(items.split(',')) for items in X_eval]

Y_train = np.asarray([stance for stance in dev_data['STANCE']])
Y_eval = np.asarray([stance for stance in test_data['STANCE']])

vectorizer = TfidfVectorizer(ngram_range=(1,1),
                             max_df=1.0,
                             lowercase=False,
                             min_df=1,
                             binary=True,
                             norm='l2',
                             use_idf=True,
                             smooth_idf=False,
                             sublinear_tf=True,
                             token_pattern=u'(?u)\\b\w+\\b',
                             encoding='utf8')


X_train = vectorizer.fit_transform(X_train)
X_eval = vectorizer.transform(X_eval)
# print vectorizer.get_feature_names()
# print vectorizer[u'IphoneSE']

tuned_parameters = {'alpha': [10 ** a for a in range(-12, 0)]}
clf = GridSearchCV(SGDClassifier(loss='hinge',
                                 random_state=0,
                                 penalty='elasticnet',
                                 l1_ratio=0.75,
                                 n_iter=10,
                                 shuffle=True,
                                 verbose=True,
                                 n_jobs=4,
                                 average=False
                                 ),
                   tuned_parameters,
                   cv=10,
                   scoring='f1_weighted'
                   )
clf.fit(X_train, Y_train)
pred = clf.predict(X_eval)
is_correct = (pred == Y_eval)
print sum(is_correct)/(len((pred))*1.0)
print clf.best_params_


test_data['PREDICT'] = pred
test_data['IS_CORRECT'] = is_correct

result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/tfidf/result/' \
              'tfidf_result.csv'

test_data[[u'﻿ID', 'TARGET', 'TEXT', 'STANCE', 'IS_CORRECT', 'PREDICT']].to_csv(
    result_path,
    sep='\t',
    index=None,
    encoding='utf8'
)