#encoding=utf8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from data_processing.load_data import load_data


dev_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                  'dev_dataA_150len.csv'


dev_data = load_data(dev_dataA_file_path)

test_dataA_file_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                  'test_dataA_150len.csv'


test_data = load_data(test_dataA_file_path)

print dev_data.head()
dev_data['SEGMENT_SENTENCES'] = dev_data['SEGMENT_SENTENCES'].apply(lambda x:' '.join(x.split(',')))
X_dev = [items for items in dev_data['SEGMENT_SENTENCES'] + ' ' + dev_data['TARGET']]
print X_dev
quit()

#
# #clf = SGDClassifier(loss='hinge', penalty='l1', n_iter=20, shuffle=True, verbose=True, n_jobs=2, average=False)
# train_d = [tweet for tweet in training_data['Tweet'] + ' ' + training_data['Target']]
# eval_d = [tweet for tweet in eval_data['Tweet'] + ' ' + eval_data['Target']]
#
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=1.0, min_df=1, binary=True, norm='l2', use_idf=True, smooth_idf=False, sublinear_tf=True, encoding='latin1')
#
# X_train = vectorizer.fit_transform(train_d)
# X_eval = vectorizer.transform(eval_d)
# Y_train = np.asarray([stance for stance in training_data['Stance']])
# Y_eval = np.asarray([stance for stance in eval_data['Stance']])
#
# tuned_parameters = {'alpha': [10 ** a for a in range(-12, 0)]}
# clf = GridSearchCV(SGDClassifier(loss='hinge', penalty='elasticnet',l1_ratio=0.75, n_iter=10, shuffle=True, verbose=False, n_jobs=4, average=False)
#                   , tuned_parameters, cv=10, scoring='f1_weighted')
# clf.fit(X_train, Y_train)
#
