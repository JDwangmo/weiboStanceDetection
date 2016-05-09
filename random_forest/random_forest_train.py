#encoding=utf8

from data_processing.load_data import load_data_indexs
import logging
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )



dev_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                        'dev_data_150len.csv'

X_dev, y_dev = load_data_indexs(dev_dataA_result_path,
                                return_label=True
                                )


test_dataA_result_path = '/home/jdwang/PycharmProjects/weiboStanceDetection/train_data/' \
                         'test_data_150len.csv'

X_test, y_test = load_data_indexs(test_dataA_result_path,
                                  return_label=True
                                  )


model = GradientBoostingClassifier(learning_rate=0.1,
                                   n_estimators=100,
                                   subsample=1.0,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.,
                                   max_depth=3,
                                   random_state=0,
                                   max_features=None
                                   )
validator = GridSearchCV(model,
                         param_grid={
                             'learning_rate':np.arange(0.1,1,0.1),
                             'n_estimators': range(100,150),
                             'max_depth':range(3,5),
                         },
                         scoring='log_loss',
                         verbose=1,
                         n_jobs=5
                         )

validator.fit(X_dev,y_dev)
print validator.best_params_
print validator.best_score_
quit()
y_pred = model.predict(X_test)
is_correct = (y_pred == y_test)
print sum(is_correct)
print sum(is_correct)/(len(y_test)*1.0)
