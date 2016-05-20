#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-05-20'
import numpy as np
import pandas as pd
import logging
import timeit
import yaml

logging.basicConfig(filename='imdb_20160520.log', filemode='w', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import nltk

data_file_path = '/home/jdwang/PycharmProjects/uciTest/dataset/sentiment labelled sentences/' \
                 'sentiment labelled sentences/imdb_labelled.txt'

data  = pd.read_csv(data_file_path,sep='\t',header=None)

sentences = data[0]
labels = data[1].as_matrix()
# print sentences
sentences_clear = []
for line in sentences:
    line = line.strip()
    seg = []
    # print line
    for item in nltk.word_tokenize(line):
        if len(item)>1:
            # print item
            seg.append(item)
    # print seg
    # print line
    sentences_clear.append( ' '.join(seg))

print sentences_clear
# quit()
# labels = np.asarray(labels,dtype=np.float64)
logging.debug('总共训练个数为：%d'%(len(labels)))

MAX_FEATURES = 2000

# logging.debug('使用模型：%s'%(CHOICES))
vectorizer = CountVectorizer(analyzer="word",
                             token_pattern=u'(?u)\\b\w+\\b',
                             tokenizer=None,
                             preprocessor=None,
                             lowercase=False,
                             stop_words=None,
                             max_features=MAX_FEATURES)

data_features = vectorizer.fit_transform(sentences)
data_features = data_features.toarray()
data_features = data_features.astype(np.float64)
# print data_features
logging.debug('字典大小：%d'%(len(vectorizer.get_feature_names())))
# quit()
num_train = int(0.7*len(labels))
num_test = len(labels) - num_train
logging.debug('训练集大小：%d'%(num_train))
logging.debug('测试集大小：%d'%(num_test))

train_data_features = data_features[:num_train]
train_labels = labels[:num_train]
test_data_features = data_features[num_train:]
test_labels = labels[num_train:]

def forest():

    # Initialize a Random Forest classifier with 100 trees
    n_estimators = 150
    logging.debug('Initialize a Random Forest classifier with %d trees'%(n_estimators))
    forest = RandomForestClassifier(n_estimators = n_estimators,random_state=0)


    forest = forest.fit(train_data_features,train_labels)

    result = forest.predict(test_data_features)
    return result

def svm():
    model = SVC()
    model.fit(X =train_data_features,y=train_labels)
    # print model
    result = model.predict(X=test_data_features)
        # libsvm.fit()
    print result
    return result



start = timeit.default_timer()
def knn():
    model = KNeighborsClassifier(n_neighbors=5,
                                 weights='distance',
                                 algorithm='kd_tree',
                                 leaf_size=30,
                                 p=2,
                                 metric='minkowski',
                                 metric_params=None,
                                 n_jobs=10
                                 )

    model.fit(train_data_features,train_labels)
    result = model.predict(test_data_features)
    return result

# result = forest()
# result = svm()
result = knn()



is_correct = test_labels == result


print sum(is_correct)
print sum(is_correct)/(1.0*num_test)
logging.debug( '准确率：%f'%(sum(is_correct)/(1.0*num_test)))
end = timeit.default_timer()
logging.debug('总共运行时间:%ds' % (end-start))
