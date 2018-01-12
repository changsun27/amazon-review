#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ChangSun
"""
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import random
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem import WordNetLemmatizer
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import gensim
#import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


def parse(path):
  g = open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def getRatio(df):
    upvote=[]
    totalvote=[]
    for row in range(len(df)):
        print(row)
        u,t = df['helpful'].iloc[row]
        upvote.append(u)
        totalvote.append(t)
    
    upvote=pd.Series(upvote)   
    df['upvote'] = upvote.values
    
    totalvote=pd.Series(totalvote)   
    df['totalvote'] = totalvote.values
    
    df['ratio'] = df['upvote']/df['totalvote']
    return(df)

def normalize_review_text(text):
    tokens = nltk.regexp_tokenize(text.lower(), word_pattern)
    wordlst = [stemmer.stem(token) for token in tokens 
               if token not in stop_words and len(token) > 2]
    return ' '.join(wordlst)


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        np.random.shuffle(self.sentences)
        return self.sentences

   
df1 = getDF('Movies_and_TV_5.json') #1697533
df1 = getRatio(df1)
df2 = getDF('CDs_and_Vinyl_5.json')  #1097592
df2 = getRatio(df2)
#test=df.iloc[0:10,]

#### explore data
####remove na values in ratio
df1_nona = df1[np.isfinite(df1['ratio'])]  #1088752
df2_nona = df2[np.isfinite(df2['ratio'])]  #801567

### keep total review greater than 10
df1_10 = df1_nona[df1_nona.totalvote >= 10] #213606
df2_10 = df2_nona[df2_nona.totalvote >= 10] #164635

df = pd.concat([df1_10, df2_10]) #378241

(df['ratio'] > 0.5).values.sum() #263540
(df['ratio'] <= 0.5).values.sum() #114701

df.to_csv('review-g-10.csv', index=False)

# remove nan in review text 
df= pd.read_csv('review-g-10.csv')
df['reviewText'].isnull().sum()
df_no_review = df
df = df.dropna(axis=0, how='any')   #377210
df.isnull().sum()

df.to_csv('review-g-10-no-na.csv', index=False)

#histogram of ratio distribution
plt.hist(df['ratio'], color='skyblue', ec='black')
plt.title('Helpful Ratio for Unsampled Data')
plt.xlabel('Helpful Ratio')
plt.ylabel('Counts')
plt.savefig('ratio-unsample.pdf')
plt.show()

# boxplot of ratio
df.loc[df['totalvote'].idxmax()]
plt.boxplot(df['totalvote'])
#plt.ylim([0,2000])

#### sample data to balance helpful and unhelpful
#df_good = df[df['ratio'] > 0.5]  #263540
#df_bad = df[df['ratio'] <= 0.5]  #114701

df_good = df[df['ratio'] >= 0.7]  #221077
df_bad = df[df['ratio'] <= 0.3]  #68516

df_good = df_good.sample(n=50000).reset_index(drop=True)
df_bad = df_bad.sample(n=50000).reset_index(drop=True)

df_s = pd.concat([df_good, df_bad]) # total 100000
df_s.to_csv('ratio-sample-10-0.7-0.3.csv', index=False)  

#histogram of ratio distribution
plt.hist(df_s['ratio'], color='pink', ec='black')
plt.title('Helpful Ratio for Sampled Data')
plt.xlabel('Helpful Ratio')
plt.ylabel('Counts')
plt.savefig('ratio-sample-0.7-0.3.pdf')
plt.show()

#################### baseline model ####################
df= pd.read_csv('ratio-sample-10-0.7-0.3.csv')  
# normalize review text
# removed capitalization, punctuation, and stopwords
col_names = ["reviewerID", "asin", "reviewText", "overall", "upvote", 
             "totalvote", "ratio"]

stop_words = stopwords.words('english')
word_pattern = re.compile("[A-Za-z]+")
stemmer = SnowballStemmer("english")
n_entries = len(df)
df_norm = pd.DataFrame(columns=col_names, index=range(n_entries))

for idx in range(n_entries):
    print(idx)
    row = df.iloc[idx]
    text = normalize_review_text(row['reviewText'])
    df_norm.iloc[idx] = [
            row['reviewerID'],
            row['asin'],
            text,
            row['overall'],
            row['upvote'],
            row['totalvote'],
            row['ratio']]

### train test split
# 80% train 20% test
df_norm['good'] = (df_norm.ratio >= 0.7).astype('int')
df_norm.to_csv('100000-norm.csv', index=False)  

df_norm = pd.read_csv('100000-norm.csv')
good_norm = df_norm[df_norm['good'] == 1]
bad_norm = df_norm[df_norm['good'] == 0]

train_g, test_g = train_test_split(good_norm, train_size=0.8,random_state=2)
train_b, test_b = train_test_split(bad_norm, train_size=0.8,random_state=2)

dftrain = pd.concat([train_g, train_b])
dftrain = dftrain.reset_index(drop=False)
dftest = pd.concat([test_g, test_b])
dftest = dftest.reset_index(drop=False)

dftrain.to_csv('train.csv', index=False)
dftest.to_csv('test.csv', index=False)

##### tf-idf
tfidf = TfidfVectorizer(decode_error='ignore', stop_words='english', min_df=0.001, 
                        ngram_range=(1,3), max_features=300, norm='l2')
tfidf_train = tfidf.fit_transform(dftrain['reviewText'])

features = tfidf.get_feature_names()
idf = tfidf.idf_
tfidf_dict = dict(zip(features, idf))
d = pd.DataFrame(list(tfidf_dict.items()), columns=['feature', 'idf'])
d.to_csv('tfidf.csv')

tfidf_train.shape
tfidf_test = tfidf.transform(dftest['reviewText'])
tfidf_test.shape

#### logistic regression
train_target = np.array(dftrain['good'])
test_target = np.array(dftest['good'])

logit = LogisticRegression(penalty='l2')
logit.fit(tfidf_train, train_target)
logit_pred = logit.predict(tfidf_test)
logit_acc = np.mean(logit_pred == test_target)  #0.815

confusion_matrix(test_target, logit_pred)
#array([[8200, 1800],
#       [1899, 8101]], dtype=int64)

print(classification_report(test_target, logit_pred))
#             precision    recall  f1-score   support

#          0       0.81      0.82      0.82     10000
#          1       0.82      0.81      0.81     10000

#avg / total       0.82      0.82      0.82     20000

##### roc curve
logit_preds_probs = logit.predict_proba(tfidf_test)  
fpr, tpr, _ = roc_curve(test_target, logit_preds_probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Reviews for Base Model')
plt.legend(loc="lower right")
plt.savefig('ROC-basemodel.pdf')
plt.show()

#### add overall feature
# add overall to tfidf_train
train_r = dftrain['overall'].values[...,None]
train_r.shape
tfidf_train_array = tfidf_train.toarray()
tfidf_train_array.shape
tfidf_train_r = np.hstack((tfidf_train_array, train_r))
tfidf_train_r.shape

test_r = dftest['overall'].values[...,None]
test_r.shape
tfidf_test_array = tfidf_test.toarray()
tfidf_test_array.shape
tfidf_test_r = np.hstack((tfidf_test_array, test_r))
tfidf_test_r.shape

logit = LogisticRegression(penalty='l2')
logit.fit(tfidf_train_r, train_target)
logit_pred = logit.predict(tfidf_test_r)
logit_acc = np.mean(logit_pred == test_target)  #0.8813

confusion_matrix(test_target, logit_pred)
#array([[8714, 1286],
#       [1087, 8913]], dtype=int64)
print(classification_report(test_target, logit_pred))
#             precision    recall  f1-score   support

#          0       0.89      0.87      0.88     10000
#          1       0.87      0.89      0.88     10000
#
#avg / total       0.88      0.88      0.88     20000
#roc
logit_preds_probs = logit.predict_proba(tfidf_test_r)  
fpr, tpr, _ = roc_curve(test_target, logit_preds_probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Reviews and Ratings for Base Model')
plt.legend(loc="lower right")
plt.savefig('ROC-basemodel-rating.pdf')
plt.show()


############## Neural Network Model ##############
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')

dftrain_g = dftrain[dftrain.good == 1]
dftrain_b = dftrain[dftrain.good == 0]
dftest_g = dftest[dftest.good == 1]
dftest_b = dftest[dftest.good == 0]

dftrain_g['reviewText'].to_csv('norm-train-g.txt', sep='\t', index=False)
dftrain_b['reviewText'].to_csv('norm-train-b.txt', sep='\t', index=False)
dftest_g['reviewText'].to_csv('norm-test-g.txt', sep='\t', index=False)
dftest_b['reviewText'].to_csv('norm-test-b.txt', sep='\t', index=False)

### not norm
df = pd.read_csv('ratio-sample-10-0.7-0.3.csv')
df_good = df[df.ratio >= 0.7]
df_bad = df[df.ratio <= 0.3]

dftrain_g, dftest_g = train_test_split(df_good, train_size=0.8,random_state=2)
dftrain_b, dftest_b = train_test_split(df_bad, train_size=0.8,random_state=2)

dftrain_g['reviewText'].to_csv('train-g.txt', sep='\t', index=False)
dftrain_b['reviewText'].to_csv('train-b.txt', sep='\t', index=False)
dftest_g['reviewText'].to_csv('test-g.txt', sep='\t', index=False)
dftest_b['reviewText'].to_csv('test-b.txt', sep='\t', index=False)



#### doc2vec
sources = {'norm-test-b.txt':'TEST_Unhelpful', 'norm-test-g.txt':'TEST_Helpful', 
           'norm-train-b.txt':'TRAIN_Unhelpful', 'norm-train-g.txt':'TRAIN_Helpful'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=10, size=300, negative=5, workers=7)

model.build_vocab(sentences.to_array())

i=1
for epoch in range(10):
    print(i)
    model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=model.iter)
    i += 1

model.save('./norm-Review-300.d2v')
model = Doc2Vec.load('./norm-Review-300.d2v')


len(model.docvecs) #100000
len(model.wv.vocab)  #153362
# model.wv.syn0
# model.wv.index2word
train_arrays = np.zeros((80000, 300))
train_labels = np.zeros(80000)

for i in range(40000):
    prefix_train_h = 'TRAIN_Helpful_' + str(i)
    prefix_train_uh = 'TRAIN_Unhelpful_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_h]
    train_arrays[40000 + i] = model.docvecs[prefix_train_uh]
    train_labels[i] = 1
    train_labels[40000 + i] = 0
    

test_arrays = np.zeros((20000, 300))
test_labels = np.zeros(20000)

for i in range(10000):
    prefix_test_h = 'TEST_Helpful_' + str(i)
    prefix_test_uh = 'TEST_Unhelpful_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_h]
    test_arrays[10000 + i] = model.docvecs[prefix_test_uh]
    test_labels[i] = 1
    test_labels[10000 + i] = 0


### RNN
# a basic 3-layer architecture.
nnmodel = Sequential()
nnmodel.add(Dense(128, activation='relu', input_dim=300))
nnmodel.add(Dropout(0.2))
nnmodel.add(Dense(64, activation='relu'))
nnmodel.add(Dropout(0.2))
nnmodel.add(Dense(1, activation='sigmoid')) # output
nnmodel.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
nnmodel.summary()
nnmodel.fit(train_arrays, train_labels, epochs=5)

score, acc = nnmodel.evaluate(test_arrays, test_labels)
print('Test accuracy:', acc)  #0.84015

nn_pred = nnmodel.predict_proba(test_arrays)

#### roc
nn_pred = nnmodel.predict_proba(test_arrays)
fpr, tpr, _ = roc_curve(test_labels, nn_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='orange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Reviews for NN model')
plt.legend(loc="lower right")
plt.savefig('ROC-nn.pdf')
plt.show()


###### add overall feature
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')

# add overall to tfidf_train
train_r = dftrain['overall'].values[...,None]
train_r.shape
nn_train = np.hstack((train_arrays, train_r))
nn_train.shape

test_r = dftest['overall'].values[...,None]
test_r.shape
nn_test = np.hstack((test_arrays, test_r))
nn_test.shape

####
### RNN
# a basic 3-layer architecture.
nnmodel2 = Sequential()
nnmodel2.add(Dense(128, activation='relu', input_dim=301))
nnmodel2.add(Dropout(0.2))
nnmodel2.add(Dense(64, activation='relu'))
nnmodel2.add(Dropout(0.2))
nnmodel2.add(Dense(1, activation='sigmoid')) # output
nnmodel2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
nnmodel2.summary()
nnmodel2.fit(nn_train, train_labels, epochs=3)

score, acc = nnmodel2.evaluate(nn_test, test_labels)
#print('Test score:', score) 0.28688
print('Test accuracy:', acc)  #0.88225


#### roc
nn_pred2 = nnmodel2.predict_proba(nn_test)
fpr, tpr, _ = roc_curve(test_labels, nn_pred2)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='orange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Reviews and Ratings for NN model')
plt.legend(loc="lower right")
plt.savefig('ROC-nn-rating.pdf')
plt.show()
