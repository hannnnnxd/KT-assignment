
# coding: utf-8

# In[1]:

import numpy as np
import scipy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re


# In[2]:

train_twitter_path = "/Users/hannnnn/Desktop/knowledge_technologies/Assignment2/2017S1-KTproj2-data/train-tweets.txt"
train_twitter_sem_path = "/Users/hannnnn/Desktop/knowledge_technologies/Assignment2/2017S1-KTproj2-data/train-labels.txt"
dev_twitter_path = "/Users/hannnnn/Desktop/knowledge_technologies/Assignment2/2017S1-KTproj2-data/dev-tweets.txt"
dev_twitter_sem_path = "/Users/hannnnn/Desktop/knowledge_technologies/Assignment2/2017S1-KTproj2-data/dev-labels.txt"
test_twitter_path = "/Users/hannnnn/Desktop/knowledge_technologies/Assignment2/2017S1-KTproj2-data/test-tweets.txt"


# In[3]:

train_twitter = open(train_twitter_path,encoding="utf-8")
train_twitter_sem = open(train_twitter_sem_path,encoding="utf-8")
dev_twitter = open(dev_twitter_path,encoding="utf-8")
dev_twitter_sem = open(dev_twitter_sem_path,encoding="utf-8")
test_twitter = open(test_twitter_path,encoding="utf-8")
sem_result = open('/Users/hannnnn/Desktop/test-labels.txt', 'w')
try:
    all_train_twitter = train_twitter.readlines()
    all_train_twitter_sem = train_twitter_sem.readlines()
    all_dev_twitter = dev_twitter.readlines()
    all_dev_twitter_sem = dev_twitter_sem.readlines()
    all_test_twitter = test_twitter.readlines()
finally: 
    train_twitter.close()
    train_twitter_sem.close()
    dev_twitter.close()
    dev_twitter_sem.close()
    test_twitter.close
# print(len(all_train_twitter))
# print(len(all_train_twitter_sem))
# print(len(all_dev_twitter))
# print(len(all_dev_twitter_sem))
# print(len(all_test_twitter))



# In[4]:

pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
start_num = 0
_all_train_twitter = []
_all_dev_twitter = []
_all_test_twitter = []
for sentence in all_train_twitter:
    _all_train_twitter.append(pattern.sub("",sentence))
for sentence in all_dev_twitter:
    _all_dev_twitter.append(pattern.sub("",sentence))
for sentence in all_test_twitter:
    _all_test_twitter.append(pattern.sub("",sentence))

# print(len(_all_train_twitter))
# print(len(all_train_twitter_sem))
# print(len(_all_dev_twitter))
# print(len(all_dev_twitter_sem))



# In[5]:

train_words = []
train_sem = []
dev_words = []
dev_sem = []
test_words = []
start_num = 0
for each_word in _all_train_twitter:
    if start_num < len(_all_train_twitter):
        temp = _all_train_twitter[start_num].split("\t")
        temp[1] = temp[1][:-1]
        train_words.append(temp[:2])
    start_num = start_num + 1
# print(len(train_words))
# print(train_words)

start_num = 0
for each_word in all_train_twitter_sem:
    if start_num < len(all_train_twitter_sem):
        temp = all_train_twitter_sem[start_num].split("\t")
        temp[1] = temp[1][:-1]
        train_sem.append(temp[:2])
    start_num = start_num + 1
# print(len(train_sem))
# print(train_sem)

start_num = 0
for each_word in _all_dev_twitter:
    if start_num < len(_all_dev_twitter):
        temp = _all_dev_twitter[start_num].split("\t")
        temp[1] = temp[1][:-1]
        dev_words.append(temp[:2])
    start_num = start_num + 1
# print(len(dev_words))
# print(dev_words)

start_num = 0
for each_word in all_dev_twitter_sem:
    if start_num < len(all_dev_twitter_sem):
        temp = all_dev_twitter_sem[start_num].split("\t")
        temp[1] = temp[1][:-1]
        dev_sem.append(temp[:2])
    start_num = start_num + 1
# print(len(dev_sem))
# print(dev_sem)

start_num = 0
for each_word in _all_test_twitter:
    if start_num < len(_all_test_twitter):
        temp = _all_test_twitter[start_num].split("\t")
        temp[1] = temp[1][:-1]
        test_words.append(temp[:2])
    start_num = start_num + 1
# print(len(test_words))
# print(test_words)


# In[6]:

train_feature_sents = []
train_sem_sents = []
dev_feature_sents = []
dev_sem_sents = []
test_feature_sents = []
for sents in train_words:
    train_feature_sents.append(sents[1])

# print(train_feature_sents)
# print(len(train_feature_sents))
    
for sents in train_sem:
    if sents[1] == "negative":
        train_sem_sents.append("-1")
    if sents[1] == "positive":
        train_sem_sents.append("1")
    if sents[1] == "neutral":
        train_sem_sents.append("0")

# print(train_sem_sents)
# print(len(train_sem_sents))

for sents in dev_words:
    dev_feature_sents.append(sents[1])
    
# print(dev_feature_sents)
# print(len(dev_feature_sents))

for sents in dev_sem:
    if sents[1] == "negative":
        dev_sem_sents.append("-1")
    if sents[1] == "positive":
        dev_sem_sents.append("1")
    if sents[1] == "neutral":
        dev_sem_sents.append("0")
        
# print(dev_sem_sents)
# print(len(dev_sem_sents))

for sents in test_words:
    test_feature_sents.append(sents[1])

# print(test_feature_sents)
# print(len(test_feature_sents))


# In[7]:

def tokenize_words(text):
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import word_tokenize

    from nltk.stem.lancaster import LancasterStemmer  
    lancaster_stemmer = LancasterStemmer() 

    sent_tokenize_list = sent_tokenize(text)
    word_tokenize_list = []
    for sents in sent_tokenize_list:
        word_tokenize_list = word_tokenize_list + word_tokenize(sents)

        
    pattern1 = re.compile(r"[\W]+")
    pattern2 = re.compile(r"\d")
    pattern3 = re.compile(r"^[a-z]+$")
    word_list = []

    for word in word_tokenize_list:
        word1 = pattern1.sub("",word)
        word2 = pattern2.sub("",word)
        word3 = pattern3.match(word)
        if word1 != "":
            if word2 != "":
                if word3 != None:
                    if len(word) > 3:
                        word_list.append(lancaster_stemmer.stem(word))
    return word_list


# In[8]:

from sklearn.feature_extraction.text import TfidfVectorizer

vec=TfidfVectorizer(max_df=0.99,min_df=10,stop_words='english',max_features=5000,analyzer = 'word',tokenizer = tokenize_words)
train_matrix=vec.fit_transform(train_feature_sents).toarray()
dev_matrix=vec.transform(dev_feature_sents)
test_matrix=vec.transform(test_feature_sents)


# In[9]:

def calculate_result(actual,pred):
    print(metrics.classification_report(actual, pred))
    print(metrics.confusion_matrix(actual,pred,labels=["-1","0","1"]))


# In[10]:

#Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
clf = MultinomialNB(alpha = 0.01)
clf.fit(train_matrix,train_sem_sents)
pred = clf.predict(dev_matrix)
calculate_result(dev_sem_sents,pred)


# In[11]:

#Decision Tree Classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(train_matrix,train_sem_sents)
pred = clf.predict(dev_matrix)
calculate_result(dev_sem_sents,pred)


# In[12]:

#writing the prediction result to file
#pred_sem_list = []
#i = 0
#for word in pred:
#    if word == '1':
#        pred_sem_list.append("positive")
#        i = i+1
#    if word == '0':
#        pred_sem_list.append("neutral")
#        i = i+1
#    if word == '-1':
#        pred_sem_list.append("negative")
#        i = i+1
#pred_list = []
#num = 0
#temp = []
#for word in all_test_twitter:
#    if num < len(all_test_twitter):
#        temp = word.split("\t")
#        temp_word = temp[0] +"\t"+ pred_sem_list[num]
#        pred_list.append(temp_word)
#        num = num +1
#
#for word in pred_list:
#    sem_result.write(word+"\n")



# In[ ]:



