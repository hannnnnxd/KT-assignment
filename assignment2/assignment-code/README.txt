This tarball contains three files (plus this README), as follows:

1.test-labels.txt:
This file contains the predicted sentiment labels, one prediction per line, in the following format:
tweet-id \t label \n
The labels are one of "positive", "negative", or "neutral".

2.KTPro_Assig2.py:
This program is used to train the data and predict the result.

There are five variables:
(1).train_twitter_path: the value is the absolute path of train-tweets.txt
(2).train_twitter_sem_path: the value is the absolute path of train-labels.txt
(3).dev_twitter_path: the value is the absolute path of dev-tweets.txt
(4).dev_twitter_sem_path: the value is the absolute path of dev-labels.txt
(5),test_twitter_path: the value is the absolute path of test-tweets.txt
Note that u should reset the value of those variables.

Program part In[1],In[2],In[3],In[4],In[5],In[6] are used to simply preprocess the input data, and get five variables:
(1).train_feature_sents: this variable is a list, every item in the list is a single tweets without url
(2).train_sem_sents: this variable is a list, every item in the list is a corresponding label with train_feature_sents
(3).dev_feature_sents: this variable is a list, every item in the list is a single tweets without url
(4).dev_sem_sents: this variable is a list, every item in the list is a corresponding label with dev_feature_sents
(5).test_feature_sents: this variable is a list, every item in the list is a single tweets without url

Program part In[7],In[8] are used to get feature:
Firstly, using TfidfVectorizer() to set the rules of selection features and vectorization.
Using fit_transform() to transform the tweets and get the vectorized feature.
Using transform() to transform the tweets, which need to be predicted, into the same format with the output of fit_transform(). 

Program part In[9],In[10],In[11] are used to train Naive Bayes models and Decision Tree Model using engineered feature and predict the result:
Using fit(x,y) to train the model. X is selected feature; Y is label.
Using predict() to predict the label.
Using calculate_result() to print the performance of the current model.

Program part In[12] are used to write the predict result in to file:
I use this program to test the performance of different models. Thus, if u want to write the prediction of test-tweets.txt. U need to change the attribute to ‘test_matrix’ in predict() function, then annotate the calculate_result() function.

** Note that this program has used many package, if u want to operate this program, u should make sure that u system contains the package mentioned in In[1].
