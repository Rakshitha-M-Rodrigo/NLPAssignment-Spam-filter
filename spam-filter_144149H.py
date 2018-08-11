# NLPAssignment-Spam-filter
#
# Author : 144149H - R.M.Rodrigo
# Bundle : NLP Assignment
# Level - B14, faculty of Information Technology, University of Moratuwa.
#

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

dataSet = pd.read_csv("SMSSpamCollection.tsv", sep='\t', names=['Label', 'Message'])
headers = dataSet.head()
print(headers)

#change lables into tags
dataSet.loc[dataSet["Label"] == 'ham', 'Label'] = 1
dataSet.loc[dataSet["Label"] == 'spam', 'Label'] = 0

newHeaders = dataSet.head
print(newHeaders)

messagesDataSet = dataSet['Message']
labelDataSet = dataSet['Label']

print('size of the dataset = ',len(messagesDataSet))
print(labelDataSet)
print('spam count = ', len(dataSet[dataSet.Label == 'spam']))
print('spam count = ', len(dataSet[dataSet.Label == 'ham']))


TtestEqualResult = 0
PredictionDataLength = 0

def calculateNgramAccuracy(range_min, range_max):
    vectorClassifier = TfidfVectorizer(min_df=1,stop_words='english', ngram_range=(range_min,range_max))
    messages_trainData, messages_testData, label_trainData, label_testData = train_test_split(messagesDataSet,labelDataSet,test_size=0.2, random_state=4)
    x_traincv = vectorClassifier.fit_transform(messages_trainData)
    trainData = x_traincv.toarray()
    print(trainData)
    featureNames = vectorClassifier.get_feature_names()
    print(featureNames)

    values = trainData[0]
    print(values)
    length = len(trainData[0])
    print(length)

    data = vectorClassifier.inverse_transform(trainData[0])
    print(data)

    actualData = messages_trainData.iloc[0]
    print(actualData)

    multiNB = MultinomialNB()
    label_trainData = label_trainData.astype('int')
    multiNBData = multiNB.fit(x_traincv, label_trainData)
    print(multiNBData)
    x_testcv = vectorClassifier.transform(messages_testData)
    predict = multiNB.predict(x_testcv)
    print(predict)

    actualTestDataLabels = np.array(label_testData)
    print(actualTestDataLabels)

    testEqualResult = 0
    for i in range (len(label_testData)):
        if (actualTestDataLabels[i] == predict[i]):
            testEqualResult += 1

    TtestEqualResult = testEqualResult
    PredictionDataLength = len(predict)

    #n-gram
    print("range",range_min,"-",range_max,"Ngram Equal Data count = ",TtestEqualResult)
    print("range",range_min,"-",range_max,"Ngram tested data count = ",PredictionDataLength)
    print("range",range_min,"-",range_max,"Ngram accuracy = ", TtestEqualResult * 100.0 / PredictionDataLength, "% ~> ", TtestEqualResult * 100.0 // PredictionDataLength, "%")

#
# User operations here can be made
#change this value to change the N-gram "N" Value
NgramValue = 3
#print("Entered Ngram value = ", NgramValue)
#NgramValue = int(input("Entered Ngram value : "))
calculateNgramAccuracy(NgramValue,NgramValue)
