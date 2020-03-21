import numpy as np
import pandas as pd
import nltk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib

nltk.download('stopwords')

dataset = pd.read_csv('spam.csv', header=0, encoding = "ISO-8859-1")
dataset = dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

dataset.columns = ['labels', 'data']

dataset['b_labels'] = dataset['labels'].map({'ham': 0, 'spam': 1})
stop_words = set(nltk.corpus.stopwords.words('english'))
Y = dataset['b_labels'].values

def train_model(X_train, X_test, y_train, y_test):
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)

    print('Adaboost:')
    print('Train score = {}'.format(model.score(X_train, y_train)))
    print('Test score = {}'.format(model.score(X_test, y_test)))

    model = MultinomialNB()
    model.fit(X_train, y_train)

    print('MultinomialNB:')
    print('Train score = {}'.format(model.score(X_train, y_train)))
    print('Test score = {}'.format(model.score(X_test, y_test)))

    return model

print('Using Count')
vectorizer = CountVectorizer(stop_words=stop_words, strip_accents='ascii')
X = vectorizer.fit_transform(dataset['data'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=2)
train_model(X_train, X_test, y_train, y_test)

vectorizer = TfidfVectorizer(stop_words=stop_words, strip_accents='ascii')
X = vectorizer.fit_transform(dataset['data'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print('Using TF-IDF')
model = train_model(X_train, X_test, y_train, y_test)

dataset = pd.read_csv('spambase.csv', header=0)
dataset.columns = ['data', 'labels']
stop_words = set(nltk.corpus.stopwords.words('english')).union(set(nltk.corpus.stopwords.words('french'))).union(["{:02d}".format(i) for i in set(range(31))])

X = vectorizer.transform(dataset['data'])
Y = dataset['labels'].values

print('Personal spam base')
print('Test score = {}'.format(model.score(X, Y)))