import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
nltk.download("punkt")
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer= PorterStemmer()

import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint


input_str="""Stayed for 1 night, I must say that it’s really rare to find a hotel like Rosa Hotel in the heart of Melaka. The place is really unique and everything is designed with a lot of effort and practicality in thoughts for their guest. Their rooms are well equipped with nice armchair with throw nicely design pillow and super comfortable bed. The staff is friendly and helpful. Will definitely be staying here in the near future"""

CORPUS = ["""Stayed for 1 night, I must say that it’s really rare to find a hotel like Rosa Hotel in the heart of Melaka. The place is really unique and everything is designed with a lot of effort and practicality in thoughts for their guest. Their rooms are well equipped with nice armchair with throw nicely design pillow and super comfortable bed. The staff is friendly and helpful. Will definitely be staying here in the near future"""]

input_str = input_str.lower()
input_str = input_str.translate(str.maketrans('', '', string.punctuation))
result = re.sub("\d","", input_str)
print(result)

from nltk.tokenize import word_tokenize
tokens = word_tokenize(input_str)
print(tokens)

from nltk import FreqDist
frequency_token = nltk.FreqDist(tokens)
print(frequency_token.most_common(10))

input_str=word_tokenize(input_str)
for word in input_str:
    print(stemmer.stem(word))

from sklearn.feature_extraction.text import CountVectorizer
def bow_extractor(corpus, ngram_range=(1,1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

bow_vectorizer, bow_features = bow_extractor(CORPUS)
features = bow_features.todense()
print(features)

feature_names = bow_vectorizer.get_feature_names()
print(feature_names)

import pandas as pd
def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                  columns=feature_names)
    print(df)

from sklearn.feature_extraction.text import TfidfTransformer
def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                               smooth_idf=True,
                               use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix

import numpy as np

feature_names = bow_vectorizer.get_feature_names()

tfidf_trans, tdidf_features = tfidf_transformer(bow_features)
features = np.round(tdidf_features.todense(), 2)
display_features(features, feature_names)

import scipy.sparse as sp
from numpy.linalg import norm
feature_names = bow_vectorizer.get_feature_names()

tf = bow_features.todense()
tf = np.array(tf, dtype='float64')

display_features(tf, feature_names)


def word_tokenizer(text):
    # tokenizes and stems the text
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def cluster_sentences(sentences, nb_of_clusters=5):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                       stop_words=stopwords.words('english'),
                                       max_df=0.9,
                                       min_df=0.1,
                                       lowercase=True)

