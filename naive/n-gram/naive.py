from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import six.moves.cPickle as pickle
# from imdbReview import extract_words



# This script is what created the dataset pickled.

dataset_path='/space/changhxu/LDA/project/aclImdb/'

import numpy
# import cPickle as pkl
import serialize as pyGet


from collections import OrderedDict
from nltk.corpus import stopwords

import glob
import os
import re
import sys
import string

x = 0
pos = 0
neg = 0
once = True

filecorpus=['text1.txt', 'text2.txt', 'text3.txt', 'text4.txt', 'text5.txt', 'text6.txt', 'text7.txt', 'text8.txt', 'text9.txt', 'text10.txt',
'text11.txt', 'text12.txt', 'text13.txt', 'text14.txt', 'text15.txt', 'text16.txt', 'text17.txt', 'text18.txt', 'text19.txt', 'text20.txt',
'text21.txt', 'text22.txt', 'text23.txt', 'text24.txt', 'text25.txt', 'text26.txt', 'text27.txt', 'text28.txt', 'text29.txt', 'text30.txt',
'text31.txt', 'text32.txt', 'text33.txt', 'text34.txt', 'text35.txt', 'text36.txt', 'text37.txt', 'text38.txt', 'text39.txt', 'text40.txt']

file_in = list(filecorpus)


def extract_words(sentences):
    result = []
#preprocessing
    stop = stopwords.words('english')
    trash_characters = '?.,!:;"$%^&*()#@+/0123456789<>=\\[]_~{}|`'
    trans = str.maketrans(trash_characters, ' '*len(trash_characters))

    for text in sentences:
        text = re.sub(r'[^\x00-\x7F]+',' ', text)

        text = text.replace('<br />', ' ')
        text = text.replace('--', ' ').replace('\'s', '')
        text = text.translate(trans)
        text = ' '.join([w for w in text.split() if w not in stop])

        words = []
        for word in text.split():
            word = word.lstrip('-\'\"').rstrip('-\'\"')
            if len(word)>2:
                words.append(word.lower())
        text = ' '.join(words)
        result.append(text.strip())
    return result


def grab_data(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = extract_words(sentences)

    return sentences

def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path

    train_x_pos = grab_data(path+'train/pos')
    train_x_neg = grab_data(path+'train/neg')
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(path+'test/pos')
    test_x_neg = grab_data(path+'test/neg')
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('train.pkl', 'wb')
    pyGet.pickle.dump((train_x, train_y), f, -1)
    f.close()
    f = open('test.pkl', 'wb')
    pyGet.pickle.dump((test_x, test_y), f, -1)
    f.close()


# if __name__ == '__main__':
#   main()


# Load All Reviews in train and test datasets
f = open('train.pkl', 'rb')
reviews = pickle.load(f)
f.close()

f = open('test.pkl', 'rb')
test = pickle.load(f)
f.close()


# Generate counts from text using a vectorizer.  
# There are other vectorizers available, and lots of options you can set.
# This performs our step of computing word counts.

vectorizer = CountVectorizer(ngram_range=(1, 3))
train_features = vectorizer.fit_transform([r for r in reviews[0]])
test_features = vectorizer.transform([r for r in test[0]])

# Fit a naive bayes model to the training data.
# This will train the model using the word counts we computer, 
#       and the existing classifications in the training set.
nb = MultinomialNB()
nb.fit(train_features, [int(r) for r in reviews[1]])

# Now we can use the model to predict classifications for our test features.
predictions = nb.predict(test_features)

# Compute the error.  
print(metrics.classification_report(test[1], predictions))
print("accuracy: {0}".format(metrics.accuracy_score(test[1], predictions)))

while True:
    for Item in filecorpus:
        sentences = []
        if(x < len(filecorpus)):
            file_in[x] = open("CORPUS/" + Item, 'r')
            sentence = file_in[x].read()
            print(str(x+1)+": "+sentence)
            sentences.append(sentence)
            input_features = vectorizer.transform(extract_words(sentences))
            #knowledgediscovery
            prediction = nb.predict(input_features)
            if prediction[0] == 1:
                print("---- \033[92mpositive\033[0m\n")
            else:
                print("---- \033[91mnegative\033[0m\n")
            x += 1
        else:
            break

    if(once):
        once = False
        sys.exit(0)