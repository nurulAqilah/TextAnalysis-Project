import string
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

nltk.download("stopwords")
nltk.download("punkt")

input_str = """Location is good. But cannot be in there over 30min.1st floor was wet and very dirty. Glass
door prevent fresh air. toilet and design was poor. You should choice another place. You can
find more good place to stay with same price"""

input_str = input_str.lower()
input_str = input_str.translate(str.maketrans('', '', string.punctuation))
result = re.sub("\d", "", input_str)

from nltk.tokenize import word_tokenize

tokens = word_tokenize(input_str)
from nltk import FreqDist

frequency_token = nltk.FreqDist(tokens)
print(frequency_token.most_common(10))

input_str = word_tokenize(input_str)

print(result)
print(input_str)
for word in input_str:
    print(stemmer.stem(word))
