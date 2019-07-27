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

input_str = """Poor customer service at check in/out counter. Totally
mess. Not welcoming or appreciate the guest.
The room not clean. The worst is the toilet. So
dirty..seriously... The size of the toilet room is small too.
While the food.. the breakfast range is so little option.. In
summary.. we are quite shock with the price that we paid
and the service that we get. Totally not worth it. It might you
last last last option to stay here"""

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