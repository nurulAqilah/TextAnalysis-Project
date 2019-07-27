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

input_str = """This hotel has no lift ...the parking attendant was very rude and we have to pay RM5 for
parking privately owned by one Indian lady. Hotel ground floor is a coffee shop. Hotel room
bed board was dusty with torn bed and dirty looking sheet .  Location is good . Walking distant
within 10 min to Jonkers Street"""

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
