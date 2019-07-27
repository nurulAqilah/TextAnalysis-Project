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

input_str = """Stayed in an apartment type room. Bathroom was filthy.
Toilet had brown stains - obviously never cleaned. Sink had
a crust of dirt and grime under rim. Only two small size,
thread bare towels provided which were far from white.
Room never cleaned properly. Fridge had mould inside.
Stayed there for six days. Didn't eat breakfast even though
paid for as I dread to think what state the kitchen might be if
the rooms are so dirty. Didn't use the swimming pool as
also dirty. Had to use general loo by swimming pool once
as reception toilet was out of use. It was absolutely filthy"""

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