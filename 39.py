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

input_str = """If you want a hotel that is faraway from the action of
Melaka, dark and dingy, has stained carpeting and sheets
then this is the place for you. After looking at the hotel
website and other people's reviews I had good hopes for
this place. When I arrived I was really surprised at the
dismal appearance of the hotel. If you come to Melaka stay
at one of the larger or botique hotels closer to the historical
attractions. It is worth it to stay at a hotel that is cleaner and
has better management then to be cheap and come here. I
stayed long enough to check out my room, the bathroom
and get on the computer to find another hotel """

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