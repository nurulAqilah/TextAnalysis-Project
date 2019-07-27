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

input_str = """I was supposed to stay here for a week but ended up
staying for just 1 day because I had enough. The room is
dirty, the shower is bad, the staff is unprofessional, and
what made me leave is that I found a cricket and a
cockroach in my room. Why can't they just clean the room
for the guests? Extremely uncaring. I will never come here
again"""

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

