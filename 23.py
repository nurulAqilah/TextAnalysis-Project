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

input_str = """Really didn't enjoy our stay here. The rooms were very smelly (cigarette smoke), and very
basic. We were not given towels, or a bed sheet.  
 	
The bathrooms are equally as terrible, with no toilet roll, and they weren't cleaned throughout
our stay in Melaka. 
 
The staff weren't particularly friendly or helpful. 
 
In addition, I don't think the location is that great - it took maybe 10 minutes to walk to the
centre of town.  
 
If I were to return to Melaka, I would stay somewhere else closer to the town. There are some
lovely guesthouses in Jonker Street for the same value."""

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
