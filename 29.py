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

input_str = """I booked this place for a quick long weekend trip with my family. The only review of this place
made it seemed like this place is not bad but I think they can improve a lot on their
cleanliness. I only had problems with how dusty some furnitures were. We had a car, so
traveling around was not a problem.  The good thing about this place is the staffs are very
helpful and airconditions work well and are very cooling. The place is very cheap too. 
 
If you are not a clean freak like me, you'll find this place acceptable."""

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
