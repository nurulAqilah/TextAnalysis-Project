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


input_str ="""This hotel is in a newly remodeled industrial building. The hotel is very nicely developed with interesting furnishings and decor.
An in-house cafe serves quality simple meals for lunch and dinner. Everything is first rate. 
There is a gym but no pool. The location is couple of miles from downtown but close to a couple of malls and major hotels. 
It is easy to get around with the Grab ride app. The breakfast is great and staff is very helpful"""

input_str=input_str.lower()
input_str = input_str.translate(str.maketrans('', '', string.punctuation))
result = re.sub("\d","", input_str)



from nltk.tokenize import word_tokenize
tokens = word_tokenize(input_str)
from nltk import FreqDist
frequency_token = nltk.FreqDist(tokens)
print(frequency_token.most_common(10))



input_str=word_tokenize(input_str)


print(result)
print(input_str)
for word in input_str :
 print(stemmer.stem(word))





