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


input_str ="""I am satisfied with the service and the location is at the center of melaka.
So you can go anywhere easily. On top of that, the interior design is so beautiful.
You can make a photoshoot here. Hehe. Will definitely come again """

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
