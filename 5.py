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


input_str="""This newly built hotel is in the style of an abandoned warehouse. Very funky and stylish. Modern fittings and very comfortable. Breakfast was in the atrium and reasonable. The Gym on the roof is ok, but needs more heavy weights and toilets! 
 
They plan an extension which will add a much needed pool.  
 
The bar is good and food was fine.  
 
My only complaint is the AC was never cold enough in the high humidity"""

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









