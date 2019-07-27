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


input_str ="""Only one night at Rosa Malacca but love the interior! 
From the moment you enter the lobby to the check in area to the room and eating areas! 
Room was comfortable, toilet on the small side but that doesn't matter. Check in area on level one, which was different for me because most check in areas on ground level.
Friendly door man, staff at check in can be more friendly though. Breakfast has the usuals.
Staff can top up food more frequently. Had to ask them to top up the coffee and some of the hot mains trays were left bare. 
Location wise, a little out of main centre but no issues asking staff to call for Grab. 
They readily helped with this! Overall, a pleasant stay and would stay here again"""
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



