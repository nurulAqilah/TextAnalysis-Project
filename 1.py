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


input_str ="""A boutique hotel that located not far from town, less than a hundred rooms but has a fantastic design design that makes me go back to stay each time.
The ambience from the lobby lounge till the reception, I love the scent and the design for both of the location.  
 
The hotel used to served great English breakfast but they have changed it to buffet but still ok """

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



