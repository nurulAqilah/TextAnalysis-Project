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


input_str ="""We enjoyed a wonderful 5 night stay in this stylish, boutique hotel.
The location is good & is around a 10 to 15 minute walk to the centre or you can easily get a 'Grab' which are plentiful & cheap. 
The hotel is like a New York warehouse conversion - very stylish with lots of attention to detail. 
It has a very relaxed, chilled vibe to it. We stayed in a deluxe room & it was spacious with a comfortable bed & was well kitted out. 
Housekeeping was excellent - the room was spotlessly clean & was cleaned each day by the time we returned from breakfast. Breakfast was super. 
I prefer a light breakfast so was delighted with the range of non cooked options (muesli, bran flakes, chia seeds etc) & lovely touches like fresh rosemary & lemon in the chilled water. (Cooked options are available too).
There is a lovely cafe & we had lunch & dinner there & the food was very good. 
There is also a great little gym on the rooftop which is well equipped. 
The staff were friendly & attentive across the hotel, right from arrival & being greeted by the smiling security man. 
They made us feel like they really cared that our stay was a memorable one with them.
I have stayed at stacks of hotels throughout SE Asia & Rosa definitely ranks as one of my very favourites. 
Thank you for making our stay such a great one"""

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




