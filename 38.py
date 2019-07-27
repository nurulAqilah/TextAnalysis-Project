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


input_str = input_str.lower()
input_str = """The worst hotel I ever stayed in
- checking in, there was this lady at the front desk just
screaming away at her staff, she's either the boss or the
manager. No system and they don't understand their own
reservations
- rooms, dirty, shower head too low, water dripping from
ceiling , bed rolling all over , mini fridge looked like a bat
cave and not working
- reserved a king bed but got 2 single bed joined together
- they were kind enough to out up a parking ticket early
morning to avoid summon. However they charge $1 when it
actually cost $0.60 without even asking if u wanted that
service
- wifi is horrible and can't even load to face book or Google
search. The staff don't even want to reset the wifi and just told me that too many users. Why advertise free wifi if ur
wifi can't service the entire hotel?
- everytime j leave the room or get back to the hotel, most
of the time I will see the lady at the front desk screaming
away with no regards to guest walking in or out . Again she
could be the manager or the owner, but whichever it is,
she's better off managing a wet market or a flea market"""
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