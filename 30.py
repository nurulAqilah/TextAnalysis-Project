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

input_str = """I am a regular Weekend travel to Malacca,most of the time I am staying near Jonker Street,
but the weekend of 14-15 Jun 2014, I had organise a group trip which I manage to find few
rooms which is near to shopping mall. the location is great but the service is very very poot
before I even arrive Malacca. I called 3 days before arrival to ask for change of the rooms
whihc i booked via agoda, unfortuanately was answered by a Malay girl which cant even speak
Singe English or understand my request, and she just put off my phone. I called again on
Friday evening which is one day beforfe departure, a staff answering my call with simple
english and replyed he understand my requested. up arrival I was surprise that I was told by
the owner of the hotel, they only had one booking which I had confirmed with his staff that I
had three booking. I even recorded down my phone conversation with his staff. after more
than 30min , finally they found my antoehr two booking but time was delayed. I was given a
tiny small room which I found spideer wed and lizard on the wall, I dont see any window at all,
and the internal locked is seriously damaged unlocked. I ask for change, finally he told me that
all single room is the same. without choice I force o upgade to double room with extra cost.
but the room seem very totally different from the picture in Agoda, the TV set no remote
control and the button are damaged. the shower heater not even working at all, the towels is
horrible, look like floor mat. I try to charge my phone, I found. my advise is to those first time
traveller who wanna stay near shopping mall, please select other nearby hotel, I believe is
much more better than this with the same price, I can even find a place cheaper than Trend
hotel with better service. Again upon check out next morning, we had problem again. the staff
mix up all our booking. No. I never never come back again even they offer FREE STAY!"""

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
