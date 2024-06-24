from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer 
import pickle 
import os
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 200
def pre_process(tweet):
    tweet = str(tweet)
    stopword = stopwords.words('english')
    tweet = [word for word in tweet.split() if word.lower() not in stopword]
    tweet = ' '.join(tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    stemmer = PorterStemmer()
    tweet = ' '.join([stemmer.stem(word) for word in tweet.split()])
    with open('./token/token_new.pickle','rb') as handle:
        tokenizer = pickle.load(handle)
    tweet = tokenizer.texts_to_sequences([tweet])
    tweet = pad_sequences(tweet, maxlen=VOCAB_SIZE)
    return tweet


# Load model
# model = load_model('./model/lstmmodel.h5')
# tweets = "I like politics"
# tweet = pre_process(tweets)
# pred = model.predict(tweet)
# print(pred)