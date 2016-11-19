#############################################
####### Data Reader / Feature Creator #######
#############################################
import string
from nltk.stem import PorterStemmer
import nltk
import re
from sklearn import feature_extraction as fe
from collections import Counter
import numpy as np
import pandas as pd

#converts text to lower case, strips punctuation, tokenizes string, removes stop words,
#returns only words longer than two characters
def tokenize(text):
    text = text.lower()
    pattern = '[' + string.punctuation +'0123456789\\n\\r\\t]+'
    text = re.sub(pattern, ' ', text)
    token = nltk.word_tokenize(text)
    token = [t for t in token if (len(t)>2 and t not in fe.stop_words.ENGLISH_STOP_WORDS)]
    return token

#porter stems words
def stemwords(words):
    stemmer = PorterStemmer()
    stems = [stemmer.stem(word) for word in words]
    return stems

#given a string of all lines for character, returns the top 50 words after
#tokenizing/stemming
def top_character_words(corpus):
    bag = stemwords(tokenize(corpus))
    count = Counter(bag)
    top_words = [word[0] for word in count.most_common(50)]
    return top_words

#given dataset and character id, takes all lines by that character
#and joins in one string (corpus)
def corpus(data, id):
    lines = [line for line in data[data.character_id==id]['normalized_text']]
    corp = " ".join(lines)
    return corp

#collects top 50 words for every unique character in the dataset
#union them w/ set, returns as a list
def top_words(data):
    unique_characters = set(data['character_id'])
    all_top_words = []
    for character in unique_characters:
        corp = corpus(data, character)
        top = top_character_words(corp)
        all_top_words += top
    unique_top_words = set(all_top_words)
    return list(unique_top_words)

#given line text and list of words to count, will stem/tokenize
#line and return list of word frequency in line text for each word in "words"
def word_rate(text, words):
    bag = stemwords(tokenize(text))
    N = 1.0*len(bag)
    if N == 0:
        word_rates = [0 for word in words]
        return word_rates
    count = Counter(bag)
    word_rates = [count[word]/N for word in words]
    return word_rates

#given data and "top words" list, returns features
#each row is [Location dummies, Top Word Rates]
def extract_features(data, words):
    features = np.array([0 for i in range(len(words))])
    for line in data['normalized_text']:
        line_feature = word_rate(line, words)
        features = np.vstack((features,line_feature))
    dummie = pd.get_dummies(data['location_id']).as_matrix()
    return np.hstack((dummie, features[1:]))





#TODO
# remove the ones where speaking_line (6th column) == False
# replace character_id with the actual character names from the character csv?
# same with location
# use character as Y, use location, normalized text, and word count to create features (also maybe episode id?)
# manually create corpus of person specific terms or phrases?

# TODO Features to add
# TFIDF?
# dummy variables of location
# stemmed words vs actual words?


