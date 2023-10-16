import nltk
from nltk.corpus import wordnet
from textblob import Word
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

nltk.download('popular')

f = open(r"E:\codified\libraries\Docker has gained immense popularity in this fast growing IT world.txt", 'r', errors='ignore')
raw = f.read()
f.close()
raw = raw.lower()
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what’s up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "Oops, I can’t understand you!"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

def spell_check(word):
    spell = Word(word)
    corrected_word = spell.correct()
    return corrected_word

flag = True
print("John: Hi!! My name is John. I’ll answer all your queries about Chatbots. If you want to exit, just type 'bye'.")

while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        user_response = ' '.join([spell_check(word) for word in user_response.split()])
        if greeting(user_response) is not None:
            print("John: " + greeting(user_response))
        else:
            print("John: ", end="")
            print(response(user_response))
            sent_tokens.remove(user_response)
    else:
        flag = False
        print("John: Goodbye! Have a nice day.")
