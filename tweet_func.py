# Import libraries
import numpy as np
import pandas as pd

# nltk library
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# --------------------------------------------------------------------------------

# Data Preparation - Tokenization and Cleaning

def preprocess(text):
    """Tokenizes raw text. Removes capitalised text, stopwords, 
    and non-alphabetical characters"""
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

def stemming(sentence):
    """Stems words in sentences using nltk package"""
    ps = PorterStemmer()
    doc = [ps.stem(w) for w in sentence]
    return doc

def lemmatization(sentence):
    """Lemmatises words in sentences using nltk package"""
    lem = WordNetLemmatizer()
    doc = [lem.lemmatize(w) for w in sentence]
    return doc
    
def filter_docs(trainsets, testsets, condition):
    """Removes text given the function condition. The text is removed
    if the condition is true"""
    counter=0
    for data in trainsets:
        number_of_rows = len(data)
        for row,text in enumerate(data):
            if condition(text):
                data.drop([row], axis=0, inplace=True)
                testsets[counter].drop([row], axis=0, inplace=True)
        counter+=1
        print("{} rows removed".format(number_of_rows-len(data)))

def has_vector_representation(word2vec_model, text):
    """check if at least one word in the sentence is in the
    word2vec dictionary"""
    return all(word not in word2vec_model for word in text)

# --------------------------------------------------------------------------------
# Evaluation

from sklearn.metrics import f1_score

# --------------------------------------------------------------------------------
# Creating vectors for each tweet

def average_vecs(sentence, Model, dimensions):
    """Average word vectors"""
    sumvec = np.zeros((dimensions), dtype='float32') # Initialise array for sentence
    numerator = np.zeros((dimensions), dtype='float32')
    nwords = 0 # Num of words in the sentence that are included in the model
    e = 1e-10 # To prevent division by zero
    for word in sentence:
        if word in Model:
            nwords = nwords + 1.
            numerator += Model[word]
    sumvec = numerator / (nwords+e)
    return sumvec

