# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Here, "header=0" indicates that the first line of the file contains column names,
# "delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 
# tells Python to ignore doubled quotes, otherwise you may encounter errors 
# trying to read the file.
train = pd.read_csv("labeledTrainData.tsv", sep='\t', quoting=3)

train.shape

train.columns.values

print(train['review'][0])

# Importing BeautifulSoup into our workspace
from bs4 import BeautifulSoup

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])

print(train['review'][0])
print(example1.get_text())

# use beautiful soup instead of regex to remove mark up # a better practice
import re

lettersOnly =   re.sub('[^a-zA-Z]', ' ', example1.get_text())

print(lettersOnly)

lowerCase = lettersOnly.lower()
words = lowerCase.split()

# removing stopword using nltk library
import nltk
nltk.download()

from nltk.corpus import stopwords
stopword = stopwords.words("english")

# remove stop words from words
words = [word for word in words if not word in stopword]

def review_to_words(review):
    
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()      
    
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


clean_review = review_to_words( train["review"][0] )
print(clean_review)

# get the number of review size
numReviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_review = []

# loop over each review 

for i in range(0 , numReviews):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_review.append(review_to_words(train["review"][i]))





















































