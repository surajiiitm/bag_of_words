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
    if ((i+1)%1000 == 0 ):
        print("review of %d of %d" %(i+1, numReviews))
    clean_train_review.append(review_to_words(train["review"][i]))

## Now that we have our training reviews tidied up, how do we convert them to 
## some kind of numeric representation for machine learning? One common approach
## is called a Bag of Words. The Bag of Words model learns a vocabulary from 
## all of the documents, then models each document by counting the number of 
## times each word appears.
    
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word",
                             tokenizer= None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)


# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_review)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

print(train_data_features.shape)

# take a look at the words in vocabulary
vocab = vectorizer.get_feature_names()
print(vocab)

# count of each word in a vocabulary
# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)
    
# training the random forest
from sklearn.ensemble import RandomForestClassifier

# initialize the random forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# fit the forest to training set, using the bag of words as features and the
# the sentiment labels as the response variable

forest = forest.fit(train_data_features, train["sentiment"])

# predicting the test set result
test = pd.read_csv("testData.tsv", delimiter='\t', quoting=3)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
print(test.shape)

test.head()
# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

for i in range(0, num_reviews):
    if (i+1)%1000 == 0:
        print("review %d of %d" %(i+1, num_reviews))
    
    clean_test_reviews.append(review_to_words(test["review"][i]))

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

# Use pandas to write the comma-separated output file
output.to_csv("Bag_of_word_model.csv", index=False, quoting=3)





















































