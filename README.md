

```python
# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Here, "header=0" indicates that the first line of the file contains column names,
# "delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 
# tells Python to ignore doubled quotes, otherwise you may encounter errors 
# trying to read the file.
train = pd.read_csv("labeledTrainData.tsv", sep='\t', quoting=3)
```


```python
train.shape
```




    (25000, 3)




```python
train.columns.values
```




    array(['id', 'sentiment', 'review'], dtype=object)




```python
print(train['review'][0])
```

    "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."



```python
# Importing BeautifulSoup into our workspace
from bs4 import BeautifulSoup

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0], "lxml")
```


```python
example1.get_text()
```




    '"With all this stuff going down at the moment with MJ i\'ve started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ\'s feeling towards the press and also the obvious message of drugs are bad m\'kay.Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci\'s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ\'s music.Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ\'s bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i\'ve gave this subject....hmmm well i don\'t know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."'




```python
# use beautiful soup instead of regex to remove mark up # a better practice
import re

lettersOnly =   re.sub('[^a-zA-Z]', ' ', example1.get_text())

print(lettersOnly)

lowerCase = lettersOnly.lower()
words = lowerCase.split()
```

     With all this stuff going down at the moment with MJ i ve started listening to his music  watching the odd documentary here and there  watched The Wiz and watched Moonwalker again  Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent  Moonwalker is part biography  part feature film which i remember going to see at the cinema when it was originally released  Some of it has subtle messages about MJ s feeling towards the press and also the obvious message of drugs are bad m kay Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring  Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him The actual feature film bit when it finally starts is only on for    minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord  Why he wants MJ dead so bad is beyond me  Because MJ overheard his plans  Nah  Joe Pesci s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno  maybe he just hates MJ s music Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence  Also  the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene Bottom line  this movie is for people who like MJ on one level or another  which i think is most people   If not  then stay away  It does try and give off a wholesome message and ironically MJ s bestest buddy in this movie is a girl  Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty  Well  with all the attention i ve gave this subject    hmmm well i don t know because people can be different behind closed doors  i know this for a fact  He is either an extremely nice but stupid guy or one of the most sickest liars  I hope he is not the latter  



```python
# removing stopword using nltk library
import nltk
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
    review_text = BeautifulSoup(review, "lxml").get_text()
    
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
```


```python
clean_review = review_to_words( train["review"][0] )
print(clean_review)
```

    stuff going moment mj started listening music watching odd documentary watched wiz watched moonwalker maybe want get certain insight guy thought really cool eighties maybe make mind whether guilty innocent moonwalker part biography part feature film remember going see cinema originally released subtle messages mj feeling towards press also obvious message drugs bad kay visually impressive course michael jackson unless remotely like mj anyway going hate find boring may call mj egotist consenting making movie mj fans would say made fans true really nice actual feature film bit finally starts minutes excluding smooth criminal sequence joe pesci convincing psychopathic powerful drug lord wants mj dead bad beyond mj overheard plans nah joe pesci character ranted wanted people know supplying drugs etc dunno maybe hates mj music lots cool things like mj turning car robot whole speed demon sequence also director must patience saint came filming kiddy bad sequence usually directors hate working one kid let alone whole bunch performing complex dance scene bottom line movie people like mj one level another think people stay away try give wholesome message ironically mj bestest buddy movie girl michael jackson truly one talented people ever grace planet guilty well attention gave subject hmmm well know people different behind closed doors know fact either extremely nice stupid guy one sickest liars hope latter



```python
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
```

    review of 1000 of 25000
    review of 2000 of 25000
    review of 3000 of 25000
    review of 4000 of 25000
    review of 5000 of 25000
    review of 6000 of 25000
    review of 7000 of 25000
    review of 8000 of 25000
    review of 9000 of 25000
    review of 10000 of 25000
    review of 11000 of 25000
    review of 12000 of 25000
    review of 13000 of 25000
    review of 14000 of 25000
    review of 15000 of 25000
    review of 16000 of 25000
    review of 17000 of 25000
    review of 18000 of 25000
    review of 19000 of 25000
    review of 20000 of 25000
    review of 21000 of 25000
    review of 22000 of 25000
    review of 23000 of 25000
    review of 24000 of 25000
    review of 25000 of 25000



```python
## Now that we have our training reviews tidied up, how do we convert them to 
## some kind of numeric representation for machine learning? One common approach
## is called a Bag of Words. The Bag of Words model learns a vocabulary from 
## all of the documents, then models each document by counting the number of 
## times each word appears.
    
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
```


```python
import sklearn
```


```python
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
```

    (25000, 5000)



```python
# take a look at the words in vocabulary
vocab = vectorizer.get_feature_names()
len(vocab)
```




    5000




```python
# count of each word in a vocabulary
# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
# for tag, count in zip(vocab, dist):
#     print(count, tag)
```


```python
# training the random forest
from sklearn.ensemble import RandomForestClassifier

# initialize the random forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# fit the forest to training set, using the bag of words as features and the
# the sentiment labels as the response variable

forest = forest.fit(train_data_features, train["sentiment"])
```


```python
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

```

    (25000, 2)
    review 1000 of 25000
    review 2000 of 25000
    review 3000 of 25000
    review 4000 of 25000
    review 5000 of 25000
    review 6000 of 25000
    review 7000 of 25000
    review 8000 of 25000
    review 9000 of 25000
    review 10000 of 25000
    review 11000 of 25000
    review 12000 of 25000
    review 13000 of 25000
    review 14000 of 25000
    review 15000 of 25000
    review 16000 of 25000
    review 17000 of 25000
    review 18000 of 25000
    review 19000 of 25000
    review 20000 of 25000
    review 21000 of 25000
    review 22000 of 25000
    review 23000 of 25000
    review 24000 of 25000
    review 25000 of 25000
