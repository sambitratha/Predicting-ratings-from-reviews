
# coding: utf-8

# In[42]:


import json
import re
from matplotlib import pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

products = {}
users = {}
reviews = {}
reviewerlist = {}
productlist = {}
words = []
# In[43]:


def extract_words():
    global products, users, reviews, reviewerlist, productlist, words
    
    usercount = 0
    productcount = 0
    with open("dataset/smallData.txt", 'r') as f:
        stop = stopwords.words('english')
        stemmer = SnowballStemmer("english")
        lemmatizer = WordNetLemmatizer()
        
        while True:
            identity = f.readline()
            if identity == "finished":
                break
            if identity not in users:
            	users[identity] = usercount
            	usercount += 1
           
            bookid = f.readline()
            if bookid not in products:
            	products[bookid] = productcount
            	productcount += 1

            if users[identity] not in productlist:
            	productlist[users[identity]] = [products[bookid]]
            else:
            	productlist[users[identity]].append(products[bookid])

            if products[bookid] not in reviewerlist:
            	reviewerlist[products[bookid]] = [users[identity]]
            else:
            	reviewerlist[products[bookid]].append(users[identity])

            text = f.readline()
            rating = f.readline()
            if (users[identity], products[bookid]) not in reviews:
            	reviews[(users[identity], products[bookid])] = []
            tokens = word_tokenize(text)
            
            for token in tokens:
                token = token.lower()
                if not re.search('[a-zA-Z]', token):
                    continue
                if token in stop:
                    continue
                token = lemmatizer.lemmatize(token)
                token = stemmer.stem(token)
                if token not in words:
                    words.append(token)
                    reviews[(users[identity], products[bookid])].append(token)
              
            reviews[(users[identity], products[bookid])].append(rating)

    return words


# In[46]:

'''
def save_info(words, wordmap):
    import pickle
    wordsfile = open("words", 'ab')
    pickle.dump(words, wordsfile)
    wordmapfile = open("wordmap", 'ab')
    pickle.dump(wordmap, wordmapfile)
    
    wordsfile.close()
    wordmapfile.close()
'''
def getwords():
	return words

def getreviews():
	return reviews

def getproducts():
	return products

def getusers():
	return users

def getreviewers(bookid):
	if bookid in reviewerlist:
		return reviewerlist[bookid]
	else:
		return None

def getproductlist(userid):
	if userid in productlist:
		return productlist[userid]
	return None

# In[47]:


if __name__ == '__main__':
    words = extract_words()
    save_info(words, wordmap)

