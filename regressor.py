import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
import PreProcess as pre
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

words = None
trainx = []
trainy = []

def process_data():
	global trainx, trainy

	with open("dataset/smallData.txt", 'r') as f:
		stop = stopwords.words('english')
		stemmer = SnowballStemmer("english")
		lemmatizer = WordNetLemmatizer()
		current = 0
		while current < 1000:
			print "inside loop"
			current += 1
			identity = f.readline()
			if identity == "finished":
				break

			bookid = f.readline()
			text = f.readline()
			rating = f.readline()
			tokens = word_tokenize(text)
			bitmap = [0] * len(words)
			print "entering"
			for token in tokens:
				token = token.lower()
				if not re.search('[a-zA-Z]', token):
					continue
				if token in stop:
					continue
				token = lemmatizer.lemmatize(token)
				token = stemmer.stem(token)
				bitmap[words.index(token)] = 1
			trainx.append(bitmap)
			trainy.append(float(rating))
			print "exiting"
        	print "appended"
        	



if __name__ == '__main__':
	words = pre.extract_words()
	process_data()
	print len(trainx), len(trainx[0]), len(words)
	regr = AdaBoostRegressor(n_estimators = 850)
	regr.fit(np.array(trainx[:900]), np.array(trainy[:900]))
	print len(trainx)
	pred = regr.predict(np.array(trainx[900:]))
	print pred
	print trainy[900:]
	print "mean square error is ", mean_squared_error(np.array(trainy[900:]), pred)