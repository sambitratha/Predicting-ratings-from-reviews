import json
import re
from matplotlib import pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pickle
import nltk
import numpy as np
import PreProcess as pre


words = None
wordmap = None
reviews = None

def load_wordmap():
	global words, wordmap
	wordsfile = open("words", 'ab')
	words  = pickle.load(wordsfile)
	wordmapfile = open("wordmap", 'ab')
	wordmap  = pickle.load(wordmapfile)
	return


def train(epochs, reviews, users, products):
	for epoch in range(epochs):
		loss = {}
		totalerror = 0.0
		for pair in reviews:
			reviewtext = reviews[pair][:-1]
			actualrating = float(reviews[pair][-1])
			predictedrating = predict(reviewtext, pair[0], pair[1])
			error = predictedrating - actualrating
			loss[(pair[0], pair[1])] = error
			totalerror += error
		print "epoch ", epoch, "error = ", totalerror
		update(loss, reviews)


def predict(review, user, product):
	rating = 0.0
	for word in review:
		w = 0.0
		for i in range(D):
			w += (model["V"][product][i]) * (model["U"][user][i]) * (model["P"][words.index(word)][i])
		rating += (model["W"][words.index(word)] + w)		
	return rating


def update(loss, reviews):
	global model

	#update 'W' parameter of model
	for (userid, productid) in loss:
		for word in reviews[(userid, productid)][:-1]:
			model["W"][words.index(word)] -= 0.000003 * loss[(userid, productid)]

	#update 'U' parameter of model
	for user in users:
		userid = users[user]
		productlist = pre.getproductlist(userid)
		for product in productlist:
			factor = np.zeros(D)
			for word in reviews[(userid, product)][:-1]:
				factor += model["P"][words.index(word)]
			model["U"][userid] -= 0.000003 * loss[(userid, product)] * (factor * model["V"][product])

	#update 'V' parameter of model
	for product in products:
		productid = products[product]
		userlist = pre.getreviewers(productid)
		for user in userlist:
			factor = np.zeros(D)
			for word in reviews[(user, productid)][:-1]:
				factor += model["P"][words.index(word)]
			model["V"][productid] -= 0.000003 * loss[(user, productid)] * (factor * model["U"][user])

	#update 'P' parameter of model
	for wordid in range(len(words)):
		for (userid, productid) in reviews:
			if words[wordid] in reviews[(userid, productid)][:-1]:
				model["P"][wordid] -= 0.000003 * loss[(userid, productid)] * (model["U"][userid] * model["V"][productid])


D = 10

if __name__ == '__main__':
	pre.extract_words()
	words = pre.getwords()
	reviews = pre.getreviews()
	products = pre.getproducts()
	users = pre.getusers()
	model = {
		"W" : np.random.rand(len(words)),
		"U" : np.random.rand(len(users), D),
		"V" : np.random.rand(len(products), D),
		"P" : np.random.rand(len(words), D)	
	}
	epochs = 10
	train(epochs, reviews, users, products)

