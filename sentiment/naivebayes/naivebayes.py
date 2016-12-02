from nltk.classify import NaiveBayesClassifier

class NaiveBayes:
	def __init__(self):
		return 

	def train(train_set):
        	print('Training classifier...')
        	classifier = NaiveBayesClassifier.train(train_set)
        	return classifier

	def test(classifier, test_set):
        	print('Testing classifier...')
        	return nltk.classify.accuracy(classifier, test_set)
