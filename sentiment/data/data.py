import nltk, random, glob
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

class Data:
	tokenizer = RegexpTokenizer(r'\w+')
	stopWords = stopwords.words('english')

	def word_feats(words):
		return dict([(word, True) for word in words])

	def preProcess(self, review):
		#remove punctuation and translate to lower case
		words = self.tokenizer.tokenize(review.lower())

		#remove stop words
		content = [w for w in words if w not in self.stopWords]

		#stem the words
		stemmer = nltk.stem.porter.PorterStemmer()
		stems = [stemmer.stem(w) for w in content]
		return stems

	def loadData(src, splitAt=0.50):
		featureSet = []
		for f in glob.glob(src + "neg/*.txt"):
			review = open(f).read()
			words = Data.preProcess(Data, review)
			featureSet.append((Data.word_feats(words), 0))

		for f in glob.glob(src + "pos/*.txt"):
			review = open(f).read()
			words = Data.preProcess(Data, review)
			featureSet.append((Data.word_feats(words), 1))

		indices = [i for i in range(len(featureSet))]
		random.shuffle(indices)

		split_at = int(len(featureSet)*splitAt)
		training_idx, testing_idx = indices[:split_at], indices[split_at:]

		train_set = [featureSet[i] for i in training_idx]
		test_set = [featureSet[i] for i in testing_idx]

		return train_set, test_set
