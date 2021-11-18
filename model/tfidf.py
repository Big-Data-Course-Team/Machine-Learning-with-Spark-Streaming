from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, TokeniEr
from pyspark.ml import Pipeline

'''
 TF-IDF: 
 - Vectorization technique
 - Converts tweets to vectors of length the size of the vocab of the corpus
 - Each value represents the TF-IDF score (more frequently in one tweet and less in others - more important)

 TF-IDF Score:
 - TF x IDF (term frequency * inverse document frequency)
 - TF = (Frequency of a word in a tweet) / (Total words in the tweet)
 - IDF = log( (Total number of tweets) / (Number of tweets containing the word))
 - IDF favors rare words, better for classification
'''

# Split the entire corpus to form a bag of words
tokenizer = Tokenizer(inputCol=<tweetCol>, 
					  outputCol=<outputFromTokenizer>)
wordsData = tokenizer.transform(<dataFrame>)

# Hash the bag of words into a feature vector
hashingTF = HashingTF(inputCol=<outputFromTokenizer>, 
					  outputCol=<outputFromHash>, 
					  numFeatures=20)

featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors

# Rescale the feature vectors for each tweet - Improves performance when text is used as features
idf = IDF(inputCol=<outputFromHash>, 
		  outputCol=<scaledFeatureVectors>)

idfModel = idf.fit(featurizedData)

rescaledData = idfModel.transform(featurizedData)

rescaledData.select(<labelColFromOriginalDF>,
					<scaledFeatureVectors>) \
			.show()
