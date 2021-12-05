'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''

from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.types import *


# Custom function for partial fitting of CV
def cv_partial_fit(self, batch_data):

	if(hasattr(self, 'vocabulary_')):
		
		# Store previous vocabulary
		old_vocab = self.vocabulary_
		old_vocab_len = len(old_vocab)
		
		# Construct vocabulary from the current batch
		self.fit(batch_data)
		
		# Increment indices of the words in the new vocab
		for word in list(self.vocabulary_.keys()):
			if word in old_vocab:
				del self.vocabulary_[word]
			else:
				self.vocabulary_[word] += old_vocab_len
		
		# Append and set new vocab
		old_vocab.update(self.vocabulary_)
		self.vocabulary_ = old_vocab

	else:
		self.fit(batch_data)
		
	return self
	
def df_preprocessing(dataframe):

	new_df = dataframe
	
	# Remove null values 
	new_df = new_df.na.replace('', None)
	new_df = new_df.na.drop()
	
	# Remove all mentioned users
	new_df = new_df.withColumn('cleaned_tweets', regexp_replace('tweet', '@\w+', ""))
	
	# Remove all punctuations - TODO: Keep or not?
	new_df = new_df.withColumn('cleaned_tweets', regexp_replace('cleaned_tweets', '[^\w\s]', ""))
	
	# Remove URLs
	new_df = new_df.withColumn('cleaned_tweets', regexp_replace('cleaned_tweets', r'http\S+', ""))
	new_df = new_df.withColumn('cleaned_tweets', regexp_replace('cleaned_tweets', r'www\S+', ""))
	
	# Remove all content that are replied tweets
	new_df = new_df.withColumn('cleaned_tweets', regexp_replace('cleaned_tweets', 'RT\s*@[^:]*:.*', ""))
	
	
	# Converts each tweet to lowercase, splits by white spaces
	tokenizer = Tokenizer(inputCol='cleaned_tweets', outputCol='tokens')
	new_df = tokenizer.transform(new_df)
	
	# Remove stop words
	remover = StopWordsRemover(inputCol='tokens', outputCol='tokens_noStop')
	new_df = remover.transform(new_df)

	return new_df
