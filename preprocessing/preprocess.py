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

def transformers_pipeline(df, spark, pca, minmaxscaler, vectorizer, inputCols = ["tweet", "sentiment"], n=3):
	
	# Get a list of rows - each row is a list of strings (tokens without stop words)
	input_col = df.select('tokens_noStop').collect()
	
	# Used for the join - original column
	input_str_arr = [row['tokens_noStop'] for row in input_col]
	
	input_arr = [str(a) for a in input_str_arr]
	
	vectorizer.partial_fit(input_arr)
	
	output_arr = vectorizer.transform(input_arr)
	output_arr = output_arr.toarray()
	
	labels_col = df.select('sentiment').collect()
	
	labels_str_arr = [row['sentiment'] for row in labels_col]
	
	labels_arr = [int(a) for a in labels_str_arr]
	
	pca_X_train = list(map(lambda x: x.tolist(), output_arr))
	pca.partial_fit(pca_X_train, labels_arr)
	pca_transformed = pca.transform(pca_X_train)
	
	minmaxscaler.partial_fit(pca_transformed)
	minmax_pca = minmaxscaler.transform(pca_transformed)	
#	minmax_hash = minmaxscaler.transform(pca_transformed)
	
	output_col = list(map(lambda x: [x[0], x[1].tolist(), x[2].tolist(), x[3].tolist()], \
							zip(input_str_arr, output_arr, pca_transformed, minmax_pca)))
	
	schema = StructType([
		StructField('tokens_noStop_copy', ArrayType(StringType())),
		StructField('hashed_vectors', ArrayType(FloatType())),
		StructField('pca_vectors', ArrayType(FloatType())),
		StructField('minmax_pca_vectors', ArrayType(FloatType())),
	])
	
	dff = spark.createDataFrame(data=output_col, schema=schema)

	df = df.join(dff, dff.tokens_noStop_copy == df.tokens_noStop, 'inner')
	df = df.drop(*['tweet', 'cleaned_tweets', 'tokens', 'tokens_noStop', 'tokens_noStop_copy'])
	
	return df
	
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
