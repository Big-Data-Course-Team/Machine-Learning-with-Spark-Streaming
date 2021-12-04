'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''

#from pyspark.ml.feature import VectorAssembler, StringIndexer, ChiSqSelector


from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover

from pyspark.sql.types import *


# Custom function for partial fitting of CV
def partial_fit(self, batch_data):

	if(hasattr(self, 'vocabulary_')):
		
		# Store previous vocabulary
		old_vocab = self.vocabulary_
		old_vocab_len = len(old_vocab)
		print("Old vocab:")
		print(list(old_vocab.items())[:20])
		print("Length:", old_vocab_len)
		
		# Construct vocabulary from the current batch
		self.fit(batch_data)
		print("Fitted vocab:")
		print(list(self.vocabulary_.items())[:20])
		
		# Increment indices of the words in the new vocab
		for word in list(self.vocabulary_.keys()):
			if word in old_vocab:
				del self.vocabulary_[word]
			else:
				self.vocabulary_[word] += old_vocab_len
		
		# Append and set new vocab
		old_vocab.update(self.vocabulary_)
		self.vocabulary_ = old_vocab
		print("Final vocab:")
		print(list(self.vocabulary_.items())[:20])

	else:
		self.fit(batch_data)
		
	return self

# Count vectorizer transformation on the column of tweets
def CV_transformer(df, spark, vectorizer, inputCols = ["tweet", "sentiment"], n=3):
	
	input_str = 'tokens_noStop'
	
	# Get a list of rows - each row is a list of strings (tokens without stop words)
	input_col = df.select(input_str).collect()
	
	# Used for the join - original column
	input_str_arr = [row[input_str] for row in input_col]
	
	# Used by count vectorizer - list of strings -> '["word", "another", "oh"]'
	input_arr = [str(a) for a in input_str_arr]
	
	# Count vectorizer - maps a list of documents to a matrix (sparse)
	vectorizer.partial_fit(input_arr)
	
	# Tranform list of tweets to list of sparse array representations
	output_arr = vectorizer.transform(input_arr)
	output_arr = output_arr.toarray()

	# Create a dataframe containing output lists to join with the original 
	output_col = list(map(lambda x: [x[0], x[1].tolist()], zip(input_str_arr, output_arr)))
	
	schema = StructType([
		StructField('tokens_noStop_copy', ArrayType(StringType())),
		StructField('count_vectors', ArrayType(IntegerType()))
	])
	
	dff = spark.createDataFrame(data=output_col, schema=schema)
	
	# Join the original and newly created dataframes on the input column
	df = df.join(dff, dff.tokens_noStop_copy == df.tokens_noStop, 'inner')
	df = df.drop('tokens_noStop_copy')

	return df

	# ------------------------------ Pipeline worked on till CV (to be tested) ----------------------------------------------
		
		# Compute the IDF score given a set of tweets
		#idf = IDF(inputCol="{0}_cv".format(i), outputCol="{0}_tfidf".format(i), minDocFreq=5)
		#df = cv.transform(idf)

	# Merges multiple columns into a vector column
	#assembler = VectorAssembler(inputCols=["{0}_tfidf".format(i) for i in range(1, n + 1)], outputCol="rawFeatures")
	
	#label_stringIdx = StringIndexer(inputCol = "Sentiment", outputCol = "label")
	
	#selector = ChiSqSelector(numTopFeatures=2**14,featuresCol='rawFeatures', outputCol="features")
	
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
