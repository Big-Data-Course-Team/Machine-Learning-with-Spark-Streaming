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
		old_vocab = self.vocabulary_
	else:
		old_vocab = {}

	self.fit(batch_data)

	
	old_vocab_set = set(old_vocab.keys())
	new_vocab = [None] * len(old_vocab)
	
	
	for k in old_vocab:
		new_vocab[old_vocab[k]] = k
		
	for k in self.vocabulary_:
		if k not in old_vocab_set:
			new_vocab += [k]
		
	self.vocabulary_ = {new_vocab[i] : i for i in range(len(new_vocab))}
	
	return self

def transformers_pipeline(df, spark, vectorizer, inputCols = ["tweet", "sentiment"], n=3):
	
	input_str = 'tokens_noStop'
	
	# Get a list of rows - each row is a list of strings (tokens without stop words)
	input_col = df.select(input_str).collect()
	
	# Used for the join - original column
	input_str_arr = [row[input_str] for row in input_col]
	
	# Used by count vectorizer - list of strings -> '["word", "another", "oh"]'
	input_arr = [str(a) for a in input_str_arr]
	
	# Count vectorizer - maps a list of documents to a matrix (sparse)
	vectorizer.partial_fit(input_arr)
	
	output_arr = vectorizer.transform(input_arr)
	output_arr = output_arr.toarray()

	output_col = list(map(lambda x: [x[0], x[1].tolist()], zip(input_str_arr, output_arr)))
	
	schema = StructType([
		StructField('tokens_noStop_copy', ArrayType(StringType())),
		StructField('count_vectors', ArrayType(IntegerType()))
	])
	
	dff = spark.createDataFrame(data=output_col, schema=schema)

	df = df.join(dff, dff.tokens_noStop_copy == df.tokens_noStop, 'inner')
	df = df.drop('tokens_noStop_copy')
	df.show()
	df.drop(['cleaned_tweets', 'tokens', 'tokens_noStop'])
	
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
	
	# Remove all content that are replied tweets
	new_df = new_df.withColumn('cleaned_tweets', regexp_replace('cleaned_tweets', 'RT\s*@[^:]*:.*', ""))
	
	
	# Converts each tweet to lowercase, splits by white spaces
	tokenizer = Tokenizer(inputCol='cleaned_tweets', outputCol='tokens')
	new_df = tokenizer.transform(new_df)
	
	# Remove stop words
	remover = StopWordsRemover(inputCol='tokens', outputCol='tokens_noStop')
	new_df = remover.transform(new_df)

	return new_df
