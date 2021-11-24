'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''

from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover

def preprocessing(dataframe):

	# Converts the input string to lowercase, splits by white spaces
    tokenizer = Tokenizer(inputCol='tweet', outputCol='words')
	new_df = tokenizer.transform(dataframe)
	
	# Remove null values 
    new_df = new_df.na.replace('', None)
    new_df = new_df.na.drop()
    
    # Regex expressions
    new_df = new_df.withColumn('words', regexp_replace('words', r'http\S+', ''))
    new_df = new_df.withColumn('words', regexp_replace('words', '@\w+', ''))
    new_df = new_df.withColumn('words', regexp_replace('words', '#', ''))
    new_df = new_df.withColumn('words', regexp_replace('words', 'RT', ''))
    new_df = new_df.withColumn('words', regexp_replace('words', ':', ''))
    
    # Remove stop words
	remover = StopWordsRemover(inputCol='words', outputCol='words_none_stop')
	new_df = remover.transform(new_df)

    return new_df
