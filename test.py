'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''

'''
 ---------------------------- Import definitions ------------------------------------
'''
import pickle
import json
import importlib

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.types import *

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier

from preprocessing.preprocess import *
from classification_models.pipeline_sparkml import *
from classification_models.logistic_regression import *
from clustering_models.kmeans_clustering import clustering

from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from spark_sklearn import linear_model
from sklearn.metrics import accuracy_score

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

import numpy as np

'''
 ---------------------------- Constant definitions ----------------------------------
'''
# Set constant for the TCP port to send from/listen to
TCP_IP = "localhost"
TCP_PORT = 6100

# Create schema
schema = StructType([
	StructField("sentiment", StringType(), False),
	StructField("tweet", StringType(), False),
])

'''
 ---------------------------- Spark definitions -------------------------------------
'''
# Create a local StreamingContext with two execution threads
sc = SparkContext("local[2]", "Sentiment")
sc.setLogLevel("WARN")
sql_context = SQLContext(sc)
	
spark = SparkSession \
.builder \
.config(conf=SparkConf()) \
.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Batch interval of 5 seconds - TODO: Change according to necessity
ssc = StreamingContext(sc, 5)

'''
 ---------------------------- Model definitions -------------------------------------
'''

# Define CountVectorizer
CountVectorizer.partial_fit = partial_fit
vectorizer = CountVectorizer(lowercase=True, analyzer = 'word', stop_words='english', ngram_range=(1,2))

# Define HashVectorizer - TODO: figure out how to get HV to work
#vectorizer = HashingVectorizer(lowercase=True, analyzer = 'word', stop_words='english', ngram_range=(1,2))

# Define, initialize BatchKMeans Model
num_clusters = 2
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 
							   init_size=1000, batch_size=1000, verbose=False, max_iter=1000)


lr_model = SGDClassifier(loss='log')

'''
 ---------------------------- Processing -------------------------------------------
'''
# Process each stream - needs to run ML models
def process(rdd):
	
	global schema, spark
	
	# Collect all records
	records = rdd.collect()
	
	# List of dicts
	dicts = [i for j in records for i in list(json.loads(j).values())]
	
	if len(dicts) == 0:
		return
	
	# Create a DataFrame with each stream	
	df = spark.createDataFrame((Row(**d) for d in dicts), schema)
	
	# ==================Data Cleaning + Test============
	df = df_preprocessing(df)
	print('\nAfter cleaning:\n')
	df.show()
	# ==================================================
	
	# ==================Preprocessing + Test============
	df = transformers_pipeline(df, spark, vectorizer)
	print("\nAfter Preprocessing:\n")
	df.show()
	# ==================================================
	
	
	input_str="tokens_noStop"	
	with open('model.pkl', 'rb') as f:
		model = pickle.load(f)
	
	#vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
	'''
	
	input_col = df.select(input_str).collect()
	#input_str_arr = [row[input_str] for row in input_col]
	input_arr = [str(a) for a in input_col]
	'''
	pca = PCA(2)
	testingData=list(map(lambda line: Vectors.dense(line), df.select("count_vectors").collect()))
	testingData = np.array(testingData)
	vector = np.vectorize(float)#not required
	#testingData = vectorizer.transform(input_arr)
	#testingData = testingData.toarray()
	testingData = np.reshape(testingData,(testingData.shape[0], -1))
	testingData= vector(testingData)
	
	testingData = pca.fit_transform(testingData)
	
	predictions = model.predict(testingData)
	#actual=df.select("Sentiment").collect()
	actual= df.select('Sentiment').rdd.map(lambda row : row[0]).collect()
	
	
	actual=[int(i) for i in actual]
	predictions=[int(i) for i in predictions]
	
	act=list()
	for i in actual:
		#if i==4:
		#	i=1
		act.append(i)
	correct=0
	for i in range(len(act)):
		if act[i]==predictions[i]:
			correct+=1
	accuracy=correct/len(act)
	print ("Accuracy: ", accuracy)

# Main entry point for all streaming functionality
if __name__ == '__main__':

	# Create a DStream - represents the stream of data received from TCP source/data server
	# Each record in 'lines' is a line of text
	lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	# TODO: check if split is necessary
	json_str = lines.flatMap(lambda x: x.split('\n'))
	
	# Process each RDD
	lines.foreachRDD(process)

	# Start processing after all the transformations have been setup
	ssc.start()             # Start the computation
	ssc.awaitTermination()  # Wait for the computation to terminate



