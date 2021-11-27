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

from preprocessing.preprocess import preprocessing
from classification_models.pipeline_sparkml import custom_model_pipeline, get_model
from clustering_models.kmeans_clustering import clustering

from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from spark_sklearn import linear_model
from sklearn.metrics import accuracy_score

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from sklearn.cluster import MiniBatchKMeans

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
	
	# Perform preprocessing
	df = preprocessing(df)
	
	# ==================Test preprocessing==============
	print('After preprocessing:\n')
	df.show()
	# ==================================================
	
	df=custom_model_pipeline(df, spark)
	
	print("After Pipeline")
	df.show()
	
	model=clustering(df, spark)
	
	with open('model.pkl', 'rb') as f:
		model = pickle.load(f)
    
	testingData=list(map(lambda line: Vectors.dense(line), df.select("count_vectors").collect()))
	vector = np.vectorize(np.float)#not required
	testingData = np.array(testingData)
	testingData = np.reshape(testingData,(testingData.shape[0], -1))
	testingData= vector(testingData)
    
	predictions = model.predict(testingData)
	actual=df.select("Sentiment").collect()
	
	accuracy=accuracy_score(actual,predictions)
	
	# print accuracy
	print ("Accuracy: {0:.4f}".format(accuracy))

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



