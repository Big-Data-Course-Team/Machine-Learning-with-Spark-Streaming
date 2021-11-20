import json
import importlib

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row, SparkSession

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
#from pyspark.mllib.classification import StreamingLogisticRegressionWithSGD


from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import NGram, VectorAssembler
from pyspark.ml.feature import ChiSqSelector

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from spark_sklearn import linear_model

from classification_models.pipeline_sparkml import model_pipeline

# Create a local StreamingContext with two execution threads
sc = SparkContext("local[2]", "Sentiment")
	
spark = SparkSession \
.builder \
.config(conf=SparkConf()) \
.getOrCreate()

# Batch interval of 5 seconds - TODO: Change according to necessity
ssc = StreamingContext(sc, 5)

sql_context = SQLContext(sc)
	
# Set constant for the TCP port to send from/listen to
TCP_IP = "localhost"
TCP_PORT = 6100
	
# Create schema
schema = StructType([
	StructField("sentiment", StringType(), False),
	StructField("tweet", StringType(), False),
])

# Process each stream - needs to run ML models
def process(rdd):
	global schema, spark
	
	# Collect all records
	records = rdd.collect()
	
	# List of dicts
	dicts = [i for j in records for i in list(json.loads(j).values())]

	
	if len(dicts) == 0:
		return
		
	df = spark.createDataFrame((Row(**d) for d in dicts), schema)
	df.show()



# Main entry point for all streaming functionality
if __name__ == '__main__':

	# Create a DStream - represents the stream of data received from TCP source/data server
	# Each record in 'lines' is a line of text
	lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	# TODO: check if split is necessary
	json_str = lines.flatMap(lambda x: x.split('\n'))
	
	# Process each RDD
	lines.foreachRDD(process)

	# The data is streamed as a JSON string (you can see this by observing the code in stream.py). 
	# You will first need to parse the JSON string, obtain the rows in each batch and then convert it to a DataFrame. 
	# The structure of the JSON string has been provided in the streaming file. 
	# Use this to parse your JSON to obtain rows, and then convert these rows into a DataFrame.
	
	pipeline=Pipeline.load("./pipeline")
	test_set = model.transform(lines)
	#import pickle
	with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
	predictions = model.predict(test_set)
	accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

	roc_auc = evaluator.evaluate(predictions)
	# print accuracy, roc_auc
	print ("Accuracy Score: {0:.4f}".format(accuracy))
	print ("ROC-AUC: {0:.4f}".format(roc_auc))
	

	# Start processing after all the transformations have been setup
	ssc.start()             # Start the computation
	ssc.awaitTermination()  # Wait for the computation to terminate

