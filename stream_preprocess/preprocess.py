from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import json

# Main entry point for all streaming functionality

# Create a local StreamingContext with two execution threads
sc = SparkContext("local[2]", "Sentiment")


spark = SparkSession \
.builder \
.config(conf=SparkConf()) \
.getOrCreate()

# Batch interval of 1 second - TODO: Change according to necessity
ssc = StreamingContext(sc, 1)

sql_context = SQLContext(sc)

# Set constant for the TCP port to send from/listen to
TCP_IP = "localhost"
TCP_PORT = 6100

row_jsons = []

# Create a DStream - represents the stream of data received from TCP source/data server
# Each record in 'lines' is a line of text
lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

# Create schema
schema = StructType([
	StructField("feature0", StringType(), False),
	StructField("feature1", StringType(), False),
])

# Create empty dataframe
df = spark.sparkContext.emptyRDD().toDF(schema)

# Each batch is a json
batch_dict = lines.map(lambda b: json.loads(b))

# Extract each individual row
row_jsons = batch_dict.map(lambda y: list(map(lambda k: y[k], y)))

#dfs = batch_dict.map(lambda y: spark.read.json(sc.parallelize( list(map(lambda k: json.dumps(y[k]), y))) ) )

#row_jsons.pprint()

def append_to_df(rdd):
	global df, schema
	appended = df.union(spark.read.schema(schema).json(rdd))
	df = appended
	df.show()
	

# Append each row to the dataframe	
row_jsons.foreachRDD(lambda rdd: append_to_df(rdd))
 
#df.show()


# The data is streamed as a JSON string (you can see this by observing the code in stream.py). 
# You will first need to parse the JSON string, obtain the rows in each batch and then convert it to a DataFrame. 
# The structure of the JSON string has been provided in the streaming file. 
# Use this to parse your JSON to obtain rows, and then convert these rows into a DataFrame.

# Start processing after all the transformations have been setup
ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

