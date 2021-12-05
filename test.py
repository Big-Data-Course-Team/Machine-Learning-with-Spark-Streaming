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
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA

from preprocessing.preprocess import *
from classification_models.logistic_regression import *
from classification_models.multinomial_nb import *
from classification_models.passive_aggressive import *
from clustering_models.kmeans_clustering import *

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

# Define the Incremental PCA
pca = IncrementalPCA(n_components=2)

# Define MinMax Scaler
minmaxscaler = MinMaxScaler()

# Define CountVectorizer
CountVectorizer.cv_partial_fit = cv_partial_fit
cv = CountVectorizer(lowercase=True, analyzer = 'word', stop_words='english', ngram_range=(1,2))

# Define HashVectorizer - TODO: figure out how to get HV to work
hv = HashingVectorizer(n_features=2**13, lowercase=True, analyzer = 'word', stop_words='english', ngram_range=(1,2))

# Define, initialize BatchKMeans Model
num_clusters = 2
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 
							   init_size=1000, batch_size=1000, verbose=False, max_iter=1000)

# Define LR Model
lr_model = SGDClassifier(loss='log')

# Define NB Model
multi_nb_model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

# Define PA Model
pac_model = PassiveAggressiveClassifier(C = 0.5, random_state = 5)


'''
 ---------------------------- Processing -------------------------------------------
'''
# Process each stream - needs to run ML models
def process(rdd):
	
	global schema, spark
	global schema, spark, pca, minmaxscaler, cv, hv, \
		   lr_model, multi_nb_model, pac_model, kmeans_model
	
	# ==================Dataframe Creation=======================
	
	# Collect all records
	records = rdd.collect()
	
	# List of dicts
	dicts = [i for j in records 
					 for i in list(json.loads(j).values())]
	
	if len(dicts) == 0:
		return
	
	# Create a DataFrame with each stream	
	df = spark.createDataFrame((Row(**d) for d in dicts), 
								schema)
	# ============================================================
	
	# ==================Data Cleaning======================
	df = df_preprocessing(df)
	# ============================================================
	
	# ==================Preprocessing======================
	df = transformers_pipeline(df, spark, pca, minmaxscaler, hv)
	# ============================================================
	
	# ==================Getting Testing Data and Actual Labels======================
	testingData = df.select("hashed_vectors", "sentiment").collect()
	X_test = np.array(list(map(lambda row: row.hashed_vectors, testingData)))
	y_test = np.array(list(map(lambda row: row.sentiment, testingData)), dtype='int64')
	# ============================================================
	
	'''
	 ---------------------------- CLASSIFICATION -------------------------------------------
	'''
		
	# ==================Testing Logistic Regression=======================
	with open('./models/lr_model.pkl', 'rb') as f:
		lr_model = pickle.load(f)
		
	predictions_lr = lr_model.predict(X_test)	
	accuracy_lr = np.count_nonzero(np.array(predictions_lr) == y_test)/y_test.shape[0]	
	print("Accuracy of LR:", accuracy_lr)
	# ============================================================
	
	# ==================Testing Multinomial Naive Bayes=======================
	
	testingData_mnb = df.select("minmax_pca_vectors", "sentiment").collect()
	X_test_mnb = np.array(list(map(lambda row: row.minmax_pca_vectors, testingData_mnb)))
	y_test_mnb = np.array(list(map(lambda row: row.sentiment, testingData_mnb)), dtype='int64')
	
	with open('./models/multi_nb_model.pkl', 'rb') as f:
		multi_nb_model = pickle.load(f)
		
	predictions_mnb = multi_nb_model.predict(X_test_mnb)	
	accuracy_mnb = np.count_nonzero(np.array(predictions_mnb) == y_test_mnb)/y_test_mnb.shape[0]	
	print("Accuracy of NB:", accuracy_mnb)
	# ============================================================
	
	# ==================Testing Passive Aggressive Model=======================
	with open('./models/pac_model.pkl', 'rb') as f:
		pac_model = pickle.load(f)
		
	predictions_pac = pac_model.predict(X_test)	
	accuracy_pac = np.count_nonzero(np.array(predictions_pac) == y_test)/y_test.shape[0]	
	print("Accuracy of PAC:", accuracy_pac)
	# ============================================================
	
	'''
	 ---------------------------- CLUSTERING -------------------------------------------
	'''	
	
	# ==================Testing KMeans Model=======================
	with open('./models/kmeans_model.pkl', 'rb') as f:
		kmeans_model = pickle.load(f)
		
	X_test = df.select('pca_vectors').collect()
	
	X_test = np.array([row['pca_vectors'] for row in X_test])
	X_test = np.reshape(X_test, (X_test.shape[0], -1))
		
	predictions_kmeans = kmeans_model.predict(X_test)	
	
	preds_1_kmeans = [4 if i == 1 else 0 for i in predictions_kmeans]
	preds_2_kmeans = [0 if i == 1 else 4 for i in predictions_kmeans]
	
	accuracy_1_kmeans = np.count_nonzero(np.array(preds_1_kmeans) == np.array(y_test)) / y_test.shape[0]
	accuracy_2_kmeans = np.count_nonzero(np.array(preds_2_kmeans) == np.array(y_test)) / y_test.shape[0]
			
	print('Accuracy of KMeans: ', accuracy_1_kmeans, accuracy_2_kmeans)
	# ============================================================
	

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



