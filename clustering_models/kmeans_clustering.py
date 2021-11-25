from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans
from sklearn.cluster import MiniBatchKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import numpy as np


def clustering(df, spark):

	trainingData=list(map(lambda line: Vectors.dense(line), df.select("count_vectors").collect()))
	vector = np.vectorize(np.float)#not required
	trainingData = np.array(trainingData)
	trainingData = np.reshape(trainingData,(trainingData.shape[0], -1))
	trainingData= vector(trainingData)
	num_clusters = 10
	kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 
		                     init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
	
	kmeans = kmeans_model.partial_fit(trainingData)
	

	# printing the predicted cluster assignments on new data points as they arrive.
	'''
	result = model.predictOnValues(testdata.map(lambda lp: (lp.label, lp.features)))
	result.pprint()
	'''
	#prediction=list(map(lambda line: Vectors.dense(line), df.select("sentiment").collect()))
	#silhouette_score=[]
	#Silhouette Score using ClusteringEvaluator() measures how close each point in one cluster is to points in the neighboring clusters
	#evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='trainingData', metricName='silhouette', distanceMeasure='squaredEuclidean')
	#score=evaluator.evaluate(trainingData)
	#silhouette_score.append(score)
	#print("Silhouette Score:",score)	                         
	



	#print("Final centers: " + str(model.latestModel().centers))






	'''
	from pyspark.ml.clustering import KMeans
	from pyspark.ml.evaluation import ClusteringEvaluator
	from pyspark.ml.feature import VectorAssembler
	from pyspark.ml.feature import StandardScaler
	#vector assembler not required if we are dealing with 1 column
	# vector assembler transforms  a set of features into a single vector
	assemble=VectorAssembler(inputCols=['SENTIMENT','TWEET'], outputCol='features')
	#df  is the batch dataframe after preprocessing
	assembled_data=assemble.transform(df)
	s
	#standardizing the features column
	sc=StandardScaler(inputCol='features',outputCol='standardized')
	data_scale=sc.fit(assembled_data)
	output=data_sc.transform(assembled_data)
	#data_scale_output.show(2)
	silhouette_score=[]
	#Silhouette Score using ClusteringEvaluator() measures how close each point in one cluster is to points in the neighboring clusters
	evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized', \
		                            metricName='silhouette', distanceMeasure='squaredEuclidean')
	#Forming clusters of 2-10                            
	for i in range(2,10):
		
		KMeans_algo=KMeans(featuresCol='standardized', k=i)
		
		KMeans_fit=KMeans_algo.fit(output)
		
		output_kn=KMeans_fit.transform(output)
		
		
		
		score=evaluator.evaluate(output_km)
		
		silhouette_score.append(score)
		
		#print("Silhouette Score:",score)
	'''
