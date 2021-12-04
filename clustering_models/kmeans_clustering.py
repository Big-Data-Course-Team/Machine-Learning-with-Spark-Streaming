'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''

import matplotlib.pyplot as plt
import os

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import termplotlib as tpl
import plotext as plx

def clustering(df, spark, kmeans_model, num_iters):

	pca = PCA(2)
	trainingData = list(map(lambda line: Vectors.dense(line), df.select("count_vectors").collect()))
	vector = np.vectorize(float)#not required
	trainingData = np.array(trainingData)
	trainingData = np.reshape(trainingData,(trainingData.shape[0], -1))
	trainingData = vector(trainingData)
	
	trainingData = pca.fit_transform(trainingData)
	
	kmeans_model = kmeans_model.partial_fit(trainingData)
	
	print("KMeans Cluster Centers:", kmeans_model.cluster_centers_)
	
	predictions = kmeans_model.predict(trainingData)
	#actual = df.select("Sentiment").collect()
	
	#print(trainingData[:,0],trainingData[:,1])
	
	# TODO: Move plotting to a separate function/file
	plt.scatter(trainingData[:,0],trainingData[:,1],color = 'red')
	
	
	if not os.path.isdir('./Clusters'):
		os.mkdir('./Clusters')
	img_file=open("./Clusters/fig"+str(num_iters), "wb+")
	plt.savefig(img_file)

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
	return kmeans_model
	
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
