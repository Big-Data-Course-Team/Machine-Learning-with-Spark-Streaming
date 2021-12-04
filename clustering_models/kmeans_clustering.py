'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''

import matplotlib.pyplot as plt
import numpy as np
import termplotlib as tpl
import plotext as plx
import os

from pyspark.mllib.clustering import StreamingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from sklearn.metrics import accuracy_score

plt.rcParams.update({'figure.figsize':(16, 9), 'figure.dpi':100})

def clustering(df, spark, kmeans_model, num_iters):
	
	X_train = df.select('pca_vectors').collect()
	
	X_train = np.array([row['pca_vectors'] for row in X_train])
	X_train = np.reshape(X_train, (X_train.shape[0], -1))
	
	actual_labels = df.select('sentiment').collect()
	actual_labels = [row['sentiment'] for row in actual_labels]
	actual_labels = list(map(int, actual_labels))
	
	kmeans_model = kmeans_model.partial_fit(X_train)
	
	print("KMeans Cluster Centers:", kmeans_model.cluster_centers_)
	
	pred_labels = kmeans_model.predict(X_train)

	pred_labels_1 = [4 if i == 1 else 0 for i in pred_labels]
	pred_labels_2 = [0 if i == 1 else 4 for i in pred_labels]
	
	accuracy_1 = np.count_nonzero(np.array(pred_labels_1) == np.array(actual_labels)) / pred_labels.shape[0]
	accuracy_2 = np.count_nonzero(np.array(pred_labels_2) == np.array(actual_labels)) / pred_labels.shape[0]
			
	print('Accuracy of KMeans: ', accuracy_1, accuracy_2)

	figure, axis = plt.subplots(1, 2)
	axis[0].scatter(X_train[:, 0], X_train[:, 1], c=pred_labels)
	axis[0].set_title('KMeans Clusters')
	axis[0].set_xlabel('PCA1')
	axis[0].set_ylabel('PCA2')
	
	axis[1].scatter(X_train[:, 0], X_train[:, 1], c=actual_labels)
	axis[1].set_title('Original Labels')
	axis[1].set_xlabel('PCA1')
	axis[1].set_ylabel('PCA2')
	
	if not os.path.isdir('./cluster_plots'):
		os.mkdir('./cluster_plots')

	img_file = open("./cluster_plots/Batch_" + str(num_iters), "wb+")
	plt.savefig(img_file)

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
