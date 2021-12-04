'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''
import numpy as np

from pyspark.mllib.linalg import Vectors
from sklearn.decomposition import PCA, IncrementalPCA


def MultiNBLearning(df, spark, classifier):
	"""
	Perform online learning of a Multinomial Naive Bayes model with batches of data
	"""
	
	# Define PCA for 4 features
	pca = PCA(4)
	
	trainingData = df.select("count_vectors", "sentiment").collect()
	
	X_train = np.array(list(map(lambda row: row.count_vectors, trainingData)))
	y_train = np.array(list(map(lambda row: row.sentiment, trainingData)), dtype='int64')

	# Transform X_train
	X_train = pca.fit_transform(X_train)
	
	# Fit the LR classifier
	classifier.partial_fit(X_train, y_train, classes=np.unique(y_train))

	predictions = classifier.predict(X_train)
	
	accuracy = np.count_nonzero(np.array(predictions) == y_train) / y_train.shape[0]

	print("Accuracy of LR:", accuracy)
	
	return classifier
	
