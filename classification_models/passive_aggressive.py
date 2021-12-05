'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''
import numpy as np
from pyspark.mllib.linalg import Vectors

def PALearning(X, y, spark, classifier):
	"""
	Perform passive aggressive classification on the dataframe with incremental learning
	"""
	
	# Fit the pac classifier
	classifier.partial_fit(X, y, classes=np.unique(y))

	predictions = classifier.predict(X)
	
	accuracy = np.count_nonzero(np.array(predictions) == y)/y.shape[0]
	
	print("Accuracy of PAC:", accuracy)
	
	return classifier
