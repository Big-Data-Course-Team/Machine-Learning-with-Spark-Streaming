'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''
import numpy as np

from pyspark.mllib.linalg import Vectors

def MultiNBLearning(X, y, spark, classifier):
	"""
	Perform online learning of a Multinomial Naive Bayes model with batches of data
	"""
	
	# Fit the MNB classifier
	classifier.partial_fit(X, y, classes=np.unique(y))

	predictions = classifier.predict(X)
	
	accuracy = np.count_nonzero(np.array(predictions) == y) / y.shape[0]

	print("Accuracy of NB:", accuracy)
	with open('./model_accuracies/mnb.txt', "a") as ma:
		ma.write(str(accuracy)+'\n')
	
	return classifier
	
