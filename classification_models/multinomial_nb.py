'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''
import numpy as np

from pyspark.mllib.linalg import Vectors

def MultiNBLearning(X, y, spark, classifier, model_version):
	"""
	Perform online learning of a Multinomial Naive Bayes model with batches of data
	"""
	
	# Fit the MNB classifier
	classifier.partial_fit(X, y, classes=np.unique(y))

	predictions = classifier.predict(X)
	
	accuracy = np.count_nonzero(np.array(predictions) == y) / y.shape[0]

	print(f"Accuracy of NB_{model_version}:", accuracy)
	with open(f'./model_accuracies/mnb_{model_version}.txt', "a") as ma:
		ma.write(str(accuracy)+'\n')
	
	return classifier
	
