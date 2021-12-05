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
from sklearn.decomposition import PCA, TruncatedSVD

import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({'figure.figsize':(16, 9), 'figure.dpi':100})

def kmeans_clustering(X, y, spark, kmeans_model, num_iters):

	svd = TruncatedSVD(n_components=2)
	
	kmeans_model = kmeans_model.partial_fit(X)
	
	pred_labels = kmeans_model.predict(X)

	pred_labels_1 = [4 if i == 1 else 0 for i in pred_labels]
	pred_labels_2 = [0 if i == 1 else 4 for i in pred_labels]
	
	accuracy_1 = np.count_nonzero(np.array(pred_labels_1) == y) / pred_labels.shape[0]
	accuracy_2 = np.count_nonzero(np.array(pred_labels_2) == y) / pred_labels.shape[0]
			
	print('Accuracy of KMeans: ', accuracy_1, '|', accuracy_2)
	with open('./model_accuracies/kmeans.txt', "a") as ma:
		ip=str(accuracy_1)+","+str(accuracy_2)+'\n'
		ma.write(ip)
	
	X_train = svd.fit_transform(X)

	figure, axis = plt.subplots(1, 2)
	axis[0].scatter(X_train[:, 0], X_train[:, 1], c=pred_labels)
	axis[0].set_title('KMeans Clusters')
	axis[0].set_xlabel('LDA1')
	axis[0].set_ylabel('LDA2')
	
	axis[1].scatter(X_train[:, 0], X_train[:, 1], c=y)
	axis[1].set_title('Original Labels')
	axis[1].set_xlabel('LDA1')
	axis[1].set_ylabel('LDA2')
	
	if not os.path.isdir('./cluster_plots'):
		os.mkdir('./cluster_plots')

	img_file = open("./cluster_plots/KMeans_Batch_" + str(num_iters), "wb+")
	plt.savefig(img_file)

	return kmeans_model
