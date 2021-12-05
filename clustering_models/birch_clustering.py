'''
MLSS: Machine Learning with Spark Streaming
Dataset: Tweet Sentiment Analysis
Submission by: Team BD_078_460_474_565
Course: Big Data, Fall 2021
'''

import os
import numpy as np
import plotext as plt
import termplotlib as tpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

def birch_clustering(df, spark, brc_model, num_iters):
	
	X_train = df.select('hashed_vectors').collect()
	
	X_train = np.array([row['hashed_vectors'] for row in X_train])
	X_train = np.reshape(X_train, (X_train.shape[0], -1))
	
	actual_labels = df.select('sentiment').collect()
	actual_labels = [row['sentiment'] for row in actual_labels]
	actual_labels = list(map(int, actual_labels))
	
	brc_model = brc_model.partial_fit(X_train)
	
	pred_labels = brc_model.predict(X_train)

	pred_labels_1 = [4 if i == 1 else 0 for i in pred_labels]
	pred_labels_2 = [0 if i == 1 else 4 for i in pred_labels]
	
	accuracy_1 = np.count_nonzero(np.array(pred_labels_1) == np.array(actual_labels)) / pred_labels.shape[0]
	accuracy_2 = np.count_nonzero(np.array(pred_labels_2) == np.array(actual_labels)) / pred_labels.shape[0]
			
	print('Accuracy of BRC: ', accuracy_1, '|', accuracy_2)

	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(X_train)

	figure, axis = plt.subplots(1, 2)
	axis[0].scatter(X_pca[:, 0], X_pca[:, 1], c=pred_labels)
	axis[0].set_title('Birch Clusters')
	axis[0].set_xlabel('PCA1')
	axis[0].set_ylabel('PCA2')
	
	axis[1].scatter(X_pca[:, 0], X_pca[:, 1], c=actual_labels)
	axis[1].set_title('Original Labels')
	axis[1].set_xlabel('PCA1')
	axis[1].set_ylabel('PCA2')
	
	if not os.path.isdir('./cluster_plots'):
		os.mkdir('./cluster_plots')

	img_file = open("./cluster_plots/Birch_Batch_" + str(num_iters), "wb+")
	plt.savefig(img_file)

	return brc_model
