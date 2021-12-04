# Sentiment Analysis with Spark Streaming
### Current Repository Structure
```
.
├── classification_models
│   ├── logistic_regression.py
│   ├── multinomial_nb.py
├── clustering_models
│   ├── kmeans_clustering.py
├── Clusters
│   └── fig1
├── preprocessing
│   ├── preprocess.py
├── test.py
└── train.py
├── stream.py
├── sentiment
│   ├── test.csv
│   └── train.csv
├── model.pkl
├── num_iters
├── requirements.txt
├── README.md
├── LICENSE
```
### About the Dataset
1. Two CSV files each for training (with 1520k records) and testing (with 80k records).
2. Sentiment is either 0 (negative) or 4 (positive).
3. Each record consists of two features, the sentiment and the text.

### Task Workflow
1. Streaming the data
2. Processing each stream of data
3. Building the ML models
4. Testing the ML models
5. Clustering

