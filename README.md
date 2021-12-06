# Sentiment Analysis with Spark Streaming
### Current Repository Structure
```
.
├── Batch_1000
├── Batch_2000
├── Batch_2500
├── Batch_3000
├── Batch_4000
├── Batch_5000
├── classification_models
│   ├── logistic_regression.py
│   ├── multinomial_nb.py
│   ├── passive_aggressive.py
├── clustering_models
│   ├── kmeans_clustering.py
│   ├── birch_clustering.py│   
├── LICENSE
├── preprocessing
│   ├── preprocess.py
├── README.md
├── requirements.txt
├── test.py
└── train.py

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

