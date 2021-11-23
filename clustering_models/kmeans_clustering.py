from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans
    
 
#stream of vectors for parsing (test data)
'''def parse(lp):
    label = float(lp[lp.find('(') + 1: lp.find(')')])
    vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))

    return LabeledPoint(label, vec)
 '''   


#train data format

#data point should be formatted as (y, [x1, x2, x3])
# test data format
#(1.0), [1.7, 0.4, 0.9]
#(4.0), [2.2, 1.8, 0.0]


#input training data should be formatted as [x1, x2, x3](each training point)
#0.0 0.0 0.0
#0.1 0.1 0.1
#0.2 0.2 0.2

#input stream should be made stream of vectors for training
#trainingData = sc.textFile("kmeans_data.txt")\
 #   .map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
 
#testingData = sc.textFile("streaming_kmeans_data_test.txt").map(parse)

# Creating a model with random clusters and specify the number of clusters to find
model = StreamingKMeans(k=2, decayFactor=0.7).setRandomCenters(2, 1.0, 0)
model.trainOn(trainingData)
#resu=model.predictOnValues(testData.map(lambda lp: (lp.label, lp.features)))


# printing the predicted cluster assignments on new data points as they arrive.
result = model.predictOnValues(testdata.map(lambda lp: (lp.label, lp.features)))
result.pprint()



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
