from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

#vector assembler not required if we are dealing with 1 column
'''
# vector assembler transforms  a set of features into a single vector
assemble=VectorAssembler(inputCols=['SENTIMENT','TWEET'], outputCol='features')
#df  is the batch dataframe after preprocessing
assembled_data=assemble.transform(df)
'''

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
