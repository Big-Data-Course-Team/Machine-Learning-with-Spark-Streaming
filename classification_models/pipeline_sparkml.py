from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import NGram, VectorAssembler
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer, ChiSqSelector

from sklearn import linear_model


def custom_model_pipeline(df, inputCols = ["tweet", "sentiment"], n=3):
    
    # Feature transformers: Tokenizer, NGrams, CountVectorizer, IDF, VectorAssembler
    
    # Converts the input string to lowercase, splits by white spaces
    tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
    df = tokenizer.transform(df)								# Needs no saving
    df.head()
    
    # Create three cols for each transformer
    for i in range(1, n+1):
		
		# Converts the input string to an array of n-grams (space-separated string of words)
		ngrams = NGram(n=n, inputCol="words", outputCol="{0}_grams".format(i))
		df = ngrams.transform(df)					# Needs no saving

		# Extracts a vocab from the tweet set
		cv = CountVectorizer(vocabSize=2**14, inputCol="{0}_grams".format(i), outputCol="{0}_tf".format(i))
		df = cv.transform(cv)
		
		# Compute the IDF score given a set of tweets
	    idf = IDF(inputCol="{0}_tf".format(i), outputCol="{0}_tfidf".format(i), minDocFreq=5)
		df = cv.transform(idf)

	# Merges multiple columns into a vector column
	assembler = VectorAssembler(inputCols=["{0}_tfidf".format(i) for i in range(1, n + 1)], outputCol="rawFeatures")
    
    label_stringIdx = StringIndexer(inputCol = "Sentiment", outputCol = "label")
    
    selector = ChiSqSelector(numTopFeatures=2**14,featuresCol='rawFeatures', outputCol="features")
    
    
    
def get_model():
	lr= linear_model.SGDClassifier()
	return lr

    
    
#To run (?)
'''
pipeline = model_pipeline()
model = pipeline.fit(train_set)
predictions = model.transform(val_set)
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

roc_auc = evaluator.evaluate(predictions)
# print accuracy, roc_auc
print ("Accuracy Score: {0:.4f}".format(accuracy))
print ("ROC-AUC: {0:.4f}".format(roc_auc))
'''

#To save and load
'''
pipeline.write().overwrite().save("./path")
pipeline = Pipeline.load("./path")
'''
