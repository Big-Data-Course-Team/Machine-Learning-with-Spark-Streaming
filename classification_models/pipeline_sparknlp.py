# Call general Spark NLP transformers and concepts
from sparknlp.base import *
# Call all annotators provided by Spark NLP
from sparknlp.annotator import *
from pyspark.ml import Pipeline


# Get raw data annotated - of type Document
documentAssembler = DocumentAssembler() \
    .setInputCol("Tweet") \
    .setOutputCol("document")


# Outputs results onto an array
finisher = Finisher() \
    .setInputCols(["token"]) \
    .setIncludeMetadata(True)
    
# Setup the pipeline
pipeline = Pipeline() \
    .setStages([
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        finisher
    ])
