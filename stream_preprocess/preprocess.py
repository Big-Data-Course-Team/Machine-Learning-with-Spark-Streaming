from pyspark import SparkContext

# Main entry point for all streaming functionality
from pyspark.streaming import StreamingContext

# Create a local StreamingContext with two execution threads
sc = SparkContext("local[2]", "Sentiment")

# Batch interval of 1 second - TODO: Change according to necessity
ssc = StreamingContext(sc, 1)

# Set constant for the TCP port to send from/listen to
TCP_PORT = 6100

# Create a DStream - represents the stream of data received from TCP source/data server
# Each record in 'lines' is a line of text
lines = ssc.socketTextStream("localhost", TCP_PORT)

# The data is streamed as a JSON string (you can see this by observing the code in stream.py). 
# You will first need to parse the JSON string, obtain the rows in each batch and then convert it to a DataFrame. 
# The structure of the JSON string has been provided in the streaming file. 
# Use this to parse your JSON to obtain rows, and then convert these rows into a DataFrame.

# Start processing after all the transformations have been setup
ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

