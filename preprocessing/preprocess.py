def preprocessing(lines):
    #words = lines.select(explode(split(lines.value, "t_end")).alias("word"))
    tokenizer = Tokenizer(outputCol="words")
	tokenizer.setInputCol("tweet")
	words = tokenizer.transform(lines)
    words = words.na.replace('', None)
    words = words.na.drop()
    words = words.withColumn('word', F.regexp_replace('word', r'http\S+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '@\w+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '#', ''))
    words = words.withColumn('word', F.regexp_replace('word', 'RT', ''))
    words = words.withColumn('word', F.regexp_replace('word', ':', ''))
    return words
