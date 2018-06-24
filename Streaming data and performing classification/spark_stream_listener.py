from pyspark import SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
import time
from pyspark.ml import PipelineModel
from collections import namedtuple

# from pyspark.sql.functions import desc

sc = SparkContext("local[2]", "Tweet Streaming App")

ssc = StreamingContext(sc, 10)
sqlContext = SQLContext(sc)
ssc.checkpoint("file:/home/ubuntu/tweets/checkpoint/")

socket_stream = ssc.socketTextStream("ec2-18-216-228-96.us-east-2.compute.amazonaws.com",
                                     5555)  # Internal ip of  the tweepy streamer
# model = PipelineModel.load("logreg.model")
lines = socket_stream.window(20)
lines.pprint()

fields = ("text")
Tweet = namedtuple('Tweet', fields)


def getSparkSessionInstance(sparkConf):
    if ("sparkSessionSingletonInstance" not in globals()):
        globals()["sparkSessionSingletonInstance"] = SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]


def do_something(time, rdd):
    print("========= %s =========" % str(time))

    # Get the singleton instance of SparkSession
    spark = getSparkSessionInstance(rdd.context.getConf())
    model = PipelineModel.load("logreg.model")
    # Convert RDD[String] to RDD[Tweet] to DataFrame
    rowRdd = rdd.map(lambda w: Tweet(w))
    linesDataFrame = spark.createDataFrame(rowRdd)
    linesDataFrame.show()
    # Creates a temporary view using the DataFrame
    linesDataFrame.createOrReplaceTempView("tweets")
    predictions = model.transform(linesDataFrame)
    predictions.show()
    results = predictions.select([c for c in predictions.columns if c in ["text", "prediction"]])
    results.show()
    # Do tweet character count on table using SQL and print it
    # lineCountsDataFrame = spark.sql("select text, length(text) as TextLength from tweets")
    # lineCountsDataFrame.show()
    results.coalesce(1).write.mode("overwrite").format("com.databricks.spark.csv").save("tweets_csv")


lines.foreachRDD(do_something)

ssc.start()
# tweets_data = sqlContext.sql("SELECT * FROM tweets")
# tweets_data.toPandas().to_csv("tweets_data.csv")
ssc.awaitTerminationOrTimeout(100)
ssc.stop()
