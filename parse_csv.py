from pyspark.sql import SparkSession
from listings_columns import listings_columns

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def parse_and_split(city):
    spark = init_spark()
    lines = spark.read.format("com.databricks.spark.csv") \
        .option("header", "True") \
        .option("multiLine", "True") \
        .option("delimiter", ",") \
        .option('quote', '"') \
        .option("quoteMode", "ALL") \
        .option('escape', '"') \
        .option("ignoreLeadingWhiteSpace", "True") \
        .option("ignoreTrailingWhiteSpace", "True") \
        .option("mode", "PERMISSIVE") \
        .option("wholeFile", "True")

    listings_csv = lines.load("./data/" + city).select(listings_columns)

    #split_index = int(len(listings_csv) * 0.8)
    #training = listings_csv[:split_index]
    #validation = listings_csv[split_index + 1:]

    return listings_csv