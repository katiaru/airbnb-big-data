from pyspark.sql import SparkSession
from listings_columns import listings_columns
from data_calculations import calculate_dataset
from data_cleaning import clean_data


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def parse_and_split(spark, city):
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
    listings_csv = calculate_dataset(clean_data(lines.load("./data/" + city).select(listings_columns)))
    return listings_csv
