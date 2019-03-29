from pyspark.sql import SparkSession
from pyspark.rdd import RDD


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def main():
    spark = init_spark()
    listings_csv = spark.read.text("./data/listings.csv").rdd.collect()
    print(str(listings_csv).encode('utf-8'))


main()