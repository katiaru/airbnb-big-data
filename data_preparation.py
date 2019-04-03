from pyspark.sql import SparkSession
from pyspark.sql import Row, DataFrame
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
    lines = spark.read.format("com.databricks.spark.csv")\
                        .option("header", "True")\
                        .option("multiLine", "True")\
                        .option("delimiter", ",")\
                        .option('quote', '"')\
                        .option("quoteMode", "ALL") \
                        .option('escape', '"')\
                        .option("ignoreLeadingWhiteSpace", "True") \
                        .option("ignoreTrailingWhiteSpace", "True") \
                        .option("mode", "PERMISSIVE") \
                        .option("wholeFile", "True")
    listings_csv = lines.load("./data/listings.csv").rdd.map(lambda r: r.id).collect()

    print(str(listings_csv).encode('utf-8'))


main()