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

columns = ['id','number_of_reviews', 'first_review', 'last_review', 'latitude', 'longitude', 'amenities',
           'security_deposit', 'cleaning_fee', 'neighbourhood_cleansed', 'bed_type', 'experiences_offered',
           'host_verifications', 'review_scores_location', 'cancellation_policy', 'price', 'room_type',
           'reviews_per_month', 'accommodates', 'review_scores_rating', 'host_is_superhost', 'host_listings_count', 'availability_30']
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
    listings_csv = lines.load("./data/listings.csv").select(columns).rdd.collect()

    print(str(listings_csv).encode('utf-8'))


main()