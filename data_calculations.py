from pyspark.sql.functions import size, col, split, mean


def calculate_dataset(listings):
    mean_review_scores_rating = listings.select([mean('review_scores_rating')]).collect()[0][0]
    listings = listings.filter(listings.review_scores_rating > mean_review_scores_rating)
    df1 = amenities_count(listings)
    df2 = verifications_count(df1)
    return df2


def neighbourhood_count(listings):
    df_count = listings.groupBy('neighbourhood_cleansed').count()
    df_neighbourhood = listings.join(df_count, "neighbourhood_cleansed")
    return df_neighbourhood.withColumnRenamed('count', 'neighbourhood_count')


def amenities_count(listings):
    return listings.withColumn('amenities_count', size(split(col("amenities"), r"\,"))).drop('amenities')


def verifications_count(listings):
    return listings.withColumn('verifications_count', size(split(col("host_verifications"), r"\,"))).drop('host_verifications')

def amenities_total_list(listings):
    org = listings.withColumn('amenities_count', split(col("amenities"), r"\,")).rdd.map(lambda x: (x.id, x.amenities_count)).flatMapValues(lambda x: x)
    print(str(org.collect()[0]).encode('utf-8'))

