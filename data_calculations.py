from pyspark.sql.functions import size, col, split, mean
import re


def calculate_dataset(listings):
    mean_review_scores_rating = listings.select([mean('review_scores_rating')]).collect()[0][0]
    above_mean_ratings = listings.filter(listings.review_scores_rating > mean_review_scores_rating).drop('review_scores_rating')
    reviews_per_month = above_mean_ratings.filter(above_mean_ratings.reviews_per_month > 0.5).drop('reviews_per_month')
    df1 = amenities_count(reviews_per_month)
    # df2 = verifications_count(df1)
    return df1


def neighbourhood_count(listings):
    df_count = listings.groupBy('neighbourhood_cleansed').count()
    df_neighbourhood = listings.join(df_count, "neighbourhood_cleansed")
    return df_neighbourhood.withColumnRenamed('count', 'neighbourhood_count')


def amenities_count(listings):
    return listings.withColumn('amenities_count', size(split(col("amenities"), r"\,")))


def verifications_count(listings):
    return listings.withColumn('verifications_count', size(split(col("host_verifications"), r"\,"))).drop('host_verifications')


def get_amenities_total_list(listings):
    amenities_total_list = listings.withColumn("amenities", split(col("amenities"), ",\s*").cast("array<string>")).rdd\
        .map(lambda x: (x.id, x.amenities)).flatMapValues(lambda x: x).map(lambda x: re.sub(r"{", "", x[1]))\
        .map(lambda x: re.sub(r"}", "", x)).map(lambda x: re.sub(r" +", "_", x)).map(lambda x: re.sub(r'"', '', x))\
        .distinct().filter(lambda x: x != '').filter(lambda x: 'translation_missing' not in x).collect()
    return amenities_total_list


def add_amenities_columns(listings, amenities_total_list):
    for amenity in amenities_total_list:
        if amenity != '':
            listings = listings.withColumn(amenity, col("amenities").contains(amenity))
    return listings

