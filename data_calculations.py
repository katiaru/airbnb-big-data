from pyspark.sql.functions import size, col, split


def calculate_dataset(listings):
    #df1 = neighbourhood_count(listings)
    df2 = amenities_count(listings)
    df3 = verifications_count(df2)
    return df3


def neighbourhood_count(listings):
    df_count = listings.groupBy('neighbourhood_cleansed').count()
    df_neighbourhood = listings.join(df_count, "neighbourhood_cleansed")
    return df_neighbourhood.withColumnRenamed('count', 'neighbourhood_count')


def amenities_count(listings):
    return listings.withColumn('amenities_count', size(split(col("amenities"), r"\,"))).drop('amenities')


def verifications_count(listings):
    return listings.withColumn('verifications_count', size(split(col("host_verifications"), r"\,"))).drop('host_verifications')
