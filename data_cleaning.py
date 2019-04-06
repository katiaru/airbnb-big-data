from pyspark.sql.functions import regexp_replace, col

def clean_data(listings):
    df1 = handle_empty_fields(listings)
    df2 = remove_dollar_signs(df1)
    df3 = convert_column_types(df2)
    return df3


def handle_empty_fields(listings):
    no_price = listings.na.drop(subset=['price'])
    no_fee_values = no_price.na.fill('0.00', ['cleaning_fee', 'security_deposit'])
    return no_fee_values

def remove_dollar_signs(listings):
    price_col = listings.withColumn('new_price', regexp_replace(col('price'), '\$', '')).drop('price')\
                                .withColumnRenamed('new_price', 'price')
    security_dep = price_col.withColumn('new_deposit', regexp_replace(col('security_deposit'), '\$', ''))\
                                .drop('security_deposit').withColumnRenamed('new_deposit', 'security_deposit')
    cleaning_fee = security_dep.withColumn('new_cleaning_fee', regexp_replace(col('cleaning_fee'), '\$', ''))\
                                .drop('cleaning_fee').withColumnRenamed('new_cleaning_fee', 'cleaning_fee')
    return cleaning_fee


def convert_column_types(listings):
    latitude = listings.withColumn("latitude", listings["latitude"].cast("double"))
    longitude = latitude.withColumn("longitude", listings["longitude"].cast("double"))
    cleaning_fee = longitude.withColumn("cleaning_fee", listings["cleaning_fee"].cast("double"))
    security_deposit = cleaning_fee.withColumn("security_deposit", listings["security_deposit"].cast("double"))
    review_scores_location = security_deposit.withColumn("review_scores_location", listings["review_scores_location"].cast("integer"))
    reviews_per_month = review_scores_location.withColumn("reviews_per_month", listings["reviews_per_month"].cast("double"))
    accommodates = reviews_per_month.withColumn("accommodates", listings["accommodates"].cast("integer"))
    review_scores_rating = accommodates.withColumn("review_scores_rating", listings["review_scores_rating"].cast("integer"))
    host_listings_count = review_scores_rating.withColumn("host_listings_count", listings["host_listings_count"].cast("integer"))
    availability_30 = host_listings_count.withColumn("availability_30", listings["availability_30"].cast("integer"))
    price = availability_30.withColumn("price", listings["price"].cast("double"))
    return price
