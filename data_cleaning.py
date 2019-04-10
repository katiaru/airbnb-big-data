from pyspark.sql.functions import regexp_replace, col, split
from pyspark.ml.feature import StringIndexer, CountVectorizer
from pyspark.ml.feature import VectorAssembler
from data_calculations import neighbourhood_count, get_amenities_total_list, add_amenities_columns


def clean_data(listings):
    df1 = handle_empty_fields(listings)
    df2 = remove_dollar_signs(df1)
    df3 = convert_column_types(df2)
    df4 = string_index(df3)
    #df5 = one_hot_vector(df4)

    return df4.na.drop()


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
    fix_price = price.filter(price['price'].isNotNull())
    return fix_price
    
def string_index(listings):
    listings = neighbourhood_count(listings)
    neighbourhood_cleansed_indexer = StringIndexer(inputCol="neighbourhood_cleansed", outputCol="neighbourhood_cleansed_index", handleInvalid = "keep").fit(listings).transform(listings)
    experiences_offered_indexer = StringIndexer(inputCol="experiences_offered", outputCol="experiences_offered_index", handleInvalid = "keep").fit(neighbourhood_cleansed_indexer).transform(neighbourhood_cleansed_indexer)
    room_type_indexer = StringIndexer(inputCol="room_type", outputCol="room_index", handleInvalid = "keep").fit(experiences_offered_indexer).transform(experiences_offered_indexer)
    bed_type_indexer = StringIndexer(inputCol="bed_type", outputCol="bed_index", handleInvalid = "keep").fit(room_type_indexer).transform(room_type_indexer)
    cancellation_indexer = StringIndexer(inputCol="cancellation_policy", outputCol="cancellation_index", handleInvalid = "keep").fit(bed_type_indexer).transform(bed_type_indexer)
    host_index = StringIndexer(inputCol="host_is_superhost", outputCol="host_index", handleInvalid = "keep").fit(cancellation_indexer).transform(cancellation_indexer)

    neighbourhood_cleansed_indexer = host_index.drop('neighbourhood_cleansed')
    experiences_offered_dropped = neighbourhood_cleansed_indexer.drop('experiences_offered')
    room_type_dropped = experiences_offered_dropped.drop('room_type')
    bed_type_dropped = room_type_dropped.drop('bed_type')
    cancellation_policy_dropped = bed_type_dropped.drop('cancellation_policy')
    host_is_superhost_dropped = cancellation_policy_dropped.drop('host_is_superhost')
    return host_is_superhost_dropped

def transform_df_to_features_vector(train_features):
    assemblerInputs = ['latitude', 'longitude', 'amenities_count', 'security_deposit', 'cleaning_fee',
                       'neighbourhood_cleansed_index', 'neighbourhood_count', 'bed_index', 'experiences_offered_index',
                       'verifications_count', 'cancellation_index', 'room_index',
                       'accommodates', 'host_index', 'host_listings_count', 'availability_30']

    amenities_total_list = get_amenities_total_list(train_features)

    train_features = add_amenities_columns(train_features, amenities_total_list)

    assemblerInputs = assemblerInputs + amenities_total_list

    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

    df = assembler.transform(train_features)
    df = df.withColumn("label", train_features.price)
    return df

