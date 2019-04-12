from pyspark.sql.functions import regexp_replace, col, split
from pyspark.ml.feature import StringIndexer, CountVectorizer
from pyspark.ml.feature import VectorAssembler
from data_calculations import neighbourhood_count, get_amenities_total_list, add_amenities_columns
from listings_columns import *


def clean_data(listings):
    df1 = handle_empty_fields(listings)
    df2 = remove_dollar_signs(df1)
    df3 = convert_column_types(df2)
    df4 = string_index(df3)
    #df5 = one_hot_vector(df4)

    return df4.na.drop()


def handle_empty_fields(listings):
    no_price = listings.na.drop(subset=['price'])
    no_fee_values = no_price.na.fill('0.00', ['cleaning_fee', 'security_deposit', 'guests_included'])
    return no_fee_values


def remove_dollar_signs(listings):
    price_col = listings.withColumn('new_price', regexp_replace(col('price'), '\$', '')).drop('price')\
                                .withColumnRenamed('new_price', 'price')
    security_dep = price_col.withColumn('new_deposit', regexp_replace(col('security_deposit'), '\$', ''))\
                                .drop('security_deposit').withColumnRenamed('new_deposit', 'security_deposit')
    cleaning_fee = security_dep.withColumn('new_cleaning_fee', regexp_replace(col('cleaning_fee'), '\$', ''))\
                                .drop('cleaning_fee').withColumnRenamed('new_cleaning_fee', 'cleaning_fee')
    extra_people = cleaning_fee.withColumn('new_extra_people', regexp_replace(col('extra_people'), '\$', ''))\
                                .drop('extra_people').withColumnRenamed('new_extra_people', 'extra_people')
    return extra_people


def convert_column_types(listings):
    latitude = listings.withColumn("latitude", listings["latitude"].cast("double"))
    longitude = latitude.withColumn("longitude", listings["longitude"].cast("double"))
    cleaning_fee = longitude.withColumn("cleaning_fee", listings["cleaning_fee"].cast("double"))
    security_deposit = cleaning_fee.withColumn("security_deposit", listings["security_deposit"].cast("double"))
    review_scores_rating = security_deposit.withColumn("review_scores_rating", listings["review_scores_rating"].cast("integer"))
    reviews_per_month = review_scores_rating.withColumn("reviews_per_month", listings["reviews_per_month"].cast("double"))
    accommodates = reviews_per_month.withColumn("accommodates", listings["accommodates"].cast("integer"))
    host_listings_count = accommodates.withColumn("host_listings_count", listings["host_listings_count"].cast("integer"))
    availability_30 = host_listings_count.withColumn("availability_30", listings["availability_30"].cast("integer"))
    availability_60 = availability_30.withColumn("availability_60", listings["availability_60"].cast("integer"))
    availability_90 = availability_60.withColumn("availability_90", listings["availability_90"].cast("integer"))
    availability_365 = availability_90.withColumn("availability_365", listings["availability_365"].cast("integer"))
    minimum_nights = availability_365.withColumn("minimum_nights", listings["minimum_nights"].cast("integer"))
    maximum_nights = minimum_nights.withColumn("maximum_nights", listings["maximum_nights"].cast("integer"))
    bathrooms = maximum_nights.withColumn("bathrooms", listings["bathrooms"].cast("double"))
    bedrooms = bathrooms.withColumn("bedrooms", listings["bedrooms"].cast("integer"))
    guests_included = bedrooms.withColumn("guests_included", listings["guests_included"].cast("integer"))
    extra_people = guests_included.withColumn("extra_people", listings["extra_people"].cast("double"))
    beds = extra_people.withColumn("beds", listings["beds"].cast("integer"))
    calculated_host_listings_count = beds.withColumn('calculated_host_listings_count', listings['calculated_host_listings_count'].cast('integer'))
    price = calculated_host_listings_count.withColumn("price", listings["price"].cast("double")).filter(calculated_host_listings_count['price'].isNotNull())

    return price
    
def string_index(listings):
    # listings = neighbourhood_count(listings)
    neighbourhood_cleansed_indexer = StringIndexer(inputCol="neighbourhood_cleansed", outputCol="neighbourhood_cleansed_index", handleInvalid = "keep").fit(listings).transform(listings)
    experiences_offered_indexer = StringIndexer(inputCol="experiences_offered", outputCol="experiences_offered_index", handleInvalid = "keep").fit(neighbourhood_cleansed_indexer).transform(neighbourhood_cleansed_indexer)
    room_type_indexer = StringIndexer(inputCol="room_type", outputCol="room_index", handleInvalid = "keep").fit(experiences_offered_indexer).transform(experiences_offered_indexer)
    bed_type_indexer = StringIndexer(inputCol="bed_type", outputCol="bed_index", handleInvalid = "keep").fit(room_type_indexer).transform(room_type_indexer)
    cancellation_indexer = StringIndexer(inputCol="cancellation_policy", outputCol="cancellation_index", handleInvalid = "keep").fit(bed_type_indexer).transform(bed_type_indexer)
    host_index = StringIndexer(inputCol="host_is_superhost", outputCol="host_index", handleInvalid = "keep").fit(cancellation_indexer).transform(cancellation_indexer)
    require_guest_phone_verification = StringIndexer(inputCol="require_guest_phone_verification", outputCol="guest_phone_verification", handleInvalid = "keep").fit(host_index).transform(host_index)
    require_guest_profile_picture = StringIndexer(inputCol="require_guest_profile_picture", outputCol="guest_profile_picture", handleInvalid = "keep").fit(require_guest_phone_verification).transform(require_guest_phone_verification)
    property_type = StringIndexer(inputCol="property_type", outputCol="property_type_index", handleInvalid = "keep").fit(require_guest_profile_picture).transform(require_guest_profile_picture)
    instant_bookable = StringIndexer(inputCol="instant_bookable", outputCol="instant_bookable_index", handleInvalid = "keep").fit(property_type).transform(property_type)
    is_business_travel_ready = StringIndexer(inputCol="is_business_travel_ready", outputCol="business_travel_index", handleInvalid = "keep").fit(instant_bookable).transform(instant_bookable)
    host_has_profile_pic = StringIndexer(inputCol="host_has_profile_pic", outputCol="host_has_profile_pic_index", handleInvalid = "keep").fit(is_business_travel_ready).transform(is_business_travel_ready)
    is_location_exact = StringIndexer(inputCol="is_location_exact", outputCol="is_location_exact_index", handleInvalid = "keep").fit(host_has_profile_pic).transform(host_has_profile_pic)
    host_identity_verified = StringIndexer(inputCol="host_identity_verified", outputCol="host_identity_verified_index", handleInvalid = "keep").fit(is_location_exact).transform(is_location_exact)

    neighbourhood_cleansed_indexer = host_identity_verified.drop('neighbourhood_cleansed')
    experiences_offered_dropped = neighbourhood_cleansed_indexer.drop('experiences_offered')
    room_type_dropped = experiences_offered_dropped.drop('room_type')
    bed_type_dropped = room_type_dropped.drop('bed_type')
    cancellation_policy_dropped = bed_type_dropped.drop('cancellation_policy')
    host_is_superhost_dropped = cancellation_policy_dropped.drop('host_is_superhost')
    require_guest_phone_verification_dropped = host_is_superhost_dropped.drop('require_guest_phone_verification')
    require_guest_profile_picture_dropped = require_guest_phone_verification_dropped.drop('require_guest_profile_picture')
    property_type_dropped = require_guest_profile_picture_dropped.drop('property_type')
    instant_bookable_dropped = property_type_dropped.drop('instant_bookable')
    is_business_travel_ready_dropped = instant_bookable_dropped.drop('is_business_travel_ready')
    host_has_profile_pic_dropped = is_business_travel_ready_dropped.drop('host_has_profile_pic')
    host_identity_verified_index_dropped = host_has_profile_pic_dropped.drop('host_identity_verified')

    return host_identity_verified_index_dropped

def transform_df_to_features_vector(train_features):
    assemblerInputs = columns_35
    amenities_total_list = get_amenities_total_list(train_features)

    train_features = add_amenities_columns(train_features, amenities_total_list)

    assemblerInputs = assemblerInputs + amenities_total_list
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

    df = assembler.transform(train_features)
    df = df.withColumn("label", train_features.price)
    return df

