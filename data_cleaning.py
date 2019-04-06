from pyspark.sql.functions import regexp_replace, col
from pyspark.ml.feature import StringIndexer


def clean_data(listings):
    df1 = handle_empty_fields(listings)
    df2 = remove_dollar_signs(df1)
    df3 = string_index(df2)
    return list(df3.rdd.collect())


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
    
def string_index(listings):
    room_type_indexer = StringIndexer(inputCol="room_type", outputCol="room_index").fit(listings).transform(listings)
    bed_type_indexer = StringIndexer(inputCol="bed_type", outputCol="bed_index").fit(room_type_indexer).transform(room_type_indexer)
    cancellation_indexer = StringIndexer(inputCol="cancellation_policy", outputCol="cancellation_index").fit(bed_type_indexer).transform(bed_type_indexer)
    host_index = StringIndexer(inputCol="host_is_superhost", outputCol="host_index").fit(cancellation_indexer).transform(cancellation_indexer)
    return host_index
