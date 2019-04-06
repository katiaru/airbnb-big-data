from pyspark.sql.functions import regexp_replace, col

def clean_data(listings):
    df1 = handle_empty_fields(listings)
    df2 = remove_dollar_signs(df1)
    return df2


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