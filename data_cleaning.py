from pyspark.sql.functions import regexp_replace, col


def clean_data(listings):
    df1 = remove_dollar_signs(listings)
    return list(df1.rdd.collect())


def remove_dollar_signs(listings):
    price_col = listings.select('price', regexp_replace(col('price'), '\$', '').alias('new_price'))
    df_price = listings.join(price_col, 'price').drop('price')
    return df_price.withColumnRenamed('new_price', 'price')