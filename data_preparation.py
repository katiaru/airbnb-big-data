from pyspark.sql import SparkSession
from pyspark.sql import Row, DataFrame
from pyspark.rdd import RDD
from sklearn.externals import joblib
from training_models import *
import argparse


listings_columns = ['id','number_of_reviews', 'latitude', 'longitude', 'amenities',
                    'security_deposit', 'cleaning_fee', 'neighbourhood_cleansed', 'bed_type', 'experiences_offered',
                    'host_verifications', 'review_scores_location', 'cancellation_policy', 'price', 'room_type',
                    'reviews_per_month', 'accommodates', 'review_scores_rating', 'host_is_superhost',
                    'host_listings_count', 'availability_30']#'first_review', 'last_review',

all_cities = ['amsterdam_listings.csv',          'broward_country_listings.csv',  'girona_listings.csv',
              'tasmania_listings.csv',           'antwerp_listings.csv',          'brussels_listings.csv',
              'rhode_island_listings.csv',       'thessaloniki_listings.csv',     'asheville_listings.csv',
              'nashville_listings.csv',          'rio_de_janerio_listings.csv',   'toronto_listings.csv',
              'hong_kong_listings.csv',          'new_orleans_listings.csv',      'rome_listings.csv',
              'austin_listings.csv',             'chicago_listings.csv',          'istanbul_listings.csv',
              'barcelona_listings.csv',          'clark_county_listings.csv',     'lisbon_listings.csv',
              'barossa_valley_listings.csv',     'columbus_listings.csv',         'london_listings.csv',
              'barwon_south_west_listings.csv',  'copenhagen_listings.csv',       'los_angeles_listings.csv',
              'beijing_listings.csv',            'denver_listings.csv',           'lyon_listings.csv',
              'bergamo_listings.csv',            'dublin_listings.csv',           'madrid_listings.csv',
              'berlin_listings.csv',             'edinburgh_listings.csv',        'malaga_listings.csv',
              'bologna_listings.csv',            'euskadi_listings.csv',          'mallorca_listings.csv',
              'bordeaux_listings.csv',           'florence_listings.csv',         'manchester_listings.csv',
              'boston_listings.csv',             'geneva_listings.csv',           'melbourne_listings.csv',
              'bristol_listings.csv',            'ghent_listings.csv',            'menorca_listings.csv',
              'milan_listings.csv',              'quebec_city_listings.csv',      'valencia_listings.csv',
              'greater_manchester_listings.csv',  'naples_listings.csv',          'vancouver_listings.csv',
              'cambridge_listings.csv',          'hawaii_listings.csv',           'vaud_listings.csv',
              'athens_listings.csv',             'cape_town_listings.csv',        'venice_listings.csv',
              'trentino_listings.csv',           'twin_cities_MSA_listings.csv',  'victoria_listings.csv',
              'new_york_listings.csv',           'salem_listings.csv',            'vienna_listings.csv',
              'northern_rivers_listings.csv',    'san_diego_listings.csv',        'washington_listings.csv',
              'oakland_listings.csv',            'san_francisco_listings.csv',    'western_australia_listings.csv',
              'oslo_listings.csv',               'santa_clara_country_listings.csv',
              'ottawa_listings.csv',             'santa_cruz_listings.csv',
              'pacific_grove_listings.csv',      'seattle_listings.csv',
              'paris_listings.csv',              'sevilla_listings.csv',
              'portland_listings.csv',           'sicily_listings.csv',
              'porto_listings.csv',              'stockholm_listings.csv',
              'prague_listings.csv',             'sydney_listings.csv',
              'puglia_listings.csv',             'taipei_listings.csv']


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def generate_model(city, ml):
    spark = init_spark()
    lines = spark.read.format("com.databricks.spark.csv") \
        .option("header", "True") \
        .option("multiLine", "True") \
        .option("delimiter", ",") \
        .option('quote', '"') \
        .option("quoteMode", "ALL") \
        .option('escape', '"') \
        .option("ignoreLeadingWhiteSpace", "True") \
        .option("ignoreTrailingWhiteSpace", "True") \
        .option("mode", "PERMISSIVE") \
        .option("wholeFile", "True")


    listings_csv = list(lines.load("./data/" + city).select(listings_columns).rdd.collect())

    split_index = int(len(listings_csv) * 0.8)
    training = listings_csv[:split_index]
    validation = listings_csv[split_index+1:]
    if ml == 'nn':
        print('Generating neural net model for city ' + city + ' ...')
        #nn_model = neural_net(training, listings_columns)
        #joblib.dump(nn_model, 'models/nn/' + city + '.joblib')
        print('Done! Saved as models/nn/' + city + '.joblib')


def main():
    # Usage: python3 data_preparation.py (train|validate|test) (nn|rf)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('request', help='train, validate or test')
    parser.add_argument('algo', help='nn (neural nets) or rf (random forest)')
    parser.add_argument('--help', action='help', help='show this help message and exit')

    args = parser.parse_args()

    request_type = args.request.lower()
    ml = args.algo.lower()
    if ml != 'nn' and ml != 'rf':
        print(ml)
        print('You must choose between neural networks (nn) or random forest (rf) as second argument for the machine learning algorithm to use! \nExiting...')
        return
    print("Options: \n")
    print(all_cities)
    city_name = input("Which city would you like to produce a model for?\n").lower()
    if city_name in all_cities:
        if request_type == 'train':
            print("Generating model for " + city_name + " using " + ml)
            generate_model(city_name, ml)
        elif request_type == 'validate':
            print("Validating for " + city_name + " using " + ml)
            #validate_model(city_name, ml)
        elif request_type == 'test':
            print("Testing for " + city_name + " using " + ml)
            #test_on_model(city_name, ml)
        else:
            print('You must choose between (train|validate|test) as request type in the first argument! \nExiting...')
            return

    else:
        print('Invalid city name: ' + city_name)

main()