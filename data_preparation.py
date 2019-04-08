from pyspark.sql import SparkSession
from pyspark.sql import Row, DataFrame
from pyspark.rdd import RDD
from sklearn.externals import joblib
from listings_columns import listings_columns
from training_models import *
from all_cities import all_cities
from parse_csv import *
import argparse
from validate import *
from data_cleaning import transform_df_to_features_vector


def generate_model(city, ml, training_data):
    if ml == 'nn':
        print('Generating neural net model for city ' + city + ' ...')
        nn_model = neural_net(training_data, [])
        joblib.dump(nn_model, 'models/nn/' + city + '.joblib')
        print('Done! Saved as models/nn/' + city + '.joblib')
    elif ml == 'rf':
        print('Generating random forest model for city ' + city + ' ...')
        rf_model = random_forest(training_data)
        st = 'models/rf/' + city
        rf_model.write().overwrite().save(st)
        print('Done! Saved as models/rf/' + city)

def main():
    spark = init_spark()
    # Usage: python3 data_preparation.py (train|validate|test) (nn|rf)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('request', help='train or test')
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
        data = parse_and_split(spark, city_name)
        data = transform_df_to_features_vector(data)
        (trainingData, testData) = data.randomSplit([0.8, 0.2])
        if request_type == 'train':
            print("Generating model for " + city_name + " using " + ml)
            generate_model(city_name, ml, trainingData)
            print("Validating for " + city_name + " using " + ml)
            validate_saved_model(city_name, ml, testData)
        elif request_type == 'test':
            print("Testing for " + city_name + " using " + ml)
            to_predict = parse_and_split(spark, 'test.csv')
            to_predict = transform_df_to_features_vector(to_predict)
            test_on_model(city_name, ml, to_predict)
        else:
            print('You must choose between (train|validate|test) as request type in the first argument! \nExiting...')
            return

    else:
        print('Invalid city name: ' + city_name)

main()