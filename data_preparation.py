from pyspark.sql import SparkSession
from pyspark.sql import Row, DataFrame
from pyspark.rdd import RDD
from sklearn.externals import joblib
from listings_columns import listings_columns
from training_models import *
from all_cities import all_cities
from parse_csv import parse_and_split
import argparse


def generate_model(city, ml, training_data):
    if ml == 'nn':
        print('Generating neural net model for city ' + city + ' ...')
        print(training_data)
        nn_model = neural_net(training_data, [])
        joblib.dump(nn_model, 'models/nn/' + city + '.joblib')
        print('Done! Saved as models/nn/' + city + '.joblib')
    elif ml == 'rf':
        print('Generating random forest model for city ' + city + ' ...')
        rf_model = random_forest(training_data, listings_columns)
        joblib.dump(rf_model, 'models/rf/' + city + '.joblib')
        print('Done! Saved as models/rf/' + city + '.joblib')

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
        data = parse_and_split(city_name)
        if request_type == 'train':
            print("Generating model for " + city_name + " using " + ml)
            generate_model(city_name, ml, data)
        elif request_type == 'validate':
            print("Validating for " + city_name + " using " + ml)
            #validate_model(city_name, ml, data)
        elif request_type == 'test':
            print("Testing for " + city_name + " using " + ml)
            #test_on_model(city_name, ml)
        else:
            print('You must choose between (train|validate|test) as request type in the first argument! \nExiting...')
            return

    else:
        print('Invalid city name: ' + city_name)

main()