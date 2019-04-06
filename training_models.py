from sklearn import neural_network
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from listings_columns import *

def neural_net(train_features, train_labels):
    classifier_nn = neural_network.MLPClassifier(solver='sgd', activation='logistic', random_state=1,
                                                 early_stopping=True, learning_rate_init=0.2)
    return classifier_nn.fit(train_features, train_labels)


def random_forest(train_features, train_labels):
    print(str(train_features.collect()).encode('utf-8'))

    assemblerInputs = ['latitude', 'longitude', 'amenities_count', 'security_deposit', 'cleaning_fee',
                       'neighbourhood_cleansed_index', 'bed_index', 'experiences_offered_index',
                       'verifications_count', 'review_scores_location', 'cancellation_index', 'room_index',
                       'reviews_per_month', 'accommodates', 'review_scores_rating', 'host_index',
                       'host_listings_count', 'availability_30']

    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

    df = assembler.transform(train_features)
    df = df.withColumn("label", train_features.price)

    # randomly split data into training and test dataset
    (trainingData, testData) = df.randomSplit([0.8, 0.2])

    # train RandomForest model
    rf = RandomForestRegressor(labelCol="label", featuresCol="features")
    rf_model = rf.fit(trainingData)
    return rf_model
