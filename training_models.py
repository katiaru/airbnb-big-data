from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row


def neural_net(train_features, train_labels):
    classifier_nn = neural_network.MLPClassifier(solver='sgd', activation='logistic', random_state=1,
                                                 early_stopping=True, learning_rate_init=0.2)
    return classifier_nn.fit(train_features, train_labels)


def random_forest(train_features, train_labels):
    print(str(train_features.collect()[0]).encode('utf-8'))
    #classifier_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #return classifier_rf.fit(train_features, train_labels)
    #def transData(row):
        #return Row(label=row["price"], features=Vectors.dense([row["longitude"], row["latitude"], row["reviews_per_month"]]))
    def transData(data):
        return data.rdd.map(lambda r: [Vectors.dense(r[1:-2]), r[-1]]).toDF(['features', 'label'])
    transformed = transData(train_features)
    transformed.show(5)

    featureIndexer = \
        VectorIndexer(inputCol="amenities", outputCol="indexedFeatures", maxCategories=4).fit(train_features)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = train_features.randomSplit([0.8, 0.2])

    # Train a RandomForest model.
    rf = RandomForestRegressor(featuresCol="indexedFeatures")

    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, rf])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)
    return model
