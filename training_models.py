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
    print(str(train_features.collect()[0]).encode('utf-8'))
    #classifier_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #return classifier_rf.fit(train_features, train_labels)
    #def transData(row):
        #return Row(label=row["price"], features=Vectors.dense([row["longitude"], row["latitude"], row["reviews_per_month"]]))
    def get_dummy(df, indexCol, categoricalCols, continuousCols, labelCol):

        indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid="skip")
                    for c in categoricalCols]

        # default setting: dropLast=True
        encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                                  outputCol="{0}_encoded".format(indexer.getOutputCol()))
                    for indexer in indexers]

        assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                              + continuousCols, outputCol="features")

        pipeline = Pipeline(stages=indexers + encoders + [assembler])

        model = pipeline.fit(df)
        data = model.transform(df)

        data = data.withColumn('label', col(labelCol))

        return data.select(indexCol, 'features', 'label')
    def transData(data):
        return data.rdd.map(lambda r: [Vectors.dense(r[1:-2]), r[-1]]).toDF(['features', 'label'])

    dummy = get_dummy(train_features, 'id', categorical_columns, continuous_columns, 'price')
    transformed = transData(dummy)
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
