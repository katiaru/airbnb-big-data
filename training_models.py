from sklearn import neural_network
from pyspark.ml.regression import RandomForestRegressor


def neural_net(train_features, train_labels):
    classifier_nn = neural_network.MLPClassifier(solver='sgd', activation='logistic', random_state=1,
                                                 early_stopping=True, learning_rate_init=0.2)
    return classifier_nn.fit(train_features, train_labels)

'''
featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1,
 minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='variance', subsamplingRate=1.0,
  seed=None, numTrees=20, featureSubsetStrategy='auto'
'''
def random_forest(train_features):
    df = train_features
    rf = RandomForestRegressor(labelCol="label", featuresCol="features", maxDepth=29, numTrees=50)
    rf_model = rf.fit(df)
    return rf_model
