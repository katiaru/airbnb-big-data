from sklearn import neural_network
from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor


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

def decision_tree(train_features):
    df = train_features
    dt = DecisionTreeRegressor(labelCol="label", featuresCol="features")
    dt_model = dt.fit(df)
    return dt_model