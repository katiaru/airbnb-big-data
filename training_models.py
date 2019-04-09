from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

'''
featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1,
 minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='variance', subsamplingRate=1.0,
  seed=None, numTrees=20, featureSubsetStrategy='auto'
'''

def random_forest(train_features):
    df = train_features
    rf = RandomForestRegressor(labelCol="label", featuresCol="features")

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 10, 20, 29]) \
        .addGrid(rf.numTrees, [5, 10, 30, 40, 50]) \
        .build()
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=1)

    rf_model = crossval.fit(df)

    return rf_model

def decision_tree(train_features):
    df = train_features
    dt = DecisionTreeRegressor(labelCol="label", featuresCol="features")
    dt_model = dt.fit(df)
    return dt_model