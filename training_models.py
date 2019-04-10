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
        .addGrid(rf.maxDepth, [2, 5, 20, 29]) \
        .addGrid(rf.numTrees, [5, 10, 40, 50]) \
        .addGrid(rf.numTrees, [32, 190]) \
        .addGrid(rf.maxBins, [150, 200]) \
        .build()
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)

    rf_model = crossval.fit(df)

    return rf_model

def decision_tree(train_features):
    df = train_features
    dt = DecisionTreeRegressor(labelCol="label", featuresCol="features")

    dtparamGrid = (ParamGridBuilder().addGrid(dt.maxDepth, [2, 5, 10, 20, 30])
                                    .addGrid(dt.maxBins, [150, 200])
                                    .build())
    
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")

    crossval = CrossValidator(estimator = dt,
                              estimatorParamMaps = dtparamGrid,
                              evaluator = evaluator,
                              numFolds=3)
    dt_model = crossval.fit(df)
    return dt_model