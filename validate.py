from pyspark.ml.regression import RandomForestRegressionModel, DecisionTreeRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from  pyspark.ml.tuning import CrossValidatorModel


def validate_saved_model(city, ml_type, test_data):
    if ml_type == "rf":
        savedModel = CrossValidatorModel.load('models/' + ml_type + '/' + city)
        print(savedModel.bestModel)
        predictions = savedModel.transform(test_data)

        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    elif ml_type == "dt":
        # savedModel = DecisionTreeRegressionModel.load('models/' + ml_type + '/' + city)
        savedModel = CrossValidatorModel.load('models/' + ml_type + '/' + city)
        print(savedModel.bestModel)

        predictions = savedModel.transform(test_data)

        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


   

def test_on_model(city, ml_type, values):
    if ml_type == "rf":
        savedModel = CrossValidatorModel.load('models/' + ml_type + '/' + city)
        print(savedModel.bestModel.extractParamMap())
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(ExtractFeatureImp(savedModel.bestModel.featureImportances, values, 'features'))
        predictions = savedModel.transform(values)
        predictions.select("prediction", "label", "features").show(1)

    elif ml_type == "dt":
        # savedModel = DecisionTreeRegressionModel.load('models/' + ml_type + '/' + city)
        savedModel = CrossValidatorModel.load('models/' + ml_type + '/' + city)
        print(savedModel.bestModel.extractParamMap())
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(ExtractFeatureImp(savedModel.bestModel.featureImportances, values, 'features'))
        predictions = savedModel.bestModel.transform(values)
        predictions.select("prediction", "label", "features").show(1)

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    # inspired by a blogpost https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))