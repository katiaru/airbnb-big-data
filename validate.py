from pyspark.ml.regression import RandomForestRegressionModel, DecisionTreeRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator

from  pyspark.ml.tuning import CrossValidatorModel


def validate_saved_model(city, ml_type, test_data):
    if ml_type == "rf":

        #savedModel = RandomForestRegressionModel.load('models/' + ml_type + '/' + city)
        savedModel = CrossValidatorModel.load('models/' + ml_type + '/' + city)
        print(savedModel.bestModel)
        predictions = savedModel.bestModel.transform(test_data)

        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    elif ml_type == "dt":
        savedModel = DecisionTreeRegressionModel.load('models/' + ml_type + '/' + city)

        predictions = savedModel.transform(test_data)

        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


   

def test_on_model(city, ml_type, values):
    if ml_type == "rf":
        #savedModel = RandomForestRegressionModel.load('models/' + ml_type + '/' + city)
        savedModel = CrossValidatorModel.load('models/' + ml_type + '/' + city)
        print(savedModel.bestModel.extractParamMap())
        print(savedModel.bestModel.featureImportances)
        predictions = savedModel.bestModel.transform(values)
        predictions.select("prediction", "label", "features").show(1)

    elif ml_type == "dt":
        savedModel = DecisionTreeRegressionModel.load('models/' + ml_type + '/' + city)
        predictions = savedModel.transform(values)
        predictions.select("prediction", "label", "features").show(1)
