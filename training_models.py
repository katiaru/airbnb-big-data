from sklearn import neural_network
from pyspark.ml.regression import RandomForestRegressor


def neural_net(train_features, train_labels):
    classifier_nn = neural_network.MLPClassifier(solver='sgd', activation='logistic', random_state=1,
                                                 early_stopping=True, learning_rate_init=0.2)
    return classifier_nn.fit(train_features, train_labels)


def random_forest(train_features):
    df = train_features
    rf = RandomForestRegressor(labelCol="label", featuresCol="features")
    rf_model = rf.fit(df)
    return rf_model
