from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier


def neural_net(train_features, train_labels):
    classifier_nn = neural_network.MLPClassifier(solver='sgd', activation='logistic', random_state=1,
                                                 early_stopping=True, learning_rate_init=0.2)
    return classifier_nn.fit(train_features, train_labels)


def random_forest(train_features, train_labels):
    classifier_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

    return classifier_rf.fit(train_features, train_labels)