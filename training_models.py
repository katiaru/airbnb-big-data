from sklearn import neural_network


def neural_net(train_features, train_labels):
    classifier_nn = neural_network.MLPClassifier(solver='sgd', activation='logistic', random_state=1,
                                                 early_stopping=True, learning_rate_init=0.2)
    return classifier_nn.fit(train_features, train_labels)