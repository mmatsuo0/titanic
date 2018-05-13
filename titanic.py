#!/usr/bin/env python
# Python 3.6.4

import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
import numpy

train_file = "train.csv"
test_file = "test.csv"
feature_name = ["Pclass", "Sex", "SibSp", "Parch"]
target_name = "Survived"


def correct_data(data):
    c = pandas.get_dummies(data, columns=["Pclass", "Sex"])
    return c


def cross_validation(model, feature, target):
    scores = cross_val_score(model, feature, target)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {}".format(scores.mean()))


def learning_prediction(model, feature, target, test_feature):
    model.fit(feature, target)
    score = model.score(feature, target)
    print("Training set score: {}".format(score))

    p = model.predict(test_feature)
    return p


if __name__ == '__main__':
    train = pandas.read_csv(train_file)
    test = pandas.read_csv(test_file)

    train_c = correct_data(train[feature_name])
    test_c = correct_data(test[feature_name])

    m = RandomForestClassifier()
    scaler = RobustScaler()
    scaler.fit(train_c)
    train_c = scaler.transform(train_c)
    test_c = scaler.transform(test_c)
    cross_validation(m, train_c, train[target_name])
    prediction = learning_prediction(m, train_c,
                                     train[target_name], test_c)

    submission = pandas.DataFrame({"PassengerId": test["PassengerId"],
                                   "Survived": prediction})
    submission.to_csv('submission.csv', index=False)

    s = len(numpy.where(prediction == 1)[0])
    d = len(numpy.where(prediction == 0)[0])
    print("survived: {0}, deceased: {1}".format(s, d))
