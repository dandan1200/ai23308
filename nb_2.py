import numpy as np
import pandas as pd
from io import StringIO

class NBModel:
    def __init__(self, epsilon=1e-8):
        self.standard_deviations = {}
        self.averages = {}

        self.past_probs = {}
        self.epsilon_value = epsilon
    
    def fit(self, x, y):
        features = np.unique(y)

        for feature in features:
            x_label = x[y == feature]
            self.averages[feature] = np.mean(x_label, axis=0)
            self.standard_deviations[feature] = np.std(x_label, axis=0) + self.epsilon_value
            self.past_probs[feature] = len(x_label) / len(x)
    
    def predict(self, X):
        predictions = []
        for x in X:
            probabilities = {}
            for avg in self.averages:
                log_prob = -0.5 * np.sum(np.log(2 * np.pi * self.standard_deviations[avg] ** 2) + ((x - self.averages[avg]) ** 2) / (self.standard_deviations[avg] ** 2))
                probabilities[avg] = np.log(self.past_probs[avg]) + log_prob

            prediction = max(probabilities, key=probabilities.get)
            predictions.append(prediction)
        return predictions
    
    def test(self, x):
        predictions = self.predict(x.iloc[:, :-1].values)
        correct = []
        for i,prediction in enumerate(predictions):
            if prediction == x.iloc[i,-1]:
                correct.append(prediction)
        
        return len(correct)/len(predictions)

def classify_nb(training_filename, testing_filename):
    # Read the training data from a file
    testing_data = pd.read_csv(testing_filename, header=None)
    x_test = testing_data.values

    train_data = pd.read_csv(training_filename, header=None)
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    model = NBModel()

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    return predictions

def get_data(filename):
    with open(filename, "r") as f:
        data = f.readlines()

    folds = []

    fold = []
    for line in data:
        line = line.strip()
        
        if line:
            if line.startswith("fold") == False:
                fold.append(line)
        else:
            folds.append(fold)
            fold = []

    folds.append(fold)

    return folds

def test_nb(filename):
    data = get_data(filename)

    #

    data = get_data(filename)

    accuracies = []

    for fold in data:
        test_data = pd.read_csv(StringIO("\n".join(fold)), header=None)

        train_folds = []
        for train_fold in data:
            if train_fold != fold:
                train_folds.append(train_fold)

        train_folds_combined = [line for lines in train_folds for line in lines]
        
        #print(len(train_folds_combined))
        train_data = pd.read_csv(StringIO("\n".join(train_folds_combined)), header=None)

        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        classifier = NBModel()
        classifier.fit(x_train, y_train)

        accuracies.append(classifier.test(test_data))
    
    print(sum(accuracies)/len(accuracies))

    print(sum(accuracies)/len(accuracies))

test_nb("folds.csv")
print("feature select")
test_nb("feature_folds.csv")