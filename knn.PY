import numpy as np
import pandas as pd
import io
class KNNModel:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        results = []
        for n in x:
            distances = np.sqrt(np.sum((self.x_train - n) ** 2, axis=1))
            ks = np.argsort(distances)[:self.k]
            k_nearest = self.y_train[ks]
            unique_labels, counts = np.unique(k_nearest, return_counts=True)
            max_count = np.max(counts)
           
            if np.sum(counts == max_count) > 1:
                prediction = "yes"
            else:
                prediction = unique_labels[np.argmax(counts)]
                
            results.append(prediction)
        return results
    
    def test(self, x):
        predictions = self.predict(x.iloc[:, :-1].values)
        correct = []
        for i,prediction in enumerate(predictions):
            if prediction == x.iloc[i,-1]:
                correct.append(prediction)
        
        #print(len(correct)/len(predictions))
        return len(correct)/len(predictions)

def classify_nn(training_filename, testing_filename, k):
    
    train_data = pd.read_csv(training_filename, header=None)

    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    testing_data = pd.read_csv(testing_filename, header=None).values

    KNN_obj = KNNModel(k)

    KNN_obj.fit(x_train, y_train)

    return KNN_obj.predict(testing_data)

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


def classify_crossvalidation(filename, k):
   
    data = get_data(filename)

    accuracies = []

    for fold in data:
        test_data = pd.read_csv(io.StringIO("\n".join(fold)), header=None)

        train_folds = []
        for train_fold in data:
            if train_fold != fold:
                train_folds.append(train_fold)

        train_folds_combined = [line for lines in train_folds for line in lines]
        
        #print(len(train_folds_combined))
        train_data = pd.read_csv(io.StringIO("\n".join(train_folds_combined)), header=None)

        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        classifier = KNNModel(k)
        classifier.fit(x_train, y_train)

        accuracies.append(classifier.test(test_data))
    
    print(sum(accuracies)/len(accuracies))


print("1: ")
classify_crossvalidation("folds.csv", 1)

print("5:")
classify_crossvalidation("folds.csv", 5)

print("feature select")
print("1: ")
classify_crossvalidation("feature_folds.csv", 1)

print("5:")
classify_crossvalidation("feature_folds.csv", 5)
