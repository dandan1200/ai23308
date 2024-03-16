import numpy as np
import pandas as pd

def classify_nb(training_filename, testing_filename):
    train_data = pd.read_csv(training_filename, header=None)
    test_data = pd.read_csv(testing_filename, header=None)

    # implement naive bayes classifier for training
    yes_proba, no_proba = calc_yes_no_probabilty(train_data)
    feature_probabilties = calc_features(train_data)    
    
    preds = []
    print(train_data)
    for row in train_data:
        print(row)
        print(f"yes: {yes_proba} no: {no_proba}")
        for i, feature in enumerate(row):
            if feature == 1:
                yes_proba *= feature_probabilties[i]["yes"]
                no_proba *= feature_probabilties[i]["no"]
            else:
                yes_proba *= (1-feature_probabilties[i]["yes"])
                no_proba *= (1-feature_probabilties[i]["no"])

        print(f"yes: {yes_proba} no: {no_proba}")

        if no_proba > yes_proba:
            preds.append("no")
        else:
            preds.append("yes")

    return preds
    

def calc_yes_no_probabilty(data):
    
    total_yes = len(data[data.iloc[:, -1] == "yes"])
    total_no = len(data[data.iloc[:, -1] == "yes"])

    #Calculate yes probs
    yes_proba = total_yes/ len(data)

    #Calculate no probs
    no_proba = total_no/ len(data)

    return yes_proba, no_proba

def calc_features(data):
    feature_probabilties = []
    i = 0
    for feature in data.iloc[:, :-1]:
        # calculate yes feature probabilty
        feature_probabilties.append({})

        yes_data = data[data.iloc[:,-1] == "yes"]
        feature_data = yes_data[feature]
        feature_probability = feature_data.sum()/ len(yes_data)

        feature_probabilties[i]["yes"] = feature_probability


        #calculate no feature probabilty

        no_data = data[data.iloc[:,-1] == "no"]
        feature_data = no_data[feature]
        feature_probability = feature_data.sum()/ len(no_data)

        feature_probabilties[i]["no"] = feature_probability

        i+= 1
    
    return feature_probabilties
