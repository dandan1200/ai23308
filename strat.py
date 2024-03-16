import pandas as pd

def strat_cross_val(filename):
    data = pd.read_csv(filename)

    folds = []

    yes = data[data.iloc[:,-1] == "yes"]
    no = data[data.iloc[:,-1] == "no"]

    num_yes = len(yes)
    num_no = len(no)

    yes_per_fold = num_yes//10
    no_per_fold = num_no//10

    for i in range(10):
        fold = []
        for j in range(yes_per_fold):
            fold.append(list(yes.iloc[j + (i* yes_per_fold),:]))

        for j in range(no_per_fold):
            fold.append(list(no.iloc[j + (i* no_per_fold),:]))
        folds.append(fold)
    
    ret_str = ""
    for i in range(10):
        ret_str += "fold" + str(i) + "\n"
        for j in range(len(folds[i])):
            ret_str += ",".join(list(folds[i][j]))
            ret_str += "\n"

        ret_str += "\n\n"

    return ret_str

if __name__ == "__main__":
    print(strat_cross_val("normalised.csv"))