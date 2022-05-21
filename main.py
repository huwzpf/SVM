from SVM_multiclass import SVM_multiclass
import pandas as pd

def main():

    data = pd.read_csv('D:/Projects/Data/data_logistic_1.csv').dropna()
    x = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    s = SVM_multiclass(x, y, 2, 0.1, 100)


if __name__ == '__main__':
    main()