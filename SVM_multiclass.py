import numpy as np
from SMO import SMO


class SVM_multiclass:
    def __init__(self, x, y, k, c, gamma):
        labels = np.negative(np.ones((len(y), k)))
        for i in range(len(y)):
            labels[i, y[i]] = 1

        self.svms = [SMO(x, labels[:, i], c, gamma) for i in range(k)]
        for svm in self.svms:
            svm.train(0.005, 50)

        print(f"Error rate on train set : {self.test(x, y, k)}")

    def test(self, x, y, k, prnt=False):
        count = 0
        outputs = np.zeros((len(y), k))
        results = np.zeros(len(y))
        for i in range(len(y)):
            for j in range(k):
                outputs[i, j] = self.svms[j].hypothesis(x[i, :])
            results[i] = np.argmax(outputs[i, :])
            if results[i] != y[i]:
                count += 1
            if prnt:
                print(results[i], y[i])
        return count/len(y)
