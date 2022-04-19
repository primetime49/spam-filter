import numpy as np
from math import sqrt
from math import exp
from math import pi
import extras

class NaiveBayes:

    def __init__(self, dataset):
        self.dataset = dataset

    # separate data to array 2D list (list of spam and list of ham)
    def separate_data_by_class(self, X, y):
        res = [[],[]]

        for i in range(len(X)):
            res[int(y[i])].append(X[i])
            
        return res

    # According to the formula in the assignment instructions
    def calculate_gaussian(self, x, mean, var):
        exponent = exp(-((x-mean)**2 / (2 * var)))
        return (1 / (sqrt(2 * pi * var))) * exponent

    # represent features (each 54) in [mean, variance]
    def get_features_data(self, X):
        res = [(self.mean(feature), self.var(feature)) for feature in zip(*X)]
        return res

    # split the above representation by spam and ham
    def get_features_data_by_class(self, X, y):
        data = self.separate_data_by_class(X, y)
        summaries = []

        for d in data:
            summaries.append(self.get_features_data(d))

        return summaries

    def mean(self, values):
        return sum(values) / float(len(values))

    def var(self, values):
        average = self.mean(values)
        variance = sum([(x - average) ** 2 for x in values]) / float(len(values) - 1)
        # to prevent ZeroDivisionError
        if variance == 0:
            return 1
        else:
            return variance

    # calculate probabilities of an email belonging to spam or ham
    def calculate_prob(self, data, email, X_train, y_train):
        res = []
        for d in data:
            score = 1
            for i in range(len(d)):
                feature = d[i]
                score *= self.calculate_gaussian(email[i], feature[0], feature[1])
            res.append(score)
        
        prob  = []
        for t in range(len(res)):
            score = 0
            this = (res[t]*(np.count_nonzero(y_train==t)/len(X_train)))
            that = (res[(t+1)%2]*(np.count_nonzero(y_train==(t+1)%2)/len(X_train)))
            #there are few emails where both gaussian scores are 0
            try:
                score = this/(this+that)
            except ZeroDivisionError:
                score = 0.5
            prob.append(score)
        return prob

    # predict the email to be spam or ham, returning [spam/ham, "winning" probability]
    def predict_class(self, data, email, X_train, y_train):
        prob = self.calculate_prob(data, email, X_train, y_train)
        high = 0
        res = 0
        i = 0
        for p in prob:
            if p > high:
                high = p
                res = i
            i += 1
        return res, high

    def perform(self, k):
        incorrect_prob = 0
        incorrect_total = 0
        correct_prob = 0
        correct_total = 0
        acc = 0
        min_correct_prob = 1
        min_incorrect_prob = 1

        for k in range(0,k):
            target = self.dataset[:, 57]
            features = self.dataset[:, :54]
            X_train,y_train,X_test,y_test = extras.split_data(k,len(self.dataset),features,target)
            data = self.get_features_data_by_class(X_train, y_train)
            true = 0
            for i in range(len(X_test)):
                res,prob = self.predict_class(data, X_test[i], X_train, y_train)
                if res == y_test[i]:
                    correct_prob += prob
                    if prob < min_correct_prob and prob > 0.5:
                        min_correct_prob = prob
                    true += 1
                else:
                    incorrect_prob += prob
                    if prob < min_incorrect_prob and prob > 0.5:
                        min_incorrect_prob = prob
                    incorrect_total += 1
            acc += (true/len(X_test))
            correct_total += true
            print('Fold '+str(k+1)+': ' + str(round(true/len(X_test)*100,3)) + '%')

        print('-----')
        print('Accuracy: ' + str(round(acc/10*100,3)) + '%')
        print('Avg \"Winning Probability\" for incorrect predictions: ' + str(round(incorrect_prob/incorrect_total*100,3)) + '%')
        print('Lowest \"Winning Probability\" for incorrect predictions: ' + str(round(min_incorrect_prob*100,3)) + '%')
        print('Avg \"Winning Probability\" for correct predictions: ' + str(round(correct_prob/correct_total*100,3)) + '%')
        print('Lowest \"Winning Probability\" for correct predictions: ' + str(round(min_correct_prob*100,3)) + '%')
