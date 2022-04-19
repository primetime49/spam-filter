import time
from NaiveBayes import NaiveBayes
from KNN import KNN
from SVM import SVM
import numpy as np

filename = "spambase.data"
file = open(filename, "r")
dataset = np.loadtxt(file, delimiter = ",")

#small tidbit just to make the dataset count even to 4600 (easier k-fold splitting)
dataset = dataset[:-1]

#randomized in the beginning so everyone have the same dataset
np.random.shuffle(dataset)

run = input('Wanna try with Naive Bayes? [y/n] ')
if run == 'y' or run == '':
    nb = NaiveBayes(dataset)
    print('------- NAIVE BAYES -------')
    start = time.time()
    nb.perform(10)
    print("Time elapsed: {} seconds".format(round(time.time() - start,4)))
    print('')

run = input('Wanna try with KNN? [y/n] ')
if run == 'y' or run == '':
    knn = KNN(dataset)
    print('------- K-NEAREST NEIGHBORS -------')
    start = time.time()
    knn.perform(10, 5)
    print("Time elapsed: {} seconds".format(round(time.time() - start,4)))
    print('')

run = input('Wanna try with SVM? [y/n] ')
if run == 'y' or run == '':
    svm = SVM(dataset)
    print('------- SVM -------')
    svm.perform(10, 'linear')
    svm.perform(10, 'poly', 2)
    svm.perform(10, 'rbf')
    svm.perform(10, 'linear', 3, True)
    svm.perform(10, 'poly', 2, True)
    svm.perform(10, 'rbf', 3, True)
