import numpy as np
from time import time
from sklearn.svm import SVC
import extras
import time

class SVM:
    def __init__(self, dataset):
        self.dataset = dataset

    def tfidf(self, X):
        n = X.shape[0]
        idf = np.log10(n/(X != 0).sum(0))
        return X*idf # X is tf

    def transform(self, X):
        # squared, and add small integers to prevent ZeroDivisionError
        sq = np.sqrt(((X+1e-100)**2).sum(axis=1, keepdims=True))
        res = np.where(sq > 0.0, X / sq, 0.)
        return res
    
    def perform(self, k, kernel, degree=3, transform_kernels=False):
        start = time.time()
        print('Kernel: '+kernel+' with degree '+str(degree))
        print('Transformed kernels? '+str(transform_kernels))
        acc = 0
        for k in range(0,k):
            target = self.dataset[:, 57]
            features = self.dataset[:, :54]
            
            #over TF/IDF representation
            features = self.tfidf(features)

            #transform the kernel to use angular information only
            if transform_kernels:
                features = self.transform(features)

            X_train,y_train,X_test,y_test = extras.split_data(k,len(self.dataset),features,target)
            clf = SVC(kernel=kernel, degree=degree, C=1.0)
            clf.fit(X_train,y_train)
            res = clf.score(X_test,y_test)

            acc += res
            print('Fold '+str(k+1)+': ' + str(round(res*100,3)) + '%')
        print('-----')
        print('Accuracy: ' + str(round(acc/10*100,3)) + '%')

        print("Time elapsed: {} seconds".format(round(time.time() - start,4)))
        print('')
    